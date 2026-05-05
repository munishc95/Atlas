from __future__ import annotations

import argparse
from datetime import date, datetime
import io
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd
import requests
from sqlmodel import Session

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import engine, init_db  # noqa: E402
from app.services.corporate_actions import import_corporate_actions  # noqa: E402
from app.services.data_store import DataStore  # noqa: E402

NSE_PAGE_URL = "https://www.nseindia.com/companies-listing/corporate-filings-actions"
NSE_ACTIONS_URL = "https://www.nseindia.com/api/corporates-corporateActions"


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )


def _headers(accept: str) -> dict[str, str]:
    return {
        "Accept": accept,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": NSE_PAGE_URL,
        "Accept-Language": "en-US,en;q=0.9",
    }


def _fetch_nse_actions_csv(*, start_date: date, end_date: date) -> bytes:
    session = requests.Session()
    session.get(NSE_PAGE_URL, headers=_headers("text/html,*/*"), timeout=30)
    params = {
        "index": "equities",
        "from_date": start_date.strftime("%d-%m-%Y"),
        "to_date": end_date.strftime("%d-%m-%Y"),
        "csv": "true",
    }
    response = session.get(
        NSE_ACTIONS_URL,
        params=params,
        headers=_headers("text/csv,*/*"),
        timeout=90,
    )
    response.raise_for_status()
    if not response.content:
        raise RuntimeError("NSE corporate actions response was empty")
    return response.content


def _parse_bonus(purpose: str) -> tuple[float, float] | None:
    match = re.search(r"\bbonus\s+(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\b", purpose, re.I)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def _parse_split_or_consolidation(purpose: str) -> tuple[float, float] | None:
    match = re.search(
        r"from\s+rs\.?\s*(\d+(?:\.\d+)?)\s*/?\-?\s*per\s+share\s+to\s+"
        r"(?:rs\.?|re)\s*(\d+(?:\.\d+)?)",
        purpose,
        re.I,
    )
    if not match:
        match = re.search(
            r"from\s+rs\.?\s*(\d+(?:\.\d+)?)\s+per\s+share\s+to\s+"
            r"(?:rs\.?|re)\s*(\d+(?:\.\d+)?)",
            purpose,
            re.I,
        )
    if not match:
        return None
    old_face = float(match.group(1))
    new_face = float(match.group(2))
    if old_face <= 0 or new_face <= 0 or old_face == new_face:
        return None
    return old_face, new_face


def _normalize_actions(
    raw: pd.DataFrame,
    *,
    bundle_symbols: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = raw.copy()
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    required = {"SYMBOL", "SERIES", "PURPOSE", "EX-DATE"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"NSE corporate actions CSV missing columns: {missing}")

    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.upper().str.strip()
    frame["SERIES"] = frame["SERIES"].astype(str).str.upper().str.strip()
    frame["PURPOSE"] = frame["PURPOSE"].astype(str).str.strip()
    frame["EX-DATE"] = pd.to_datetime(frame["EX-DATE"], errors="coerce", dayfirst=True).dt.date
    frame = frame.dropna(subset=["SYMBOL", "EX-DATE"])
    frame = frame[(frame["SERIES"] == "EQ") & frame["SYMBOL"].isin(bundle_symbols)]

    rows: list[dict[str, Any]] = []
    skipped_supported_like = 0
    for row in frame.to_dict(orient="records"):
        symbol = str(row["SYMBOL"])
        purpose = str(row["PURPOSE"])
        ex_date = row["EX-DATE"]
        bonus = _parse_bonus(purpose)
        if bonus is not None:
            rows.append(
                {
                    "symbol": symbol,
                    "ex_date": ex_date,
                    "action_type": "BONUS",
                    "ratio_num": bonus[0],
                    "ratio_den": bonus[1],
                    "source": "NSE_CORPORATE_ACTIONS",
                }
            )
            continue

        split = _parse_split_or_consolidation(purpose)
        if split is not None:
            rows.append(
                {
                    "symbol": symbol,
                    "ex_date": ex_date,
                    "action_type": "SPLIT",
                    "ratio_num": split[0],
                    "ratio_den": split[1],
                    "source": "NSE_CORPORATE_ACTIONS",
                }
            )
            continue

        if re.search(r"\bdemerger\b", purpose, re.I):
            rows.append(
                {
                    "symbol": symbol,
                    "ex_date": ex_date,
                    "action_type": "DEMERGER",
                    "ratio_num": 1.0,
                    "ratio_den": 1.0,
                    "source": "NSE_CORPORATE_ACTIONS",
                }
            )
            continue

        if re.search(r"\b(bonus|split|sub-division|consolidation)\b", purpose, re.I):
            skipped_supported_like += 1

    normalized = pd.DataFrame(rows)
    if not normalized.empty:
        normalized = normalized.sort_values(["symbol", "ex_date", "action_type"]).drop_duplicates(
            subset=["symbol", "ex_date", "action_type"],
            keep="last",
        )
    summary = {
        "raw_rows": int(len(raw)),
        "scoped_eq_rows": int(len(frame)),
        "normalized_rows": int(len(normalized)),
        "skipped_supported_like": int(skipped_supported_like),
        "symbols_with_actions": int(normalized["symbol"].nunique()) if not normalized.empty else 0,
    }
    return normalized, summary


def import_free_nse_corporate_actions(
    *,
    bundle_id: int,
    start_date: date,
    end_date: date,
    mode: str,
    output_path: Path,
) -> dict[str, Any]:
    init_db()
    store = _store()
    with Session(engine) as session:
        bundle_symbols = set(store.get_bundle_symbols(session, bundle_id, timeframe="1d"))
    raw_bytes = _fetch_nse_actions_csv(start_date=start_date, end_date=end_date)
    raw_frame = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8-sig")
    normalized, normalization_summary = _normalize_actions(
        raw_frame,
        bundle_symbols=bundle_symbols,
    )
    if normalized.empty:
        import_summary = {
            "status": "SKIPPED",
            "imported_count": 0,
            "inserted_count": 0,
            "updated_count": 0,
            "deleted_count": 0,
            "warnings": [{"code": "no_supported_actions", "message": "No split/bonus actions found."}],
        }
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        normalized.to_csv(output_path, index=False)
        with Session(engine) as session:
            import_summary = import_corporate_actions(
                session,
                path=str(output_path),
                mode=mode,
            )

    return {
        "bundle_id": int(bundle_id),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "output_path": str(output_path),
        "normalization": normalization_summary,
        "import": import_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch free official NSE corporate actions and import split/bonus adjustments."
    )
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--start-date", type=_parse_day, default=date(2020, 1, 1))
    parser.add_argument("--end-date", type=_parse_day, default=date.today())
    parser.add_argument("--mode", choices=["UPSERT", "REPLACE"], default="UPSERT")
    parser.add_argument(
        "--output-path",
        default="data/inbox/_metadata/corporate_actions_nse.csv",
    )
    args = parser.parse_args()

    result = import_free_nse_corporate_actions(
        bundle_id=int(args.bundle_id),
        start_date=args.start_date,
        end_date=args.end_date,
        mode=str(args.mode),
        output_path=Path(str(args.output_path)),
    )
    print(result)


if __name__ == "__main__":
    main()
