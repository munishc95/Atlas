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
from app.services.data_store import DataStore  # noqa: E402

NSE_BASE_URL = "https://www.nseindia.com"
NSE_ACTIONS_PAGE_URL = f"{NSE_BASE_URL}/companies-listing/corporate-filings-actions"
NSE_BOARD_MEETINGS_PAGE_URL = (
    f"{NSE_BASE_URL}/companies-listing/corporate-filings-board-meetings"
)
NSE_ACTIONS_URL = f"{NSE_BASE_URL}/api/corporates-corporateActions"
NSE_BOARD_MEETINGS_URL = f"{NSE_BASE_URL}/api/corporate-board-meetings"


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


def _headers(*, accept: str, referer: str) -> dict[str, str]:
    return {
        "Accept": accept,
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": referer,
        "Accept-Language": "en-US,en;q=0.9",
    }


def _fetch_nse_csv(
    *,
    page_url: str,
    api_url: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    session = requests.Session()
    session.get(page_url, headers=_headers(accept="text/html,*/*", referer=page_url), timeout=30)
    response = session.get(
        api_url,
        params={
            "index": "equities",
            "from_date": start_date.strftime("%d-%m-%Y"),
            "to_date": end_date.strftime("%d-%m-%Y"),
            "csv": "true",
        },
        headers=_headers(accept="text/csv,*/*", referer=page_url),
        timeout=90,
    )
    response.raise_for_status()
    if not response.content:
        return pd.DataFrame()
    return pd.read_csv(io.BytesIO(response.content), encoding="utf-8-sig")


def _clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    clean = frame.copy()
    clean.columns = [str(column).strip().upper() for column in clean.columns]
    return clean


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _severity_for_action(purpose: str) -> tuple[str, str, int, int]:
    text = purpose.lower()
    if re.search(r"\b(demerger|scheme of arrangement|rights|buyback)\b", text):
        return "BLOCK", "CORPORATE_ACTION", 7, 1
    if re.search(r"\b(split|sub-division|consolidation|bonus)\b", text):
        return "BLOCK", "CORPORATE_ACTION", 3, 1
    if "dividend" in text:
        return "WARN", "DIVIDEND", 1, 0
    return "WARN", "CORPORATE_ACTION", 1, 0


def _severity_for_board_meeting(purpose: str, details: str) -> tuple[str, str, int, int]:
    text = f"{purpose} {details}".lower()
    if re.search(r"\b(financial results?|audited results?|unaudited results?|quarterly results?|earnings)\b", text):
        return "BLOCK", "RESULTS", 2, 1
    if re.search(r"\b(demerger|amalgamation|merger|scheme of arrangement|rights|buyback)\b", text):
        return "BLOCK", "BOARD_EVENT", 3, 1
    if re.search(r"\b(fund raising|preferential issue|qip|bonus|split|sub-division)\b", text):
        return "WARN", "BOARD_EVENT", 2, 0
    return "WARN", "BOARD_MEETING", 1, 0


def _action_events(
    raw: pd.DataFrame,
    *,
    bundle_symbols: set[str],
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = _clean_columns(raw)
    required = {"SYMBOL", "SERIES", "PURPOSE", "EX-DATE"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.upper().str.strip()
    frame["SERIES"] = frame["SERIES"].astype(str).str.upper().str.strip()
    frame["PURPOSE"] = frame["PURPOSE"].map(_clean_text)
    frame["EX-DATE"] = pd.to_datetime(frame["EX-DATE"], errors="coerce", dayfirst=True).dt.date
    frame = frame.dropna(subset=["SYMBOL", "EX-DATE"])
    frame = frame[(frame["SERIES"] == "EQ") & frame["SYMBOL"].isin(bundle_symbols)]

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        severity, event_type, before_days, after_days = _severity_for_action(str(row["PURPOSE"]))
        rows.append(
            {
                "event_date": row["EX-DATE"],
                "scope": "SYMBOL",
                "symbol": row["SYMBOL"],
                "event_type": event_type,
                "severity": severity,
                "title": f"{row['SYMBOL']} corporate action: {row['PURPOSE']}",
                "source": "NSE_CORPORATE_ACTIONS_SYNC",
                "blackout_before_days": before_days,
                "blackout_after_days": after_days,
            }
        )
    return pd.DataFrame(rows)


def _board_meeting_events(
    raw: pd.DataFrame,
    *,
    bundle_symbols: set[str],
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = _clean_columns(raw)
    required = {"SYMBOL", "PURPOSE", "MEETING DATE"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.upper().str.strip()
    frame["PURPOSE"] = frame["PURPOSE"].map(_clean_text)
    details_col = "DETAILS" if "DETAILS" in frame.columns else "PURPOSE"
    frame["DETAILS"] = frame[details_col].map(_clean_text)
    frame["MEETING DATE"] = pd.to_datetime(
        frame["MEETING DATE"],
        errors="coerce",
        dayfirst=True,
    ).dt.date
    frame = frame.dropna(subset=["SYMBOL", "MEETING DATE"])
    frame = frame[frame["SYMBOL"].isin(bundle_symbols)]

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        purpose = str(row["PURPOSE"])
        details = str(row["DETAILS"])
        severity, event_type, before_days, after_days = _severity_for_board_meeting(purpose, details)
        rows.append(
            {
                "event_date": row["MEETING DATE"],
                "scope": "SYMBOL",
                "symbol": row["SYMBOL"],
                "event_type": event_type,
                "severity": severity,
                "title": f"{row['SYMBOL']} board meeting: {purpose}",
                "source": "NSE_BOARD_MEETINGS_SYNC",
                "blackout_before_days": before_days,
                "blackout_after_days": after_days,
            }
        )
    return pd.DataFrame(rows)


def sync_event_risk_calendar(
    *,
    bundle_id: int,
    start_date: date,
    end_date: date,
    output_path: Path,
    include_actions: bool,
    include_board_meetings: bool,
) -> dict[str, Any]:
    init_db()
    store = _store()
    with Session(engine) as session:
        bundle_symbols = set(store.get_bundle_symbols(session, bundle_id, timeframe="1d"))

    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    if include_actions:
        raw_actions = _fetch_nse_csv(
            page_url=NSE_ACTIONS_PAGE_URL,
            api_url=NSE_ACTIONS_URL,
            start_date=start_date,
            end_date=end_date,
        )
        action_events = _action_events(raw_actions, bundle_symbols=bundle_symbols)
        source_counts["NSE_CORPORATE_ACTIONS_SYNC"] = int(len(action_events))
        frames.append(action_events)

    if include_board_meetings:
        raw_meetings = _fetch_nse_csv(
            page_url=NSE_BOARD_MEETINGS_PAGE_URL,
            api_url=NSE_BOARD_MEETINGS_URL,
            start_date=start_date,
            end_date=end_date,
        )
        meeting_events = _board_meeting_events(raw_meetings, bundle_symbols=bundle_symbols)
        source_counts["NSE_BOARD_MEETINGS_SYNC"] = int(len(meeting_events))
        frames.append(meeting_events)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "event_date",
        "scope",
        "symbol",
        "event_type",
        "severity",
        "title",
        "source",
        "blackout_before_days",
        "blackout_after_days",
    ]
    non_empty_frames = [frame for frame in frames if not frame.empty]
    if non_empty_frames:
        merged = pd.concat(non_empty_frames, ignore_index=True)
    else:
        merged = pd.DataFrame(columns=columns)
    if merged.empty:
        merged = pd.DataFrame(columns=columns)
    else:
        merged = (
            merged[columns]
            .sort_values(["event_date", "scope", "symbol", "event_type", "source"])
            .drop_duplicates(subset=["event_date", "scope", "symbol", "event_type", "source"])
            .reset_index(drop=True)
        )
    merged.to_csv(output_path, index=False)
    return {
        "bundle_id": int(bundle_id),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "output_path": str(output_path),
        "event_count": int(len(merged)),
        "source_counts": source_counts,
        "symbols_with_events": int(merged["symbol"].nunique()) if not merged.empty else 0,
    }


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Sync free official event-risk calendar rows for Atlas."
    )
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--start-date", type=_parse_day, required=True)
    parser.add_argument("--end-date", type=_parse_day, required=True)
    parser.add_argument(
        "--output-path",
        default=settings.event_risk_generated_calendar_path,
    )
    parser.add_argument("--skip-actions", action="store_true")
    parser.add_argument("--skip-board-meetings", action="store_true")
    args = parser.parse_args()

    result = sync_event_risk_calendar(
        bundle_id=int(args.bundle_id),
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=Path(str(args.output_path)),
        include_actions=not bool(args.skip_actions),
        include_board_meetings=not bool(args.skip_board_meetings),
    )
    print(result)


if __name__ == "__main__":
    main()
