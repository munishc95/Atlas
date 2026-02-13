from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

from app.core.config import get_settings
from app.services.trading_calendar import calendar_file_path


NSE_TRADING_HOLIDAYS_API = "https://www.nseindia.com/api/holiday-master?type=trading"


def _utc_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _normalize_entry_date(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normalize_payload(
    *,
    year: int,
    segment: str,
    holidays: list[dict[str, Any]],
    special_sessions: list[dict[str, Any]] | None = None,
    source: str,
) -> dict[str, Any]:
    return {
        "year": int(year),
        "segment": str(segment).upper(),
        "holidays": holidays,
        "special_sessions": special_sessions or [],
        "source": source,
        "last_refreshed": _utc_today(),
    }


def _load_from_json(path: Path, *, year: int, segment: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        holidays = payload.get("holidays", [])
        specials = payload.get("special_sessions", [])
    elif isinstance(payload, list):
        holidays = payload
        specials = []
    else:
        raise ValueError("Unsupported JSON shape for calendar import.")
    if not isinstance(holidays, list):
        raise ValueError("JSON holidays must be a list.")
    normalized: list[dict[str, Any]] = []
    for row in holidays:
        if not isinstance(row, dict):
            continue
        day = _normalize_entry_date(row.get("date"))
        if day is None:
            continue
        normalized.append(
            {
                "date": day,
                "name": str(row.get("name", "Exchange Holiday")).strip(),
                "type": str(row.get("type", "TRADING_HOLIDAY")).strip().upper(),
            }
        )
    normalized_specials: list[dict[str, Any]] = []
    if isinstance(specials, list):
        for row in specials:
            if not isinstance(row, dict):
                continue
            day = _normalize_entry_date(row.get("date"))
            if day is None:
                continue
            normalized_specials.append(
                {
                    "date": day,
                    "label": str(row.get("label", "Special Session")).strip(),
                    "open": str(row.get("open", "09:15")).strip(),
                    "close": str(row.get("close", "15:30")).strip(),
                }
            )
    return _normalize_payload(
        year=year,
        segment=segment,
        holidays=normalized,
        special_sessions=normalized_specials,
        source=f"Manual import ({path.name})",
    )


def _load_from_csv(path: Path, *, year: int, segment: str) -> dict[str, Any]:
    holidays: list[dict[str, Any]] = []
    specials: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            day = _normalize_entry_date(row.get("date"))
            if day is None:
                continue
            row_type = str(row.get("type", "TRADING_HOLIDAY")).strip().upper()
            if row_type in {"SPECIAL_SESSION", "SPECIAL"}:
                specials.append(
                    {
                        "date": day,
                        "label": str(row.get("label") or row.get("name") or "Special Session").strip(),
                        "open": str(row.get("open", "09:15")).strip(),
                        "close": str(row.get("close", "15:30")).strip(),
                    }
                )
            else:
                holidays.append(
                    {
                        "date": day,
                        "name": str(row.get("name", "Exchange Holiday")).strip(),
                        "type": row_type or "TRADING_HOLIDAY",
                    }
                )
    return _normalize_payload(
        year=year,
        segment=segment,
        holidays=holidays,
        special_sessions=specials,
        source=f"Manual import ({path.name})",
    )


def _extract_candidates(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for item in value.values():
            rows.extend(_extract_candidates(item))
        return rows
    return []


def _fetch_nse_holidays(*, year: int, segment: str) -> dict[str, Any] | None:
    request = urllib.request.Request(
        NSE_TRADING_HOLIDAYS_API,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json",
            "Referer": "https://www.nseindia.com/",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None

    candidates = _extract_candidates(payload)
    holidays: list[dict[str, Any]] = []
    for row in candidates:
        row_type = str(
            row.get("type")
            or row.get("tradingType")
            or row.get("holidayType")
            or "TRADING_HOLIDAY"
        ).strip().upper()
        if row_type not in {"TRADING_HOLIDAY", "HOLIDAY"}:
            continue
        day = _normalize_entry_date(
            row.get("date")
            or row.get("tradingDate")
            or row.get("trading_date")
            or row.get("holidayDate")
        )
        if day is None or not day.startswith(f"{year:04d}-"):
            continue
        holidays.append(
            {
                "date": day,
                "name": str(row.get("description") or row.get("name") or "Exchange Holiday").strip(),
                "type": "TRADING_HOLIDAY",
            }
        )
    if not holidays:
        return None
    holidays = sorted({row["date"]: row for row in holidays}.values(), key=lambda row: row["date"])
    return _normalize_payload(
        year=year,
        segment=segment,
        holidays=holidays,
        source="NSE API (best-effort)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh local NSE trading calendar files.")
    parser.add_argument("--year", type=int, required=True, help="Calendar year, e.g. 2026")
    parser.add_argument("--segment", default="EQUITIES", help="Trading segment, default EQUITIES")
    parser.add_argument(
        "--from-file",
        dest="from_file",
        default=None,
        help="Optional local CSV/JSON file path for manual import",
    )
    args = parser.parse_args()

    settings = get_settings()
    target = calendar_file_path(year=args.year, segment=args.segment, settings=settings)
    target.parent.mkdir(parents=True, exist_ok=True)

    if args.from_file:
        source_path = Path(args.from_file)
        if not source_path.exists():
            print(f"[calendar] File not found: {source_path}")
            return 1
        try:
            if source_path.suffix.lower() == ".csv":
                payload = _load_from_csv(source_path, year=args.year, segment=args.segment)
            else:
                payload = _load_from_json(source_path, year=args.year, segment=args.segment)
        except Exception as exc:  # noqa: BLE001
            print(f"[calendar] Could not import override file: {exc}")
            return 1
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[calendar] Updated {target}")
        return 0

    payload = _fetch_nse_holidays(year=args.year, segment=args.segment)
    if payload is None:
        if target.exists():
            print(
                "[calendar] NSE fetch failed/unavailable; existing calendar kept intact at "
                f"{target}"
            )
            return 0
        print(
            "[calendar] NSE fetch failed and no existing file is present. "
            "Provide --from-file to seed the calendar."
        )
        return 1

    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calendar] Refreshed {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
