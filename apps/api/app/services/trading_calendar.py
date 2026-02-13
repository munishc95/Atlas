from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.core.config import Settings, get_settings


IST_ZONE = ZoneInfo("Asia/Kolkata")
DEFAULT_SEGMENT = "EQUITIES"


@dataclass(frozen=True)
class SessionWindow:
    open_time: str | None
    close_time: str | None
    is_special: bool
    label: str | None = None


def _segment_token(segment: str | None) -> str:
    token = str(segment or DEFAULT_SEGMENT).strip().upper()
    if token in {"EQUITIES", "NSE_EQUITIES", "EQ"}:
        return "EQUITIES"
    return token


def _safe_time(value: str | None, default: str) -> str:
    raw = value if isinstance(value, str) and value.strip() else default
    parts = raw.split(":", maxsplit=1)
    if len(parts) != 2:
        return default
    try:
        hour = max(0, min(23, int(parts[0])))
        minute = max(0, min(59, int(parts[1])))
    except ValueError:
        return default
    return f"{hour:02d}:{minute:02d}"


def parse_time_hhmm(value: str | None, *, default: str = "15:35") -> time:
    text = _safe_time(value, default)
    hour, minute = text.split(":", maxsplit=1)
    return time(hour=int(hour), minute=int(minute))


def _calendar_root(settings: Settings | None = None) -> Path:
    cfg = settings or get_settings()
    return Path(cfg.calendar_data_root)


def _calendar_path(*, year: int, segment: str, settings: Settings | None = None) -> Path:
    seg = _segment_token(segment).lower()
    return _calendar_root(settings) / f"nse_{seg}_holidays_{year}.json"


def calendar_file_path(
    *,
    year: int,
    segment: str = DEFAULT_SEGMENT,
    settings: Settings | None = None,
) -> Path:
    return _calendar_path(year=year, segment=segment, settings=settings)


def _read_calendar_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def _parse_iso_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _extract_holidays(payload: dict[str, Any]) -> dict[date, dict[str, Any]]:
    result: dict[date, dict[str, Any]] = {}
    rows = payload.get("holidays", [])
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            continue
        day = _parse_iso_date(row.get("date"))
        if day is None:
            continue
        result[day] = {
            "name": str(row.get("name", "")).strip(),
            "type": str(row.get("type", "TRADING_HOLIDAY")).strip().upper(),
        }
    return result


def _extract_special_sessions(payload: dict[str, Any]) -> dict[date, dict[str, Any]]:
    result: dict[date, dict[str, Any]] = {}
    rows = payload.get("special_sessions", [])
    if not isinstance(rows, list):
        return result
    for row in rows:
        if not isinstance(row, dict):
            continue
        day = _parse_iso_date(row.get("date"))
        if day is None:
            continue
        result[day] = {
            "open": _safe_time(row.get("open"), "09:15"),
            "close": _safe_time(row.get("close"), "15:30"),
            "label": str(row.get("label", "")).strip() or None,
        }
    return result


def _load_year(
    *,
    year: int,
    segment: str = DEFAULT_SEGMENT,
    settings: Settings | None = None,
) -> tuple[dict[date, dict[str, Any]], dict[date, dict[str, Any]]]:
    path = _calendar_path(year=year, segment=segment, settings=settings)
    payload = _read_calendar_file(path)
    return _extract_holidays(payload), _extract_special_sessions(payload)


def _day_map_for_years(
    *,
    years: set[int],
    segment: str = DEFAULT_SEGMENT,
    settings: Settings | None = None,
) -> tuple[dict[date, dict[str, Any]], dict[date, dict[str, Any]]]:
    holidays: dict[date, dict[str, Any]] = {}
    specials: dict[date, dict[str, Any]] = {}
    for year in sorted(years):
        yr_holidays, yr_specials = _load_year(year=year, segment=segment, settings=settings)
        holidays.update(yr_holidays)
        specials.update(yr_specials)
    return holidays, specials


def is_trading_day(
    day: date,
    segment: str = DEFAULT_SEGMENT,
    *,
    settings: Settings | None = None,
) -> bool:
    holidays, specials = _day_map_for_years(years={day.year}, segment=segment, settings=settings)
    if day in specials:
        return True
    if day in holidays:
        return False
    return day.weekday() < 5


def next_trading_day(
    day: date,
    segment: str = DEFAULT_SEGMENT,
    *,
    settings: Settings | None = None,
) -> date:
    cursor = day
    for _ in range(400):
        cursor += timedelta(days=1)
        if is_trading_day(cursor, segment=segment, settings=settings):
            return cursor
    raise RuntimeError("Could not determine next trading day within 400 days.")


def previous_trading_day(
    day: date,
    segment: str = DEFAULT_SEGMENT,
    *,
    settings: Settings | None = None,
) -> date:
    cursor = day
    for _ in range(400):
        cursor -= timedelta(days=1)
        if is_trading_day(cursor, segment=segment, settings=settings):
            return cursor
    raise RuntimeError("Could not determine previous trading day within 400 days.")


def get_session(
    day: date,
    segment: str = DEFAULT_SEGMENT,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    default_open = _safe_time(cfg.nse_equities_open_time_ist, "09:15")
    default_close = _safe_time(cfg.nse_equities_close_time_ist, "15:30")
    holidays, specials = _day_map_for_years(years={day.year}, segment=segment, settings=cfg)
    special = specials.get(day)
    trading = is_trading_day(day, segment=segment, settings=cfg)
    holiday_name = holidays.get(day, {}).get("name")
    if special is not None:
        return {
            "open_time": str(special.get("open", default_open)),
            "close_time": str(special.get("close", default_close)),
            "is_special": True,
            "label": special.get("label"),
            "is_trading_day": True,
            "holiday_name": holiday_name,
        }
    if not trading:
        return {
            "open_time": None,
            "close_time": None,
            "is_special": False,
            "label": None,
            "is_trading_day": False,
            "holiday_name": holiday_name,
        }
    return {
        "open_time": default_open,
        "close_time": default_close,
        "is_special": False,
        "label": None,
        "is_trading_day": True,
        "holiday_name": holiday_name,
    }


def list_trading_days(
    *,
    start_date: date,
    end_date: date,
    segment: str = DEFAULT_SEGMENT,
    settings: Settings | None = None,
) -> list[date]:
    if end_date < start_date:
        return []
    days: list[date] = []
    cursor = start_date
    while cursor <= end_date:
        if is_trading_day(cursor, segment=segment, settings=settings):
            days.append(cursor)
        cursor += timedelta(days=1)
    return days


def compute_next_scheduled_run_ist(
    *,
    auto_run_enabled: bool,
    auto_run_time_ist: str,
    last_run_date: str | None,
    segment: str = DEFAULT_SEGMENT,
    now_ist: datetime | None = None,
    settings: Settings | None = None,
) -> str | None:
    if not auto_run_enabled:
        return None
    now = now_ist or datetime.now(IST_ZONE)
    cfg = settings or get_settings()
    run_time = parse_time_hhmm(auto_run_time_ist, default=cfg.operate_auto_run_time_ist)
    candidate = now.date()

    if not is_trading_day(candidate, segment=segment, settings=cfg):
        candidate = next_trading_day(candidate, segment=segment, settings=cfg)
    else:
        if now.time() >= run_time:
            candidate = next_trading_day(candidate, segment=segment, settings=cfg)
        if isinstance(last_run_date, str) and last_run_date == candidate.isoformat():
            candidate = next_trading_day(candidate, segment=segment, settings=cfg)

    scheduled = datetime.combine(candidate, run_time, tzinfo=IST_ZONE)
    return scheduled.isoformat()
