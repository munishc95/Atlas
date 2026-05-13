from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import io
import json
from pathlib import Path
import re
from typing import Any, Callable

import pandas as pd
import requests
from sqlmodel import Session

from app.core.config import Settings
from app.services.data_store import DataStore

NSE_BASE_URL = "https://www.nseindia.com"
NSE_ACTIONS_PAGE_URL = f"{NSE_BASE_URL}/companies-listing/corporate-filings-actions"
NSE_BOARD_MEETINGS_PAGE_URL = (
    f"{NSE_BASE_URL}/companies-listing/corporate-filings-board-meetings"
)
NSE_ANNOUNCEMENTS_PAGE_URL = (
    f"{NSE_BASE_URL}/companies-listing/corporate-filings-announcements"
)
NSE_ACTIONS_URL = f"{NSE_BASE_URL}/api/corporates-corporateActions"
NSE_BOARD_MEETINGS_URL = f"{NSE_BASE_URL}/api/corporate-board-meetings"
NSE_ANNOUNCEMENTS_URL = f"{NSE_BASE_URL}/api/corporate-announcements"

EVENT_COLUMNS = [
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

FetchCsv = Callable[[str, str, date, date, Settings], pd.DataFrame]


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


def fetch_nse_csv(
    page_url: str,
    api_url: str,
    start_date: date,
    end_date: date,
    settings: Settings,
) -> pd.DataFrame:
    timeout = float(getattr(settings, "nse_announcements_timeout_seconds", 18.0))
    with requests.Session() as http:
        http.get(
            page_url,
            headers=_headers(accept="text/html,*/*", referer=page_url),
            timeout=timeout,
        )
        response = http.get(
            api_url,
            params={
                "index": "equities",
                "from_date": start_date.strftime("%d-%m-%Y"),
                "to_date": end_date.strftime("%d-%m-%Y"),
                "csv": "true",
            },
            headers=_headers(accept="text/csv,*/*", referer=page_url),
            timeout=max(timeout, 30.0),
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


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _parse_day_series(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce", dayfirst=True).dt.date


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


def _severity_for_announcement(subject: str, details: str) -> tuple[str, str, int, int]:
    text = f"{subject} {details}".lower()
    if re.search(r"\b(financial results?|audited results?|unaudited results?|quarterly results?|earnings)\b", text):
        return "BLOCK", "RESULTS", 2, 1
    if re.search(
        r"\b(demerger|amalgamation|merger|scheme of arrangement|buyback|rights issue|"
        r"open offer|delisting|insolvency|bankruptcy|liquidation|winding up|default|"
        r"fraud|forensic|sebi order|nclt|search and seizure|rating downgrade)\b",
        text,
    ):
        return "BLOCK", "PRICE_SENSITIVE_ANNOUNCEMENT", 3, 1
    if re.search(
        r"\b(acquisition|disposal|slump sale|joint venture|fund raising|qip|preferential issue|"
        r"pledge|litigation|penalty|resignation|credit rating|order win|contract|"
        r"capacity expansion|plant shutdown|approval)\b",
        text,
    ):
        return "WARN", "ANNOUNCEMENT", 1, 0
    return "INFO", "ANNOUNCEMENT", 0, 0


def action_events(raw: pd.DataFrame, *, bundle_symbols: set[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = _clean_columns(raw)
    required = {"SYMBOL", "SERIES", "PURPOSE", "EX-DATE"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.upper().str.strip()
    frame["SERIES"] = frame["SERIES"].astype(str).str.upper().str.strip()
    frame["PURPOSE"] = frame["PURPOSE"].map(_clean_text)
    frame["EX-DATE"] = _parse_day_series(frame["EX-DATE"])
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
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def board_meeting_events(raw: pd.DataFrame, *, bundle_symbols: set[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = _clean_columns(raw)
    required = {"SYMBOL", "PURPOSE", "MEETING DATE"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame["SYMBOL"] = frame["SYMBOL"].astype(str).str.upper().str.strip()
    frame["PURPOSE"] = frame["PURPOSE"].map(_clean_text)
    details_col = "DETAILS" if "DETAILS" in frame.columns else "PURPOSE"
    frame["DETAILS"] = frame[details_col].map(_clean_text)
    frame["MEETING DATE"] = _parse_day_series(frame["MEETING DATE"])
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
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def announcement_events(raw: pd.DataFrame, *, bundle_symbols: set[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    frame = _clean_columns(raw)
    symbol_col = _first_column(frame, ("SYMBOL", "SYMB", "SCRIP CODE"))
    date_col = _first_column(
        frame,
        (
            "BROADCAST DATE/TIME",
            "DISSEMINATION DATE/TIME",
            "ANNOUNCEMENT DATE",
            "DATE",
        ),
    )
    if symbol_col is None or date_col is None:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    subject_col = _first_column(frame, ("SUBJECT", "DESC", "DESCRIPTION", "CATEGORY", "PURPOSE"))
    details_col = _first_column(frame, ("DETAILS", "MORE", "SUBCATEGORY", "ATTACHMENT", "SUBJECT"))

    frame["SYMBOL"] = frame[symbol_col].astype(str).str.upper().str.strip()
    frame["EVENT_DATE"] = _parse_day_series(frame[date_col])
    frame["SUBJECT_TEXT"] = frame[subject_col].map(_clean_text) if subject_col else ""
    frame["DETAILS_TEXT"] = frame[details_col].map(_clean_text) if details_col else ""
    frame = frame.dropna(subset=["SYMBOL", "EVENT_DATE"])
    frame = frame[frame["SYMBOL"].isin(bundle_symbols)]

    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        subject = str(row.get("SUBJECT_TEXT", ""))
        details = str(row.get("DETAILS_TEXT", ""))
        severity, event_type, before_days, after_days = _severity_for_announcement(subject, details)
        title = subject or details or event_type
        rows.append(
            {
                "event_date": row["EVENT_DATE"],
                "scope": "SYMBOL",
                "symbol": row["SYMBOL"],
                "event_type": event_type,
                "severity": severity,
                "title": f"{row['SYMBOL']} announcement: {title}",
                "source": "NSE_ANNOUNCEMENTS_SYNC",
                "blackout_before_days": before_days,
                "blackout_after_days": after_days,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def _normalize_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    clean = frame.copy()
    for column in EVENT_COLUMNS:
        if column not in clean.columns:
            clean[column] = ""
    clean = clean[EVENT_COLUMNS]
    clean["event_date"] = pd.to_datetime(clean["event_date"], errors="coerce").dt.date
    clean = clean.dropna(subset=["event_date"])
    clean["scope"] = clean["scope"].astype(str).str.upper().str.strip()
    clean["symbol"] = clean["symbol"].astype(str).str.upper().str.strip()
    clean["event_type"] = clean["event_type"].astype(str).str.upper().str.strip()
    clean["severity"] = clean["severity"].astype(str).str.upper().str.strip()
    clean["title"] = clean["title"].map(_clean_text)
    clean["source"] = clean["source"].astype(str).str.upper().str.strip()
    for column in ("blackout_before_days", "blackout_after_days"):
        clean[column] = pd.to_numeric(clean[column], errors="coerce").fillna(0).astype(int)
    return clean


def _read_existing_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=EVENT_COLUMNS)
    try:
        return _normalize_events(pd.read_csv(path))
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=EVENT_COLUMNS)


def _write_events(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _normalize_events(frame)
    if not clean.empty:
        clean = (
            clean.sort_values(["event_date", "scope", "symbol", "event_type", "source", "title"])
            .drop_duplicates(
                subset=["event_date", "scope", "symbol", "event_type", "source", "title"],
                keep="last",
            )
            .reset_index(drop=True)
        )
    clean.to_csv(path, index=False)


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def sync_event_risk_calendar(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    start_date: date,
    end_date: date,
    output_path: Path | None = None,
    include_actions: bool = True,
    include_board_meetings: bool = True,
    include_announcements: bool = True,
    fetch_csv: FetchCsv = fetch_nse_csv,
) -> dict[str, Any]:
    symbols = set(store.get_bundle_symbols(session, bundle_id, timeframe="1d"))
    output = output_path or Path(settings.event_risk_generated_calendar_path)

    jobs: list[tuple[str, str, str, Callable[[pd.DataFrame], pd.DataFrame]]] = []
    if include_actions:
        jobs.append(
            (
                "NSE_CORPORATE_ACTIONS_SYNC",
                NSE_ACTIONS_PAGE_URL,
                NSE_ACTIONS_URL,
                lambda raw: action_events(raw, bundle_symbols=symbols),
            )
        )
    if include_board_meetings:
        jobs.append(
            (
                "NSE_BOARD_MEETINGS_SYNC",
                NSE_BOARD_MEETINGS_PAGE_URL,
                NSE_BOARD_MEETINGS_URL,
                lambda raw: board_meeting_events(raw, bundle_symbols=symbols),
            )
        )
    if include_announcements:
        jobs.append(
            (
                "NSE_ANNOUNCEMENTS_SYNC",
                NSE_ANNOUNCEMENTS_PAGE_URL,
                NSE_ANNOUNCEMENTS_URL,
                lambda raw: announcement_events(raw, bundle_symbols=symbols),
            )
        )

    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    source_errors: dict[str, str] = {}
    successful_sources: set[str] = set()
    for source, page_url, api_url, converter in jobs:
        try:
            raw = fetch_csv(page_url, api_url, start_date, end_date, settings)
            events = converter(raw)
            frames.append(events)
            source_counts[source] = int(len(events))
            successful_sources.add(source)
        except Exception as exc:  # noqa: BLE001
            source_counts[source] = 0
            source_errors[source] = str(exc)

    if not successful_sources and source_errors:
        return {
            "status": "FAILED",
            "bundle_id": int(bundle_id),
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "output_path": str(output),
            "event_count": int(len(_read_existing_events(output))),
            "source_counts": source_counts,
            "source_errors": source_errors,
            "symbols_with_events": 0,
            "refreshed": False,
        }

    existing = _read_existing_events(output)
    if successful_sources and not existing.empty:
        existing = existing[~existing["source"].isin(successful_sources)]
    merged_parts = [existing] + [frame for frame in frames if not frame.empty]
    merged = pd.concat(merged_parts, ignore_index=True) if merged_parts else pd.DataFrame(columns=EVENT_COLUMNS)
    _write_events(output, merged)
    written = _read_existing_events(output)
    status = "PARTIAL" if source_errors else "SUCCEEDED"
    return {
        "status": status,
        "bundle_id": int(bundle_id),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "output_path": str(output),
        "event_count": int(len(written)),
        "source_counts": source_counts,
        "source_errors": source_errors,
        "symbols_with_events": int(written["symbol"].nunique()) if not written.empty else 0,
        "refreshed": True,
    }


def _bool_setting(overrides: dict[str, Any], settings: Settings, key: str, default: bool) -> bool:
    value = overrides.get(key, getattr(settings, key, default))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _int_setting(overrides: dict[str, Any], settings: Settings, key: str, default: int) -> int:
    try:
        return int(overrides.get(key, getattr(settings, key, default)))
    except (TypeError, ValueError):
        return int(default)


def refresh_event_risk_calendar_for_signals(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int | None,
    asof_dt: datetime,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
    fetch_csv: FetchCsv = fetch_nse_csv,
) -> dict[str, Any]:
    scope = dict(overrides or {})
    if bundle_id is None or int(bundle_id) <= 0:
        return {"status": "SKIPPED", "reason": "no_bundle", "refreshed": False}
    if not _bool_setting(scope, settings, "event_risk_enabled", True):
        return {"status": "SKIPPED", "reason": "event_risk_disabled", "refreshed": False}
    if not _bool_setting(scope, settings, "event_risk_sync_before_signals", False):
        return {"status": "SKIPPED", "reason": "sync_disabled", "refreshed": False}

    output_path = Path(str(scope.get("event_risk_generated_calendar_path") or settings.event_risk_generated_calendar_path))
    meta_path = Path(str(scope.get("event_risk_sync_meta_path") or settings.event_risk_sync_meta_path))
    asof_day = asof_dt.astimezone(timezone.utc).date() if asof_dt.tzinfo else asof_dt.date()
    start_date = asof_day - timedelta(
        days=max(0, _int_setting(scope, settings, "event_risk_sync_lookback_days", 3))
    )
    end_date = asof_day + timedelta(
        days=max(0, _int_setting(scope, settings, "event_risk_sync_lookahead_days", 14))
    )
    include_actions = _bool_setting(scope, settings, "event_risk_sync_include_actions", True)
    include_board = _bool_setting(scope, settings, "event_risk_sync_include_board_meetings", True)
    include_announcements = _bool_setting(
        scope,
        settings,
        "event_risk_sync_include_announcements",
        True,
    )
    params = {
        "bundle_id": int(bundle_id),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "include_actions": include_actions,
        "include_board_meetings": include_board,
        "include_announcements": include_announcements,
        "output_path": str(output_path),
    }
    min_interval = max(
        0,
        _int_setting(scope, settings, "event_risk_sync_min_interval_minutes", 60),
    )
    meta = _read_meta(meta_path)
    if not force and output_path.exists() and min_interval > 0 and meta.get("params") == params:
        updated_at = pd.to_datetime(meta.get("updated_at"), utc=True, errors="coerce")
        if not pd.isna(updated_at):
            age_minutes = (
                pd.Timestamp.now(tz="UTC") - updated_at
            ).total_seconds() / 60.0
            if age_minutes < min_interval:
                return {
                    "status": "SKIPPED",
                    "reason": "recent_sync",
                    "age_minutes": float(age_minutes),
                    "refreshed": False,
                    **params,
                }

    result = sync_event_risk_calendar(
        session=session,
        settings=settings,
        store=store,
        bundle_id=int(bundle_id),
        start_date=start_date,
        end_date=end_date,
        output_path=output_path,
        include_actions=include_actions,
        include_board_meetings=include_board,
        include_announcements=include_announcements,
        fetch_csv=fetch_csv,
    )
    if str(result.get("status")) in {"SUCCEEDED", "PARTIAL"}:
        _write_meta(
            meta_path,
            {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "params": params,
                "result": result,
            },
        )
    return result
