from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date as dt_date
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import pandas as pd
from sqlmodel import Session

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DataQualityReport, DatasetBundle
from app.providers.nse_bhavcopy_provider import NseBhavcopyProvider
from app.services.data_quality import get_latest_data_quality_report
from app.services.data_store import DataStore
from app.services.trading_calendar import list_trading_days
from app.services.universe_history import bundle_has_membership_history, list_membership_history


LOCAL_MISSING_EXCHANGE_PRESENT = "local_missing_exchange_present"
EXCHANGE_NO_ROW_ACTIVE_MEMBER = "exchange_no_row_active_member"
EXCHANGE_NO_ROW_MEMBERSHIP_UNKNOWN = "exchange_no_row_membership_unknown"
OUTSIDE_HISTORICAL_MEMBERSHIP = "outside_historical_membership"
PROVIDER_DAY_UNAVAILABLE = "provider_day_unavailable"
PROVIDER_DAY_MISMATCH = "provider_day_mismatch"

RECOMMENDATIONS = {
    LOCAL_MISSING_EXCHANGE_PRESENT: "backfill_local_rows",
    EXCHANGE_NO_ROW_ACTIVE_MEMBER: "add_no_trade_or_suspension_exception",
    EXCHANGE_NO_ROW_MEMBERSHIP_UNKNOWN: "import_membership_history_or_add_no_trade_exception",
    OUTSIDE_HISTORICAL_MEMBERSHIP: "fix_historical_membership",
    PROVIDER_DAY_UNAVAILABLE: "refresh_provider_archive_or_calendar",
    PROVIDER_DAY_MISMATCH: "fix_trading_calendar_or_provider_cache",
}

FrameLookup = Callable[[str], pd.DataFrame]
DayFrameLookup = Callable[[dt_date], pd.DataFrame]
TradingDaysLookup = Callable[[dt_date, dt_date], list[dt_date]]
MembershipLookup = Callable[[str, dt_date], bool | None]


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _symbol_tokens_from_gap_issues(report: DataQualityReport) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for issue in report.issues_json or []:
        if not isinstance(issue, dict):
            continue
        if str(issue.get("code", "")) != "gap_exceeds_threshold":
            continue
        symbol = str(issue.get("symbol", "")).strip().upper()
        if symbol and symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def _local_bar_dates(frame: pd.DataFrame) -> list[dt_date]:
    if frame.empty or "datetime" not in frame.columns:
        return []
    dt = pd.to_datetime(frame["datetime"], utc=True, errors="coerce").dropna()
    return sorted(set(dt.dt.date.tolist()))


def _discover_gap_events(
    symbol: str,
    local_dates: list[dt_date],
    *,
    max_gap_bars: int,
    trading_days_lookup: TradingDaysLookup,
) -> list[dict[str, Any]]:
    if len(local_dates) < 2:
        return []

    events: list[dict[str, Any]] = []
    for index in range(1, len(local_dates)):
        previous_day = local_dates[index - 1]
        next_day = local_dates[index]
        expected = trading_days_lookup(previous_day, next_day)
        missing_days = [day for day in expected if previous_day < day < next_day]
        if len(missing_days) <= max_gap_bars:
            continue
        events.append(
            {
                "symbol": symbol,
                "previous_bar_date": previous_day,
                "next_bar_date": next_day,
                "missing_days": missing_days,
            }
        )
    return events


IST_ZONE = ZoneInfo("Asia/Kolkata")


def _day_symbols(day_frame: pd.DataFrame, *, expected_day: dt_date) -> dict[str, Any]:
    if day_frame.empty:
        return {"status": PROVIDER_DAY_UNAVAILABLE, "symbols": None, "observed_dates": []}
    if "symbol" not in day_frame.columns:
        return {"status": PROVIDER_DAY_UNAVAILABLE, "symbols": None, "observed_dates": []}
    scoped = day_frame
    observed_dates: list[str] = []
    if "datetime" in day_frame.columns:
        dt = pd.to_datetime(day_frame["datetime"], utc=True, errors="coerce")
        local_dates = dt.dt.tz_convert(IST_ZONE).dt.date
        observed_dates = sorted({item.isoformat() for item in local_dates.dropna().tolist()})
        scoped = day_frame[local_dates == expected_day]
        if scoped.empty:
            return {
                "status": PROVIDER_DAY_MISMATCH,
                "symbols": None,
                "observed_dates": observed_dates[:10],
            }
    return {
        "status": "ok",
        "observed_dates": observed_dates[:10],
        "symbols": {
            str(item).strip().upper()
            for item in scoped["symbol"].tolist()
            if str(item).strip()
        },
    }


def _classify_missing_day(
    *,
    symbol: str,
    day: dt_date,
    day_symbols_lookup: Callable[[dt_date], dict[str, Any]],
    membership_lookup: MembershipLookup | None,
) -> dict[str, Any]:
    active_member = membership_lookup(symbol, day) if membership_lookup is not None else None
    if active_member is False:
        return {
            "date": day.isoformat(),
            "classification": OUTSIDE_HISTORICAL_MEMBERSHIP,
            "membership_active": False,
            "exchange_symbol_present": None,
            "reason": "Symbol was not active in bundle membership history on this date.",
        }

    exchange_payload = day_symbols_lookup(day)
    exchange_symbols = exchange_payload.get("symbols")
    exchange_status = str(exchange_payload.get("status", "ok"))
    if exchange_status == PROVIDER_DAY_MISMATCH:
        return {
            "date": day.isoformat(),
            "classification": PROVIDER_DAY_MISMATCH,
            "membership_active": active_member,
            "exchange_symbol_present": None,
            "observed_dates": exchange_payload.get("observed_dates", []),
            "reason": "Provider returned rows whose bar date did not match the requested date.",
        }
    if exchange_symbols is None:
        return {
            "date": day.isoformat(),
            "classification": PROVIDER_DAY_UNAVAILABLE,
            "membership_active": active_member,
            "exchange_symbol_present": None,
            "reason": "Provider day frame was unavailable or empty.",
        }

    exchange_present = symbol.upper() in exchange_symbols
    if exchange_present:
        return {
            "date": day.isoformat(),
            "classification": LOCAL_MISSING_EXCHANGE_PRESENT,
            "membership_active": active_member,
            "exchange_symbol_present": True,
            "reason": "Exchange source has a row, but local OHLCV is missing it.",
        }

    if active_member is True:
        classification = EXCHANGE_NO_ROW_ACTIVE_MEMBER
        reason = "Symbol was an active member, but the exchange source had no row."
    else:
        classification = EXCHANGE_NO_ROW_MEMBERSHIP_UNKNOWN
        reason = "Exchange source had no row and historical membership was not available."
    return {
        "date": day.isoformat(),
        "classification": classification,
        "membership_active": active_member,
        "exchange_symbol_present": False,
        "reason": reason,
    }


def _primary_classification(counts: Counter[str]) -> str | None:
    if not counts:
        return None
    for token in (
        LOCAL_MISSING_EXCHANGE_PRESENT,
        OUTSIDE_HISTORICAL_MEMBERSHIP,
        EXCHANGE_NO_ROW_ACTIVE_MEMBER,
        EXCHANGE_NO_ROW_MEMBERSHIP_UNKNOWN,
        PROVIDER_DAY_UNAVAILABLE,
        PROVIDER_DAY_MISMATCH,
    ):
        if counts.get(token, 0) > 0:
            return token
    return counts.most_common(1)[0][0]


def _recommendation(counts: Counter[str]) -> str | None:
    primary = _primary_classification(counts)
    if primary is None:
        return None
    return RECOMMENDATIONS.get(primary)


def _merge_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        token = str(row.get("classification", "")).strip()
        if token:
            counts[token] += 1
    return counts


def build_data_quality_gap_audit_payload(
    report: DataQualityReport,
    *,
    bundle_name: str | None = None,
    max_gap_bars: int,
    frame_lookup: FrameLookup,
    day_frame_lookup: DayFrameLookup,
    trading_days_lookup: TradingDaysLookup,
    membership_lookup: MembershipLookup | None = None,
    membership_history_available: bool = False,
    generated_at: datetime | None = None,
    max_gap_events_per_symbol: int = 50,
    max_date_samples_per_gap: int = 12,
) -> dict[str, Any]:
    generated = generated_at or _utc_now()
    target_symbols = _symbol_tokens_from_gap_issues(report)
    day_symbol_cache: dict[dt_date, dict[str, Any]] = {}

    def day_symbols_lookup(day: dt_date) -> dict[str, Any]:
        if day not in day_symbol_cache:
            day_symbol_cache[day] = _day_symbols(day_frame_lookup(day), expected_day=day)
        return day_symbol_cache[day]

    symbol_rows: list[dict[str, Any]] = []
    global_counts: Counter[str] = Counter()
    gap_events_total = 0
    missing_dates_total = 0
    symbols_by_primary: Counter[str] = Counter()

    for symbol in target_symbols:
        frame = frame_lookup(symbol)
        local_dates = _local_bar_dates(frame)
        events = _discover_gap_events(
            symbol,
            local_dates,
            max_gap_bars=int(max_gap_bars),
            trading_days_lookup=trading_days_lookup,
        )
        symbol_counts: Counter[str] = Counter()
        gap_rows: list[dict[str, Any]] = []
        first_missing: str | None = None
        last_missing: str | None = None

        for event in events:
            classified = [
                _classify_missing_day(
                    symbol=symbol,
                    day=day,
                    day_symbols_lookup=day_symbols_lookup,
                    membership_lookup=membership_lookup,
                )
                for day in event["missing_days"]
            ]
            counts = _merge_counts(classified)
            symbol_counts.update(counts)
            global_counts.update(counts)
            gap_events_total += 1
            missing_dates_total += len(classified)
            event_first = event["missing_days"][0].isoformat()
            event_last = event["missing_days"][-1].isoformat()
            first_missing = event_first if first_missing is None else min(first_missing, event_first)
            last_missing = event_last if last_missing is None else max(last_missing, event_last)
            gap_rows.append(
                {
                    "previous_bar_date": event["previous_bar_date"].isoformat(),
                    "next_bar_date": event["next_bar_date"].isoformat(),
                    "missing_dates_count": len(classified),
                    "classification_counts": dict(sorted(counts.items())),
                    "classified_dates": classified,
                    "sample_dates": classified[:max(0, int(max_date_samples_per_gap))],
                }
            )

        primary = _primary_classification(symbol_counts)
        if primary is not None:
            symbols_by_primary[primary] += 1
        symbol_rows.append(
            {
                "symbol": symbol,
                "gap_events": len(events),
                "missing_dates_total": int(sum(symbol_counts.values())),
                "classification_counts": dict(sorted(symbol_counts.items())),
                "primary_classification": primary,
                "recommended_action": _recommendation(symbol_counts),
                "first_missing_date": first_missing,
                "last_missing_date": last_missing,
                "gaps": gap_rows[:max(0, int(max_gap_events_per_symbol))],
                "truncated_gap_events": max(0, len(gap_rows) - max(0, int(max_gap_events_per_symbol))),
            }
        )

    symbol_rows = sorted(
        symbol_rows,
        key=lambda row: (
            -int(row.get("missing_dates_total", 0)),
            str(row.get("symbol", "")),
        ),
    )
    return {
        "generated_at": _iso(generated),
        "report": {
            "id": int(report.id) if report.id is not None else None,
            "bundle_id": int(report.bundle_id),
            "bundle_name": bundle_name,
            "timeframe": str(report.timeframe),
            "status": str(report.status),
            "created_at": _iso(report.created_at),
        },
        "settings": {
            "max_gap_bars": int(max_gap_bars),
            "membership_history_available": bool(membership_history_available),
            "provider": "NSE_BHAVCOPY",
        },
        "summary": {
            "symbols_with_gap_issues": len(target_symbols),
            "symbols_audited": len(symbol_rows),
            "gap_events_total": int(gap_events_total),
            "missing_dates_total": int(missing_dates_total),
            "classification_counts": dict(sorted(global_counts.items())),
            "symbols_by_primary_classification": dict(sorted(symbols_by_primary.items())),
            "backfillable_missing_dates": int(global_counts.get(LOCAL_MISSING_EXCHANGE_PRESENT, 0)),
            "backfillable_symbols": int(
                sum(
                    1
                    for row in symbol_rows
                    if row.get("classification_counts", {}).get(LOCAL_MISSING_EXCHANGE_PRESENT, 0)
                )
            ),
            "provider_days_checked": len(day_symbol_cache),
        },
        "symbols": symbol_rows,
    }


def gap_audit_markdown(payload: dict[str, Any], *, max_symbols: int = 80) -> str:
    report = payload.get("report", {}) if isinstance(payload.get("report"), dict) else {}
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    settings = payload.get("settings", {}) if isinstance(payload.get("settings"), dict) else {}
    rows = payload.get("symbols", []) if isinstance(payload.get("symbols"), list) else []
    lines = [
        "# Data Quality Gap Audit",
        "",
        f"- Report id: {report.get('id')}",
        f"- Bundle: {report.get('bundle_name') or report.get('bundle_id')}",
        f"- Timeframe: {report.get('timeframe')}",
        f"- Status: {report.get('status')}",
        f"- Max gap bars: {settings.get('max_gap_bars')}",
        f"- Membership history available: {settings.get('membership_history_available')}",
        f"- Symbols audited: {summary.get('symbols_audited')}",
        f"- Gap events: {summary.get('gap_events_total')}",
        f"- Missing dates: {summary.get('missing_dates_total')}",
        f"- Backfillable missing dates: {summary.get('backfillable_missing_dates')}",
        "",
        "## Classification Counts",
        "",
    ]
    counts = summary.get("classification_counts", {})
    if isinstance(counts, dict) and counts:
        for key, value in sorted(counts.items()):
            lines.append(f"- {key}: {value}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Top Symbols",
            "",
            "| Symbol | Primary | Action | Missing Dates | Gap Events | First | Last | Counts |",
            "|---:|---|---|---:|---:|---|---|---|",
        ]
    )
    for row in rows[:max(0, int(max_symbols))]:
        counts_text = ", ".join(
            f"{key}={value}"
            for key, value in sorted((row.get("classification_counts") or {}).items())
        )
        lines.append(
            "| "
            f"{row.get('symbol')} | "
            f"{row.get('primary_classification')} | "
            f"{row.get('recommended_action')} | "
            f"{row.get('missing_dates_total')} | "
            f"{row.get('gap_events')} | "
            f"{row.get('first_missing_date') or ''} | "
            f"{row.get('last_missing_date') or ''} | "
            f"{counts_text} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_data_quality_gap_audit_files(
    payload: dict[str, Any],
    *,
    output_dir: str | Path = "data/reports/data-quality",
) -> dict[str, str]:
    report = payload.get("report", {}) if isinstance(payload.get("report"), dict) else {}
    bundle_id = report.get("bundle_id", "unknown")
    timeframe = report.get("timeframe", "unknown")
    report_id = report.get("id", "latest")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"data-quality-gap-audit-bundle-{bundle_id}-{timeframe}-report-{report_id}"
    json_path = root / f"{stem}.json"
    md_path = root / f"{stem}.md"
    json_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    md_path.write_text(gap_audit_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _membership_lookup_from_history(
    session: Session,
    *,
    bundle_id: int,
) -> tuple[bool, MembershipLookup | None]:
    if not bundle_has_membership_history(session, bundle_id=int(bundle_id)):
        return False, None
    rows_by_symbol: dict[str, list[tuple[dt_date, dt_date | None]]] = defaultdict(list)
    for row in list_membership_history(session, bundle_id=int(bundle_id)):
        rows_by_symbol[str(row.symbol).upper()].append((row.effective_from, row.effective_to))

    def lookup(symbol: str, day: dt_date) -> bool | None:
        intervals = rows_by_symbol.get(str(symbol).upper(), [])
        if not intervals:
            return False
        for start, end in intervals:
            if start <= day and (end is None or end >= day):
                return True
        return False

    return True, lookup


def generate_data_quality_gap_audit(
    *,
    session: Session,
    settings: Settings,
    bundle_id: int,
    timeframe: str = "1d",
    report_id: int | None = None,
    write_files: bool = False,
    output_dir: str | Path = "data/reports/data-quality",
    store: DataStore | None = None,
) -> dict[str, Any]:
    tf = str(timeframe).strip() or "1d"
    if tf.lower() != "1d":
        raise APIError(
            code="unsupported_timeframe",
            message="Gap audit currently supports daily OHLCV only.",
            status_code=400,
        )
    if report_id is not None:
        report = session.get(DataQualityReport, int(report_id))
        if report is None:
            raise APIError(code="not_found", message="Data quality report not found", status_code=404)
    else:
        report = get_latest_data_quality_report(
            session,
            bundle_id=int(bundle_id),
            timeframe=tf,
        )
        if report is None:
            raise APIError(code="not_found", message="No data quality report found", status_code=404)
    if int(report.bundle_id) != int(bundle_id) or str(report.timeframe).lower() != tf.lower():
        raise APIError(
            code="report_scope_mismatch",
            message="Data quality report does not match requested bundle/timeframe.",
            status_code=400,
        )

    bundle = session.get(DatasetBundle, int(bundle_id))
    resolved_store = store or DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )
    provider = NseBhavcopyProvider(session=session, settings=settings, store=resolved_store)
    segment = str(settings.trading_calendar_segment or "EQUITIES")
    max_gap_bars = max(0, _safe_int(settings.operate_max_gap_bars, 0))
    membership_available, membership_lookup = _membership_lookup_from_history(
        session,
        bundle_id=int(bundle_id),
    )

    def frame_lookup(symbol: str) -> pd.DataFrame:
        return resolved_store.load_ohlcv(
            symbol=symbol,
            timeframe=tf,
            session=session,
            adjustment_mode=settings.data_adjustment_mode,
        )

    def trading_days_lookup(start: dt_date, end: dt_date) -> list[dt_date]:
        return list_trading_days(
            start_date=start,
            end_date=end,
            segment=segment,
            settings=settings,
        )

    payload = build_data_quality_gap_audit_payload(
        report,
        bundle_name=bundle.name if bundle is not None else None,
        max_gap_bars=max_gap_bars,
        frame_lookup=frame_lookup,
        day_frame_lookup=provider._day_frame,
        trading_days_lookup=trading_days_lookup,
        membership_lookup=membership_lookup,
        membership_history_available=membership_available,
    )
    if write_files:
        payload["files"] = write_data_quality_gap_audit_files(payload, output_dir=output_dir)
    return payload
