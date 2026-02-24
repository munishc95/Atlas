from __future__ import annotations

from datetime import date as dt_date
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import DataQualityReport
from app.services.data_provenance import provenance_summary
from app.services.data_store import DataStore
from app.services.data_updates import STATUS_FAIL as COVERAGE_FAIL
from app.services.data_updates import STATUS_WARN as COVERAGE_WARN
from app.services.data_updates import compute_data_coverage
from app.services.operate_events import emit_operate_event
from app.services.trading_calendar import (
    get_session as calendar_get_session,
    is_trading_day,
    list_trading_days,
    parse_time_hhmm,
    previous_trading_day,
)


STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"
IST_ZONE = ZoneInfo("Asia/Kolkata")


def _issue(
    *,
    severity: str,
    code: str,
    message: str,
    symbol: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "severity": str(severity).upper(),
        "code": str(code),
        "message": str(message),
    }
    if symbol:
        payload["symbol"] = str(symbol).upper()
    if details:
        payload["details"] = details
    return payload


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _calendar_segment(*, settings: Settings, overrides: dict[str, Any] | None = None) -> str:
    state = overrides or {}
    token = str(state.get("trading_calendar_segment", settings.trading_calendar_segment)).strip().upper()
    return token or "EQUITIES"


def _stale_limit_minutes(
    timeframe: str,
    *,
    settings: Settings,
    overrides: dict[str, Any] | None = None,
) -> int:
    tf = str(timeframe).strip().lower()
    state = overrides or {}
    if tf in {"4h_ish", "2h", "1h", "30m", "15m"}:
        return max(
            15,
            _safe_int(
                state.get(
                    "data_quality_max_stale_minutes_intraday",
                    state.get(
                        "operate_max_stale_minutes_4h_ish",
                        settings.data_quality_max_stale_minutes_intraday,
                    ),
                ),
                settings.data_quality_max_stale_minutes_intraday,
            ),
        )
    return max(
        60,
        _safe_int(
            state.get(
                "data_quality_max_stale_minutes_1d",
                state.get("operate_max_stale_minutes_1d", settings.data_quality_max_stale_minutes_1d),
            ),
            settings.data_quality_max_stale_minutes_1d,
        ),
    )


def _stale_severity(
    *,
    settings: Settings,
    overrides: dict[str, Any] | None = None,
) -> str:
    state = overrides or {}
    mode = str(state.get("operate_mode", settings.operate_mode)).strip().lower()
    explicit_token = str(state.get("data_quality_stale_severity", "")).strip().upper()
    explicit_override = bool(state.get("data_quality_stale_severity_override", False))
    default_token = str(settings.data_quality_stale_severity).strip().upper()

    if mode == "live" and not explicit_override:
        return STATUS_FAIL

    token = explicit_token if explicit_token in {STATUS_WARN, STATUS_FAIL} else default_token
    return STATUS_FAIL if token == STATUS_FAIL else STATUS_WARN


def _coverage_pct(
    frame: pd.DataFrame,
    *,
    settings: Settings,
    segment: str,
) -> float:
    if frame.empty:
        return 0.0
    dt = pd.to_datetime(frame["datetime"], utc=True).sort_values()
    first = dt.iloc[0].date()
    last = dt.iloc[-1].date()
    expected = len(
        list_trading_days(
            start_date=first,
            end_date=last,
            segment=segment,
            settings=settings,
        )
    )
    observed = len(pd.Series(dt.dt.date).drop_duplicates())
    if expected <= 0:
        return 100.0
    return float(max(0.0, min(100.0, (observed / expected) * 100.0)))


def _gap_issues_daily(
    *,
    symbol: str,
    frame: pd.DataFrame,
    max_gap_bars: int,
    settings: Settings,
    segment: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if frame.empty:
        return issues
    dt = pd.to_datetime(frame["datetime"], utc=True).sort_values().drop_duplicates()
    if len(dt) < 2:
        return issues
    for idx in range(1, len(dt)):
        prev_day = dt.iloc[idx - 1].date()
        next_day = dt.iloc[idx].date()
        expected_days = list_trading_days(
            start_date=prev_day,
            end_date=next_day,
            segment=segment,
            settings=settings,
        )
        missing_days = [day for day in expected_days if prev_day < day < next_day]
        missing_count = len(missing_days)
        if missing_count > max_gap_bars:
            issues.append(
                _issue(
                    severity=STATUS_FAIL,
                    code="gap_exceeds_threshold",
                    symbol=symbol,
                    message=(
                        f"Detected {missing_count} missing trading bars between "
                        f"{prev_day.isoformat()} and {next_day.isoformat()}."
                    ),
                    details={
                        "missing_bars": missing_count,
                        "max_gap_bars": int(max_gap_bars),
                        "missing_dates": [day.isoformat() for day in missing_days[:10]],
                    },
                )
            )
    return issues


def _outlier_issues(
    *,
    symbol: str,
    frame: pd.DataFrame,
    zscore_threshold: float,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if frame.empty or len(frame) < 10:
        return issues
    close = pd.Series(frame["close"], dtype=float)
    returns = close.pct_change().dropna()
    if returns.empty:
        return issues
    std = float(returns.std(ddof=0))
    if std <= 1e-12:
        return issues
    z = ((returns - returns.mean()) / std).abs()
    outlier_count = int((z > zscore_threshold).sum())
    if outlier_count > 0:
        issues.append(
            _issue(
                severity=STATUS_WARN,
                code="return_outliers",
                symbol=symbol,
                message=f"Found {outlier_count} return outliers (|z| > {zscore_threshold:.2f}).",
                details={"outlier_count": outlier_count, "zscore_threshold": zscore_threshold},
            )
        )
    split_like = int((returns.abs() >= 0.35).sum())
    if split_like > 0:
        issues.append(
            _issue(
                severity=STATUS_WARN,
                code="corporate_action_anomaly",
                symbol=symbol,
                message="Detected split-like jump(s) in close returns. Verify corporate action adjustments.",
                details={"jump_count": split_like, "jump_threshold": 0.35},
            )
        )
    return issues


def _validate_symbol_frame(
    *,
    symbol: str,
    frame: pd.DataFrame,
    timeframe: str,
    max_gap_bars: int,
    zscore_threshold: float,
    settings: Settings,
    segment: str,
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if frame.empty:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="missing_ohlcv",
                symbol=symbol,
                message="No OHLCV rows available.",
            )
        )
        return issues

    dt = pd.to_datetime(frame["datetime"], utc=True)
    if not dt.is_monotonic_increasing:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="non_monotonic_timestamps",
                symbol=symbol,
                message="Timestamps are not strictly increasing.",
            )
        )

    duplicate_count = int(dt.duplicated().sum())
    if duplicate_count > 0:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="duplicate_timestamps",
                symbol=symbol,
                message=f"Found {duplicate_count} duplicate timestamps.",
                details={"duplicate_count": duplicate_count},
            )
        )

    high = pd.Series(frame["high"], dtype=float)
    low = pd.Series(frame["low"], dtype=float)
    open_px = pd.Series(frame["open"], dtype=float)
    close_px = pd.Series(frame["close"], dtype=float)
    invalid_count = int(((high < low) | (open_px < low) | (open_px > high) | (close_px < low) | (close_px > high)).sum())
    if invalid_count > 0:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="invalid_ohlc_ranges",
                symbol=symbol,
                message=f"Found {invalid_count} rows with invalid OHLC bounds.",
                details={"invalid_rows": invalid_count},
            )
        )

    if str(timeframe).lower() == "1d":
        issues.extend(
            _gap_issues_daily(
                symbol=symbol,
                frame=frame,
                max_gap_bars=max_gap_bars,
                settings=settings,
                segment=segment,
            )
        )
    issues.extend(_outlier_issues(symbol=symbol, frame=frame, zscore_threshold=zscore_threshold))
    return issues


def run_data_quality_report(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    overrides: dict[str, Any] | None = None,
    reference_ts: datetime | None = None,
    correlation_id: str | None = None,
) -> DataQualityReport:
    tf = str(timeframe).strip() or "1d"
    state = overrides or {}
    max_gap_bars = max(
        0,
        _safe_int(state.get("operate_max_gap_bars", settings.operate_max_gap_bars), settings.operate_max_gap_bars),
    )
    zscore_threshold = max(
        2.0,
        _safe_float(state.get("operate_outlier_zscore", settings.operate_outlier_zscore), settings.operate_outlier_zscore),
    )
    stale_limit = _stale_limit_minutes(tf, settings=settings, overrides=state)
    stale_severity = _stale_severity(settings=settings, overrides=state)
    segment = _calendar_segment(settings=settings, overrides=state)
    operate_mode = str(state.get("operate_mode", settings.operate_mode)).strip().lower()
    low_confidence_threshold = max(
        0.0,
        min(
            100.0,
            _safe_float(
                state.get(
                    "data_quality_confidence_fail_threshold",
                    settings.data_quality_confidence_fail_threshold,
                ),
                settings.data_quality_confidence_fail_threshold,
            ),
        ),
    )

    symbols = store.get_bundle_symbols(session, bundle_id, timeframe=tf)
    issues: list[dict[str, Any]] = []
    last_bar_ts: datetime | None = None
    coverage_values: list[float] = []

    for symbol in symbols:
        frame = store.load_ohlcv(symbol=symbol, timeframe=tf)
        if not frame.empty:
            frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
            symbol_last = frame["datetime"].max().to_pydatetime()
            if last_bar_ts is None or symbol_last > last_bar_ts:
                last_bar_ts = symbol_last
            coverage_values.append(_coverage_pct(frame, settings=settings, segment=segment))
        issues.extend(
            _validate_symbol_frame(
                symbol=symbol,
                frame=frame,
                timeframe=tf,
                max_gap_bars=max_gap_bars,
                zscore_threshold=zscore_threshold,
                settings=settings,
                segment=segment,
            )
        )

    if not symbols:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="empty_bundle",
                message=f"No symbols found for bundle_id={bundle_id} timeframe={tf}.",
            )
        )

    now = reference_ts if reference_ts is not None else datetime.now(timezone.utc)
    if last_bar_ts is None:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="missing_last_bar",
                message="No last bar timestamp available for staleness check.",
            )
        )
    else:
        age_minutes = (now - last_bar_ts).total_seconds() / 60.0
        stale = age_minutes > stale_limit
        stale_context: dict[str, Any] = {"age_minutes": age_minutes, "stale_limit_minutes": stale_limit}
        if tf.lower() == "1d":
            now_ist = now.astimezone(IST_ZONE)
            last_bar_day_ist = last_bar_ts.astimezone(IST_ZONE).date()
            today_ist = now_ist.date()
            if is_trading_day(today_ist, segment=segment, settings=settings):
                session_info = calendar_get_session(today_ist, segment=segment, settings=settings)
                close_cutoff = parse_time_hhmm(
                    session_info.get("close_time"),
                    default=settings.nse_equities_close_time_ist,
                )
                expected_day = (
                    today_ist
                    if now_ist.time() >= close_cutoff
                    else previous_trading_day(today_ist, segment=segment, settings=settings)
                )
            else:
                expected_day = previous_trading_day(today_ist, segment=segment, settings=settings)
            stale = last_bar_day_ist < expected_day
            stale_context.update(
                {
                    "last_bar_day_ist": last_bar_day_ist.isoformat(),
                    "expected_latest_trading_day": expected_day.isoformat(),
                    "calendar_segment": segment,
                }
            )
        if stale:
            issues.append(
                _issue(
                    severity=stale_severity,
                    code="stale_data",
                    message=(
                        f"Latest bar is stale by {age_minutes:.0f} minutes "
                        f"(limit {stale_limit} minutes)."
                    ),
                    details=stale_context,
                )
            )

    coverage_summary = compute_data_coverage(
        session=session,
        settings=settings,
        store=store,
        bundle_id=bundle_id,
        timeframe=tf,
        overrides=state,
        reference_ts=now,
        top_n=25,
    )
    coverage_pct = float(coverage_summary.get("coverage_pct", 0.0))
    missing_pct = float(coverage_summary.get("missing_pct", 0.0))
    coverage_status = str(coverage_summary.get("status", STATUS_OK)).upper()
    missing_symbols = [str(item) for item in coverage_summary.get("missing_symbols", [])]
    inactive_symbols = [str(item) for item in coverage_summary.get("inactive_symbols", [])]
    latest_expected_day: dt_date | None = None
    latest_expected_day_token = coverage_summary.get("expected_latest_trading_day")
    if isinstance(latest_expected_day_token, str) and latest_expected_day_token:
        try:
            latest_expected_day = dt_date.fromisoformat(latest_expected_day_token)
        except ValueError:
            latest_expected_day = None
    provenance = provenance_summary(
        session,
        bundle_id=bundle_id,
        timeframe=tf,
        latest_day=latest_expected_day,
        low_conf_threshold=float(low_confidence_threshold),
    )
    coverage_by_source_provider = {
        str(key): float(value)
        for key, value in dict(provenance.get("coverage_by_source_provider", {})).items()
    }
    low_confidence_days_count = int(provenance.get("low_confidence_days_count", 0) or 0)
    low_confidence_symbols_count = int(provenance.get("low_confidence_symbols_count", 0) or 0)
    latest_day_all_low_confidence = bool(provenance.get("latest_day_all_low_confidence", False))
    if coverage_status == COVERAGE_FAIL:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="coverage_below_fail_threshold",
                message=(
                    f"{missing_pct:.2f}% symbols missing latest trading bar, exceeding fail threshold."
                ),
                details={
                    "coverage_pct": coverage_pct,
                    "missing_pct": missing_pct,
                    "missing_symbols_count": len(missing_symbols),
                    "missing_symbols_sample": missing_symbols[:20],
                    "expected_latest_trading_day": coverage_summary.get("expected_latest_trading_day"),
                },
            )
        )
    elif coverage_status == COVERAGE_WARN:
        issues.append(
            _issue(
                severity=STATUS_WARN,
                code="coverage_below_warn_threshold",
                message=(
                    f"{missing_pct:.2f}% symbols missing latest trading bar, exceeding warning threshold."
                ),
                details={
                    "coverage_pct": coverage_pct,
                    "missing_pct": missing_pct,
                    "missing_symbols_count": len(missing_symbols),
                    "missing_symbols_sample": missing_symbols[:20],
                    "expected_latest_trading_day": coverage_summary.get("expected_latest_trading_day"),
                },
            )
        )
    if inactive_symbols:
        issues.append(
            _issue(
                severity=STATUS_WARN,
                code="inactive_symbols_detected",
                message=f"{len(inactive_symbols)} symbols marked inactive for selection due to stale data.",
                details={
                    "inactive_symbols_count": len(inactive_symbols),
                    "inactive_symbols_sample": inactive_symbols[:20],
                    "inactive_after_missing_days": coverage_summary.get("thresholds", {}).get(
                        "inactive_after_missing_days"
                    ),
                },
            )
        )
    if operate_mode == "live" and coverage_by_source_provider:
        fallback_pct = sum(
            float(value)
            for key, value in coverage_by_source_provider.items()
            if str(key).upper() not in {"UPSTOX"}
        )
        if fallback_pct > 0.0:
            issues.append(
                _issue(
                    severity=STATUS_WARN,
                    code="fallback_source_live_mode",
                    message="Latest trading day includes fallback/inbox source data in live mode.",
                    details={
                        "coverage_by_source_provider": coverage_by_source_provider,
                        "fallback_pct": round(float(fallback_pct), 3),
                    },
                )
            )
    if operate_mode == "live" and latest_day_all_low_confidence:
        issues.append(
            _issue(
                severity=STATUS_FAIL,
                code="low_confidence_latest_day",
                message=(
                    "Latest trading day confidence is below threshold for all symbols in live mode."
                ),
                details={
                    "confidence_fail_threshold": float(low_confidence_threshold),
                    "low_confidence_symbols_count": int(low_confidence_symbols_count),
                    "coverage_by_source_provider": coverage_by_source_provider,
                },
            )
        )

    has_fail = any(str(item.get("severity", "")).upper() == STATUS_FAIL for item in issues)
    has_warn = any(str(item.get("severity", "")).upper() == STATUS_WARN for item in issues)
    status = STATUS_FAIL if has_fail else (STATUS_WARN if has_warn else STATUS_OK)
    if coverage_values and coverage_pct <= 0.0:
        coverage_pct = float(np.mean(coverage_values))

    report = DataQualityReport(
        bundle_id=bundle_id,
        timeframe=tf,
        status=status,
        issues_json=issues,
        last_bar_ts=last_bar_ts,
        coverage_pct=coverage_pct,
        checked_symbols=len(symbols),
        total_symbols=len(symbols),
        coverage_by_source_json=coverage_by_source_provider,
        low_confidence_days_count=int(low_confidence_days_count),
        low_confidence_symbols_count=int(low_confidence_symbols_count),
    )
    session.add(report)

    if status == STATUS_FAIL:
        emit_operate_event(
            session,
            severity="ERROR",
            category="DATA",
            message="Data quality report failed; safe-mode guardrails may activate.",
            details={
                "bundle_id": bundle_id,
                "timeframe": tf,
                "issue_count": len(issues),
                "report_status": status,
            },
            correlation_id=correlation_id,
        )
    elif status == STATUS_WARN:
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="Data quality report has warnings.",
            details={
                "bundle_id": bundle_id,
                "timeframe": tf,
                "issue_count": len(issues),
                "report_status": status,
            },
            correlation_id=correlation_id,
        )

    session.commit()
    session.refresh(report)
    return report


def get_latest_data_quality_report(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
) -> DataQualityReport | None:
    return session.exec(
        select(DataQualityReport)
        .where(DataQualityReport.bundle_id == bundle_id)
        .where(DataQualityReport.timeframe == str(timeframe))
        .order_by(DataQualityReport.created_at.desc(), DataQualityReport.id.desc())
    ).first()


def list_data_quality_history(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
    days: int = 7,
) -> list[DataQualityReport]:
    since = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    stmt = select(DataQualityReport).where(DataQualityReport.created_at >= since)
    if bundle_id is not None:
        stmt = stmt.where(DataQualityReport.bundle_id == bundle_id)
    if timeframe:
        stmt = stmt.where(DataQualityReport.timeframe == str(timeframe))
    stmt = stmt.order_by(DataQualityReport.created_at.desc(), DataQualityReport.id.desc())
    return list(session.exec(stmt).all())
