from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import DataQualityReport
from app.services.data_store import DataStore
from app.services.operate_events import emit_operate_event


STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


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


def _stale_limit_minutes(
    timeframe: str,
    *,
    settings: Settings,
    overrides: dict[str, Any] | None = None,
) -> int:
    tf = str(timeframe).strip().lower()
    state = overrides or {}
    if tf == "4h_ish":
        return max(
            30,
            _safe_int(
                state.get("operate_max_stale_minutes_4h_ish", settings.operate_max_stale_minutes_4h_ish),
                settings.operate_max_stale_minutes_4h_ish,
            ),
        )
    return max(
        60,
        _safe_int(
            state.get("operate_max_stale_minutes_1d", settings.operate_max_stale_minutes_1d),
            settings.operate_max_stale_minutes_1d,
        ),
    )


def _coverage_pct(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    dt = pd.to_datetime(frame["datetime"], utc=True).sort_values()
    first = dt.iloc[0].date()
    last = dt.iloc[-1].date()
    expected = len(pd.bdate_range(first, last))
    observed = len(pd.Series(dt.dt.date).drop_duplicates())
    if expected <= 0:
        return 100.0
    return float(max(0.0, min(100.0, (observed / expected) * 100.0)))


def _gap_issues_daily(
    *,
    symbol: str,
    frame: pd.DataFrame,
    max_gap_bars: int,
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
        business_gap = int(np.busday_count(prev_day, next_day)) - 1
        if business_gap > max_gap_bars:
            issues.append(
                _issue(
                    severity=STATUS_FAIL,
                    code="gap_exceeds_threshold",
                    symbol=symbol,
                    message=(
                        f"Detected {business_gap} missing business bars between "
                        f"{prev_day.isoformat()} and {next_day.isoformat()}."
                    ),
                    details={"missing_bars": business_gap, "max_gap_bars": int(max_gap_bars)},
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
        issues.extend(_gap_issues_daily(symbol=symbol, frame=frame, max_gap_bars=max_gap_bars))
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
            coverage_values.append(_coverage_pct(frame))
        issues.extend(
            _validate_symbol_frame(
                symbol=symbol,
                frame=frame,
                timeframe=tf,
                max_gap_bars=max_gap_bars,
                zscore_threshold=zscore_threshold,
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
        if age_minutes > stale_limit:
            issues.append(
                _issue(
                    severity=STATUS_WARN,
                    code="stale_data",
                    message=(
                        f"Latest bar is stale by {age_minutes:.0f} minutes "
                        f"(limit {stale_limit} minutes)."
                    ),
                    details={"age_minutes": age_minutes, "stale_limit_minutes": stale_limit},
                )
            )

    has_fail = any(str(item.get("severity", "")).upper() == STATUS_FAIL for item in issues)
    has_warn = any(str(item.get("severity", "")).upper() == STATUS_WARN for item in issues)
    status = STATUS_FAIL if has_fail else (STATUS_WARN if has_warn else STATUS_OK)
    coverage_pct = float(np.mean(coverage_values)) if coverage_values else 0.0

    report = DataQualityReport(
        bundle_id=bundle_id,
        timeframe=tf,
        status=status,
        issues_json=issues,
        last_bar_ts=last_bar_ts,
        coverage_pct=coverage_pct,
        checked_symbols=len(symbols),
        total_symbols=len(symbols),
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
