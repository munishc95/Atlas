from __future__ import annotations

from datetime import date as dt_date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import (
    DataQualityReport,
    DataUpdateRun,
    Job,
    OperateEvent,
    PaperRun,
    PaperState,
    ProviderUpdateRun,
    UpstoxTokenRequestRun,
)
from app.services.trading_calendar import (
    compute_next_scheduled_run_ist,
    get_session as calendar_get_session,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
)
from app.services.upstox_auth import token_status as upstox_token_status


ALLOWED_SEVERITIES = {"INFO", "WARN", "ERROR"}
ALLOWED_CATEGORIES = {"DATA", "EXECUTION", "POLICY", "SYSTEM"}
IST_ZONE = ZoneInfo("Asia/Kolkata")


def _serialize_token_request(row: UpstoxTokenRequestRun) -> dict[str, Any]:
    status = str(row.status or "").upper()
    if status == "REQUESTED":
        status = "PENDING"
    if status == "FAILED":
        status = "ERROR"
    return {
        "id": row.id,
        "provider_kind": row.provider_kind,
        "status": status,
        "requested_at": row.requested_at.isoformat() if row.requested_at is not None else None,
        "authorization_expiry": (
            row.authorization_expiry.isoformat() if row.authorization_expiry is not None else None
        ),
        "approved_at": row.approved_at.isoformat() if row.approved_at is not None else None,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at is not None else None,
        "resolution_reason": row.resolution_reason,
        "notifier_url": row.notifier_url,
        "client_id": row.client_id,
        "user_id": row.user_id,
        "correlation_nonce": row.correlation_nonce,
        "last_error": row.last_error,
        "created_at": row.created_at.isoformat() if row.created_at is not None else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at is not None else None,
    }


def _latest_token_request_run(session: Session) -> UpstoxTokenRequestRun | None:
    return session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == "UPSTOX")
        .order_by(UpstoxTokenRequestRun.requested_at.desc(), UpstoxTokenRequestRun.id.desc())
    ).first()


def _weekly_eval_day_for_week(
    *,
    target_day: dt_date,
    day_of_week: int,
    segment: str,
    settings: Settings,
) -> dt_date:
    anchor = target_day - timedelta(days=target_day.weekday()) + timedelta(days=day_of_week % 7)
    if is_trading_day(anchor, segment=segment, settings=settings):
        return anchor
    return next_trading_day(anchor, segment=segment, settings=settings)


def _compute_next_auto_eval_run_ist(
    *,
    auto_eval_enabled: bool,
    auto_eval_frequency: str,
    auto_eval_day_of_week: int,
    auto_eval_time_ist: str,
    last_run_date: str | None,
    segment: str,
    settings: Settings,
) -> str | None:
    if not auto_eval_enabled:
        return None
    now = datetime.now(IST_ZONE)
    try:
        run_time = datetime.strptime(auto_eval_time_ist, "%H:%M").time()
    except ValueError:
        run_time = datetime.strptime("16:00", "%H:%M").time()
    frequency = str(auto_eval_frequency or "WEEKLY").strip().upper()
    for offset in range(0, 40):
        day = now.date() + timedelta(days=offset)
        if not is_trading_day(day, segment=segment, settings=settings):
            continue
        if frequency == "DAILY":
            due = True
        else:
            due = day == _weekly_eval_day_for_week(
                target_day=day,
                day_of_week=auto_eval_day_of_week,
                segment=segment,
                settings=settings,
            )
        if not due:
            continue
        if day == now.date() and now.time() >= run_time:
            continue
        if isinstance(last_run_date, str) and last_run_date == day.isoformat():
            continue
        return datetime.combine(day, run_time, tzinfo=IST_ZONE).isoformat()
    return None


def _safe_upper(value: str, *, allowed: set[str], fallback: str) -> str:
    token = str(value or "").strip().upper()
    if token in allowed:
        return token
    return fallback


def emit_operate_event(
    session: Session,
    *,
    severity: str,
    category: str,
    message: str,
    details: dict[str, Any] | None = None,
    correlation_id: str | None = None,
    commit: bool = False,
) -> OperateEvent:
    event = OperateEvent(
        severity=_safe_upper(severity, allowed=ALLOWED_SEVERITIES, fallback="INFO"),
        category=_safe_upper(category, allowed=ALLOWED_CATEGORIES, fallback="SYSTEM"),
        message=str(message).strip()[:256],
        details_json=dict(details or {}),
        correlation_id=(str(correlation_id)[:64] if correlation_id else None),
    )
    session.add(event)
    if commit:
        session.commit()
        session.refresh(event)
    return event


def list_operate_events(
    session: Session,
    *,
    since: datetime | None = None,
    severity: str | None = None,
    category: str | None = None,
    limit: int = 200,
) -> list[OperateEvent]:
    stmt = select(OperateEvent).order_by(OperateEvent.ts.desc(), OperateEvent.id.desc())
    if since is not None:
        stmt = stmt.where(OperateEvent.ts >= since)
    if severity:
        stmt = stmt.where(
            OperateEvent.severity
            == _safe_upper(severity, allowed=ALLOWED_SEVERITIES, fallback="INFO")
        )
    if category:
        stmt = stmt.where(
            OperateEvent.category
            == _safe_upper(category, allowed=ALLOWED_CATEGORIES, fallback="SYSTEM")
        )
    stmt = stmt.limit(max(1, min(int(limit), 500)))
    return list(session.exec(stmt).all())


def _latest_data_quality_report(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
) -> DataQualityReport | None:
    stmt = select(DataQualityReport)
    if bundle_id is not None:
        stmt = stmt.where(DataQualityReport.bundle_id == bundle_id)
    if timeframe:
        stmt = stmt.where(DataQualityReport.timeframe == timeframe)
    stmt = stmt.order_by(DataQualityReport.created_at.desc(), DataQualityReport.id.desc()).limit(1)
    return session.exec(stmt).first()


def _latest_data_update_run(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
) -> DataUpdateRun | None:
    stmt = select(DataUpdateRun)
    if bundle_id is not None:
        stmt = stmt.where(DataUpdateRun.bundle_id == bundle_id)
    if timeframe:
        stmt = stmt.where(DataUpdateRun.timeframe == timeframe)
    stmt = stmt.order_by(DataUpdateRun.created_at.desc(), DataUpdateRun.id.desc()).limit(1)
    return session.exec(stmt).first()


def _latest_provider_update_run(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
) -> ProviderUpdateRun | None:
    stmt = select(ProviderUpdateRun)
    if bundle_id is not None:
        stmt = stmt.where(ProviderUpdateRun.bundle_id == bundle_id)
    if timeframe:
        stmt = stmt.where(ProviderUpdateRun.timeframe == timeframe)
    stmt = stmt.order_by(ProviderUpdateRun.created_at.desc(), ProviderUpdateRun.id.desc()).limit(1)
    return session.exec(stmt).first()


def _latest_operate_run_provider_stage_status(session: Session) -> str | None:
    row = session.exec(
        select(Job)
        .where(Job.type == "operate_run")
        .where(Job.status.in_(["SUCCEEDED", "DONE"]))
        .order_by(Job.ended_at.desc(), Job.created_at.desc())
        .limit(1)
    ).first()
    if row is None or not isinstance(row.result_json, dict):
        return None
    summary = row.result_json.get("summary")
    if not isinstance(summary, dict):
        return None
    value = summary.get("provider_stage_status")
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def get_operate_health_summary(
    session: Session,
    settings: Settings,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    # Local import avoids a circular dependency:
    # operate_events -> upstox_token_request -> operate_events.
    from app.services.upstox_token_request import notifier_health_payload

    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    latest_summary = (
        latest_run.summary_json
        if latest_run is not None and isinstance(latest_run.summary_json, dict)
        else {}
    )
    target_bundle_id = (
        bundle_id
        if bundle_id is not None
        else (latest_run.bundle_id if latest_run is not None else None)
    )
    target_timeframe = timeframe
    if not target_timeframe and latest_run is not None:
        summary = latest_run.summary_json if isinstance(latest_run.summary_json, dict) else {}
        tfs = summary.get("timeframes", [])
        if isinstance(tfs, list) and tfs:
            target_timeframe = str(tfs[0])
    if not target_timeframe:
        target_timeframe = "1d"

    latest_quality = _latest_data_quality_report(
        session,
        bundle_id=target_bundle_id,
        timeframe=target_timeframe,
    )
    latest_update = _latest_data_update_run(
        session,
        bundle_id=target_bundle_id,
        timeframe=target_timeframe,
    )
    latest_provider_update = _latest_provider_update_run(
        session,
        bundle_id=target_bundle_id,
        timeframe=target_timeframe,
    )
    provider_token = upstox_token_status(
        session,
        settings=settings,
        allow_env_fallback=True,
    )
    state = session.get(PaperState, 1)
    state_settings = dict(state.settings_json or {}) if state is not None else {}
    safe_mode_on_fail = bool(
        state_settings.get("operate_safe_mode_on_fail", settings.operate_safe_mode_on_fail)
    )
    safe_mode_action = str(
        state_settings.get("operate_safe_mode_action", settings.operate_safe_mode_action)
    )
    operate_mode = str(state_settings.get("operate_mode", settings.operate_mode)).strip().lower()
    auto_run_enabled = bool(
        state_settings.get("operate_auto_run_enabled", settings.operate_auto_run_enabled)
    )
    auto_run_time_ist = str(
        state_settings.get("operate_auto_run_time_ist", settings.operate_auto_run_time_ist)
    )
    auto_run_include_data_updates = bool(
        state_settings.get(
            "operate_auto_run_include_data_updates",
            settings.operate_auto_run_include_data_updates,
        )
    )
    auto_eval_enabled = bool(
        state_settings.get("operate_auto_eval_enabled", settings.operate_auto_eval_enabled)
    )
    auto_eval_frequency = (
        str(state_settings.get("operate_auto_eval_frequency", settings.operate_auto_eval_frequency))
        .strip()
        .upper()
    )
    try:
        auto_eval_day_of_week = (
            int(
                state_settings.get(
                    "operate_auto_eval_day_of_week", settings.operate_auto_eval_day_of_week
                )
            )
            % 7
        )
    except (TypeError, ValueError):
        auto_eval_day_of_week = int(settings.operate_auto_eval_day_of_week) % 7
    auto_eval_time_ist = str(
        state_settings.get("operate_auto_eval_time_ist", settings.operate_auto_eval_time_ist)
    )
    calendar_segment = str(
        state_settings.get("trading_calendar_segment", settings.trading_calendar_segment)
    )
    last_auto_run_date = state_settings.get("operate_last_auto_run_date")
    last_auto_eval_date = state_settings.get("operate_last_auto_eval_date")
    next_scheduled_run_ist = compute_next_scheduled_run_ist(
        auto_run_enabled=auto_run_enabled,
        auto_run_time_ist=auto_run_time_ist,
        last_run_date=(str(last_auto_run_date) if isinstance(last_auto_run_date, str) else None),
        segment=calendar_segment,
        settings=settings,
    )
    next_auto_eval_run_ist = _compute_next_auto_eval_run_ist(
        auto_eval_enabled=auto_eval_enabled,
        auto_eval_frequency=auto_eval_frequency,
        auto_eval_day_of_week=auto_eval_day_of_week,
        auto_eval_time_ist=auto_eval_time_ist,
        last_run_date=(str(last_auto_eval_date) if isinstance(last_auto_eval_date, str) else None),
        segment=calendar_segment,
        settings=settings,
    )
    upstox_auto_renew_enabled = bool(
        state_settings.get("upstox_auto_renew_enabled", settings.upstox_auto_renew_enabled)
    )
    upstox_auto_renew_time_ist = str(
        state_settings.get("upstox_auto_renew_time_ist", settings.upstox_auto_renew_time_ist)
    )
    upstox_auto_renew_threshold = max(
        1,
        int(
            state_settings.get(
                "upstox_auto_renew_if_expires_within_hours",
                settings.upstox_auto_renew_if_expires_within_hours,
            )
        ),
    )
    upstox_auto_renew_lead_hours = max(
        1,
        int(
            state_settings.get(
                "upstox_auto_renew_lead_hours_before_open",
                settings.upstox_auto_renew_lead_hours_before_open,
            )
        ),
    )
    last_upstox_auto_renew_date = state_settings.get("operate_last_upstox_auto_renew_date")
    provider_stage_on_token_invalid = str(
        state_settings.get(
            "operate_provider_stage_on_token_invalid",
            settings.operate_provider_stage_on_token_invalid,
        )
    ).strip().upper() or "SKIP"
    next_upstox_auto_renew_ist = compute_next_scheduled_run_ist(
        auto_run_enabled=upstox_auto_renew_enabled,
        auto_run_time_ist=upstox_auto_renew_time_ist,
        last_run_date=(
            str(last_upstox_auto_renew_date)
            if isinstance(last_upstox_auto_renew_date, str)
            else None
        ),
        segment=calendar_segment,
        settings=settings,
    )
    expires_within_hours: float | None = None
    try:
        expires_at = str(provider_token.get("expires_at") or "").strip()
        if expires_at:
            expiry_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if expiry_dt.tzinfo is None:
                expiry_dt = expiry_dt.replace(tzinfo=timezone.utc)
            expires_within_hours = round(
                (expiry_dt - datetime.now(timezone.utc)).total_seconds() / 3600.0,
                3,
            )
    except ValueError:
        expires_within_hours = None
    latest_token_request = _latest_token_request_run(session)
    upstox_notifier_health = notifier_health_payload(session, settings=settings)
    today_ist = datetime.now(IST_ZONE).date()
    today_is_trading_day = is_trading_day(today_ist, segment=calendar_segment, settings=settings)
    today_session = calendar_get_session(today_ist, segment=calendar_segment, settings=settings)
    next_day = next_trading_day(today_ist, segment=calendar_segment, settings=settings)
    prev_day = previous_trading_day(today_ist, segment=calendar_segment, settings=settings)

    mode = "NORMAL"
    mode_reason: str | None = None
    if (
        latest_quality is not None
        and str(latest_quality.status).upper() == "FAIL"
        and safe_mode_on_fail
    ):
        mode = "SAFE MODE"
        mode_reason = "data_quality_fail_safe_mode"

    since_24h = datetime.now(timezone.utc) - timedelta(hours=24)
    recent_events = list_operate_events(session, since=since_24h, limit=500)
    severity_counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
    for row in recent_events:
        key = str(row.severity).upper()
        severity_counts[key] = severity_counts.get(key, 0) + 1

    duration_map: dict[str, dict[str, Any]] = {}
    for row in recent_events:
        details = row.details_json if isinstance(row.details_json, dict) else {}
        job_kind = str(details.get("job_kind", "")).strip()
        if not job_kind:
            continue
        duration = details.get("duration_seconds")
        try:
            duration_value = float(duration)
        except (TypeError, ValueError):
            continue
        existing = duration_map.get(job_kind)
        if existing is None:
            duration_map[job_kind] = {
                "duration_seconds": duration_value,
                "status": str(details.get("status", "unknown")),
                "ts": row.ts.isoformat(),
            }
        else:
            existing_ts = str(existing.get("ts", ""))
            if row.ts.isoformat() > existing_ts:
                duration_map[job_kind] = {
                    "duration_seconds": duration_value,
                    "status": str(details.get("status", "unknown")),
                    "ts": row.ts.isoformat(),
                }

    provider_stage_status = _latest_operate_run_provider_stage_status(session)

    return {
        "mode": mode,
        "mode_reason": mode_reason,
        "safe_mode_on_fail": safe_mode_on_fail,
        "safe_mode_action": safe_mode_action,
        "operate_mode": operate_mode,
        "calendar_segment": calendar_segment,
        "calendar_today_ist": today_ist.isoformat(),
        "calendar_is_trading_day_today": today_is_trading_day,
        "calendar_session_today": today_session,
        "calendar_next_trading_day": next_day.isoformat(),
        "calendar_previous_trading_day": prev_day.isoformat(),
        "auto_run_enabled": auto_run_enabled,
        "auto_run_time_ist": auto_run_time_ist,
        "auto_run_include_data_updates": auto_run_include_data_updates,
        "last_auto_run_date": last_auto_run_date,
        "next_scheduled_run_ist": next_scheduled_run_ist,
        "auto_eval_enabled": auto_eval_enabled,
        "auto_eval_frequency": auto_eval_frequency,
        "auto_eval_day_of_week": auto_eval_day_of_week,
        "auto_eval_time_ist": auto_eval_time_ist,
        "last_auto_eval_date": last_auto_eval_date,
        "next_auto_eval_run_ist": next_auto_eval_run_ist,
        "active_bundle_id": target_bundle_id,
        "active_timeframe": target_timeframe,
        "latest_data_quality": latest_quality.model_dump() if latest_quality is not None else None,
        "latest_data_update": latest_update.model_dump() if latest_update is not None else None,
        "latest_provider_update": (
            latest_provider_update.model_dump() if latest_provider_update is not None else None
        ),
        "upstox_token_status": provider_token,
        "upstox_token_request_latest": (
            _serialize_token_request(latest_token_request)
            if latest_token_request is not None
            else None
        ),
        "upstox_auto_renew_enabled": upstox_auto_renew_enabled,
        "upstox_auto_renew_time_ist": upstox_auto_renew_time_ist,
        "upstox_auto_renew_if_expires_within_hours": upstox_auto_renew_threshold,
        "upstox_auto_renew_lead_hours_before_open": upstox_auto_renew_lead_hours,
        "operate_provider_stage_on_token_invalid": provider_stage_on_token_invalid,
        "operate_last_upstox_auto_renew_date": last_upstox_auto_renew_date,
        "next_upstox_auto_renew_ist": next_upstox_auto_renew_ist,
        "upstox_token_expires_within_hours": expires_within_hours,
        "upstox_notifier_health": upstox_notifier_health,
        "provider_stage_status": provider_stage_status,
        "latest_paper_run_id": int(latest_run.id)
        if latest_run is not None and latest_run.id is not None
        else None,
        "current_regime": latest_run.regime if latest_run is not None else None,
        "no_trade": latest_summary.get("no_trade", {}),
        "no_trade_triggered": bool(latest_summary.get("no_trade_triggered", False)),
        "no_trade_reasons": latest_summary.get("no_trade_reasons", []),
        "ensemble_weights_source": latest_summary.get("ensemble_weights_source"),
        "ensemble_regime_used": latest_summary.get("ensemble_regime_used"),
        "last_run_step_at": latest_run.asof_ts.isoformat() if latest_run is not None else None,
        "recent_event_counts_24h": severity_counts,
        "fast_mode_enabled": bool(settings.fast_mode_enabled),
        "last_job_durations": duration_map,
    }
