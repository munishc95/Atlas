from __future__ import annotations

from datetime import date, datetime, timedelta
import threading
from typing import Any
from zoneinfo import ZoneInfo

from redis.exceptions import RedisError
from rq import Queue
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import DatasetBundle, PaperRun, PaperState
from app.db.session import engine
from app.services.jobs import create_job
from app.services.operate_events import emit_operate_event
from app.services.upstox_auth import token_status as upstox_token_status
from app.services.trading_calendar import (
    compute_next_scheduled_run_ist as calendar_next_scheduled_run_ist,
    get_session as calendar_get_session,
    is_trading_day,
    next_trading_day,
    parse_time_hhmm,
)

IST_ZONE = ZoneInfo("Asia/Kolkata")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def compute_next_scheduled_run_ist(
    *,
    auto_run_enabled: bool,
    auto_run_time_ist: str,
    last_run_date: str | None,
    segment: str = "EQUITIES",
    settings: Settings | None = None,
    now_ist: datetime | None = None,
) -> str | None:
    return calendar_next_scheduled_run_ist(
        auto_run_enabled=auto_run_enabled,
        auto_run_time_ist=auto_run_time_ist,
        last_run_date=last_run_date,
        segment=segment,
        now_ist=now_ist,
        settings=settings or get_settings(),
    )


def _weekly_eval_day_for_week(
    *,
    target_day: date,
    day_of_week: int,
    segment: str,
    settings: Settings,
) -> date:
    anchor = target_day - timedelta(days=target_day.weekday()) + timedelta(days=day_of_week % 7)
    if is_trading_day(anchor, segment=segment, settings=settings):
        return anchor
    return next_trading_day(anchor, segment=segment, settings=settings)


def _weekly_eval_matches_day(
    *,
    day: date,
    day_of_week: int,
    segment: str,
    settings: Settings,
) -> bool:
    return day == _weekly_eval_day_for_week(
        target_day=day,
        day_of_week=day_of_week,
        segment=segment,
        settings=settings,
    )


def compute_next_auto_eval_run_ist(
    *,
    auto_eval_enabled: bool,
    auto_eval_frequency: str,
    auto_eval_day_of_week: int,
    auto_eval_time_ist: str,
    last_run_date: str | None,
    segment: str = "EQUITIES",
    settings: Settings | None = None,
    now_ist: datetime | None = None,
) -> str | None:
    if not auto_eval_enabled:
        return None
    cfg = settings or get_settings()
    now = now_ist or datetime.now(IST_ZONE)
    run_time = parse_time_hhmm(auto_eval_time_ist, default=cfg.operate_auto_eval_time_ist)
    frequency = str(auto_eval_frequency or "WEEKLY").strip().upper()

    for offset in range(0, 40):
        day = now.date() + timedelta(days=offset)
        if not is_trading_day(day, segment=segment, settings=cfg):
            continue
        if frequency == "DAILY":
            due = True
        else:
            due = _weekly_eval_matches_day(
                day=day,
                day_of_week=auto_eval_day_of_week,
                segment=segment,
                settings=cfg,
            )
        if not due:
            continue
        if day == now.date() and now.time() >= run_time:
            continue
        if isinstance(last_run_date, str) and last_run_date == day.isoformat():
            continue
        return datetime.combine(day, run_time, tzinfo=IST_ZONE).isoformat()
    return None


def _resolve_scheduler_context(session: Session) -> dict[str, Any]:
    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    state = session.get(PaperState, 1)
    state_settings = dict(state.settings_json or {}) if state is not None else {}

    bundle_id: int | None = None
    timeframe = "1d"
    regime = "TREND_UP"
    policy_id: int | None = None
    active_ensemble_id: int | None = None

    if latest_run is not None:
        bundle_id = latest_run.bundle_id
        regime = str(latest_run.regime or regime)
        summary = latest_run.summary_json if isinstance(latest_run.summary_json, dict) else {}
        tfs = summary.get("timeframes", [])
        if isinstance(tfs, list) and tfs:
            timeframe = str(tfs[0] or timeframe)

    if bundle_id is None:
        row = session.exec(select(DatasetBundle).order_by(DatasetBundle.created_at.desc())).first()
        if row is not None and row.id is not None:
            bundle_id = int(row.id)

    try:
        if state_settings.get("active_policy_id") is not None:
            policy_id = int(state_settings.get("active_policy_id"))
    except (TypeError, ValueError):
        policy_id = None
    try:
        if state_settings.get("active_ensemble_id") is not None:
            active_ensemble_id = int(state_settings.get("active_ensemble_id"))
    except (TypeError, ValueError):
        active_ensemble_id = None

    return {
        "bundle_id": bundle_id,
        "timeframe": timeframe,
        "regime": regime,
        "policy_id": policy_id,
        "active_ensemble_id": active_ensemble_id,
        "state": state,
        "state_settings": state_settings,
    }


def _enqueue_job(
    *,
    session: Session,
    queue: Queue,
    settings: Settings,
    job_type: str,
    task_path: str,
    payload: dict[str, Any],
) -> str:
    job = create_job(session, job_type)
    queue.enqueue(
        task_path,
        job.id,
        payload,
        max_runtime_seconds=settings.job_default_timeout_seconds,
        job_timeout=settings.job_default_timeout_seconds,
    )
    return job.id


def run_auto_operate_once(
    *, session: Session, queue: Queue, settings: Settings, now_ist: datetime | None = None
) -> bool:
    context = _resolve_scheduler_context(session)
    state: PaperState | None = context["state"]
    if state is None:
        return False

    state_settings = context["state_settings"]
    segment = str(state_settings.get("trading_calendar_segment", settings.trading_calendar_segment))
    now = now_ist or datetime.now(IST_ZONE)
    today = now.date()
    bundle_id = context["bundle_id"]
    timeframe = str(context["timeframe"] or "1d")
    regime = str(context["regime"] or "TREND_UP")
    policy_id = context["policy_id"]
    active_ensemble_id = context["active_ensemble_id"]
    if not is_trading_day(today, segment=segment, settings=settings):
        return False

    triggered = False
    merged = dict(state_settings)

    auto_run_enabled = bool(
        state_settings.get("operate_auto_run_enabled", settings.operate_auto_run_enabled)
    )
    include_data_updates = bool(
        state_settings.get(
            "operate_auto_run_include_data_updates",
            settings.operate_auto_run_include_data_updates,
        )
    )
    provider_updates_enabled = bool(
        state_settings.get(
            "data_updates_provider_enabled",
            settings.data_updates_provider_enabled,
        )
    )
    raw_provider_timeframes = state_settings.get("data_updates_provider_timeframes")
    if isinstance(raw_provider_timeframes, list):
        provider_timeframes = {
            str(item).strip().lower() for item in raw_provider_timeframes if str(item).strip()
        }
    else:
        provider_timeframe_token = str(
            state_settings.get(
                "data_updates_provider_timeframe_enabled",
                settings.data_updates_provider_timeframe_enabled,
            )
        )
        provider_timeframes = {
            str(item).strip().lower()
            for item in provider_timeframe_token.split(",")
            if str(item).strip()
        }
    provider_timeframe_allowed = str(timeframe).strip().lower() in provider_timeframes
    run_time = parse_time_hhmm(
        str(state_settings.get("operate_auto_run_time_ist", settings.operate_auto_run_time_ist)),
        default=settings.operate_auto_run_time_ist,
    )
    last_run_date = state_settings.get("operate_last_auto_run_date")
    auto_renew_enabled = bool(
        state_settings.get("upstox_auto_renew_enabled", settings.upstox_auto_renew_enabled)
    )
    auto_renew_only_when_provider_enabled = bool(
        state_settings.get(
            "upstox_auto_renew_only_when_provider_enabled",
            settings.upstox_auto_renew_only_when_provider_enabled,
        )
    )
    auto_renew_time = parse_time_hhmm(
        str(state_settings.get("upstox_auto_renew_time_ist", settings.upstox_auto_renew_time_ist)),
        default=settings.upstox_auto_renew_time_ist,
    )
    auto_renew_threshold_hours = max(
        1,
        _safe_int(
            state_settings.get(
                "upstox_auto_renew_if_expires_within_hours",
                settings.upstox_auto_renew_if_expires_within_hours,
            ),
            settings.upstox_auto_renew_if_expires_within_hours,
        ),
    )
    last_auto_renew_date = state_settings.get("operate_last_upstox_auto_renew_date")
    if (
        auto_renew_enabled
        and now.time() >= auto_renew_time
        and not (
            isinstance(last_auto_renew_date, str) and last_auto_renew_date == today.isoformat()
        )
        and (provider_updates_enabled or not auto_renew_only_when_provider_enabled)
    ):
        token = upstox_token_status(
            session,
            settings=settings,
            allow_env_fallback=True,
        )
        should_request = False
        reason = "token_valid"
        if not bool(token.get("connected")):
            should_request = True
            reason = "token_missing"
        elif bool(token.get("is_expired")):
            should_request = True
            reason = "token_expired"
        else:
            expiry_raw = token.get("expires_at")
            if isinstance(expiry_raw, str) and expiry_raw.strip():
                try:
                    expiry_ts = datetime.fromisoformat(expiry_raw.replace("Z", "+00:00"))
                    if expiry_ts.tzinfo is None:
                        expiry_ts = expiry_ts.replace(tzinfo=ZoneInfo("UTC"))
                    hours_left = (
                        expiry_ts - now.astimezone(ZoneInfo("UTC"))
                    ).total_seconds() / 3600.0
                    if hours_left <= float(auto_renew_threshold_hours):
                        should_request = True
                        reason = "token_expires_soon"
                except ValueError:
                    pass
        if should_request:
            try:
                queued_id = _enqueue_job(
                    session=session,
                    queue=queue,
                    settings=settings,
                    job_type="upstox_token_request",
                    task_path="app.jobs.tasks.run_upstox_token_request_job",
                    payload={"source": "scheduler_auto_renew"},
                )
                emit_operate_event(
                    session,
                    severity="INFO",
                    category="SYSTEM",
                    message="upstox_auto_renew_triggered",
                    details={
                        "date": today.isoformat(),
                        "reason": reason,
                        "threshold_hours": auto_renew_threshold_hours,
                        "queued_job_id": queued_id,
                    },
                    correlation_id=queued_id,
                )
            except Exception as exc:  # noqa: BLE001
                emit_operate_event(
                    session,
                    severity="WARN",
                    category="SYSTEM",
                    message="upstox_auto_renew_failed",
                    details={
                        "date": today.isoformat(),
                        "reason": reason,
                        "error": str(exc),
                    },
                    correlation_id=None,
                )
        merged["operate_last_upstox_auto_renew_date"] = today.isoformat()
        triggered = True

    if auto_run_enabled and now.time() >= run_time:
        if not (isinstance(last_run_date, str) and last_run_date == today.isoformat()):
            queued_jobs: dict[str, str] = {}
            if isinstance(bundle_id, int) and bundle_id > 0 and include_data_updates:
                if provider_updates_enabled and provider_timeframe_allowed:
                    queued_jobs["provider_updates"] = _enqueue_job(
                        session=session,
                        queue=queue,
                        settings=settings,
                        job_type="provider_updates",
                        task_path="app.jobs.tasks.run_provider_updates_job",
                        payload={"bundle_id": bundle_id, "timeframe": timeframe},
                    )
                queued_jobs["data_updates"] = _enqueue_job(
                    session=session,
                    queue=queue,
                    settings=settings,
                    job_type="data_updates",
                    task_path="app.jobs.tasks.run_data_updates_job",
                    payload={"bundle_id": bundle_id, "timeframe": timeframe},
                )
            if isinstance(bundle_id, int) and bundle_id > 0:
                queued_jobs["data_quality"] = _enqueue_job(
                    session=session,
                    queue=queue,
                    settings=settings,
                    job_type="data_quality",
                    task_path="app.jobs.tasks.run_data_quality_job",
                    payload={"bundle_id": bundle_id, "timeframe": timeframe},
                )

            queued_jobs["paper_step"] = _enqueue_job(
                session=session,
                queue=queue,
                settings=settings,
                job_type="paper_step",
                task_path="app.jobs.tasks.run_paper_step_job",
                payload={
                    "regime": regime,
                    "bundle_id": bundle_id,
                    "auto_generate_signals": True,
                    "signals": [],
                    "mark_prices": {},
                    "asof": now.astimezone(ZoneInfo("UTC")).isoformat(),
                },
            )

            queued_jobs["daily_report"] = _enqueue_job(
                session=session,
                queue=queue,
                settings=settings,
                job_type="daily_report",
                task_path="app.jobs.tasks.run_daily_report_job",
                payload={
                    "date": today.isoformat(),
                    "bundle_id": bundle_id,
                    "policy_id": policy_id,
                },
            )

            merged["operate_last_auto_run_date"] = today.isoformat()
            emit_operate_event(
                session,
                severity="INFO",
                category="SYSTEM",
                message="operate_auto_run_triggered",
                details={
                    "date": today.isoformat(),
                    "bundle_id": bundle_id,
                    "timeframe": timeframe,
                    "policy_id": policy_id,
                    "include_data_updates": include_data_updates,
                    "provider_updates_enabled": provider_updates_enabled,
                    "calendar_segment": segment,
                    "session": calendar_get_session(today, segment=segment, settings=settings),
                    "queued_jobs": queued_jobs,
                },
                correlation_id=queued_jobs.get("paper_step"),
            )
            triggered = True

    auto_eval_enabled = bool(
        state_settings.get("operate_auto_eval_enabled", settings.operate_auto_eval_enabled)
    )
    auto_eval_frequency = (
        str(state_settings.get("operate_auto_eval_frequency", settings.operate_auto_eval_frequency))
        .strip()
        .upper()
    )
    auto_eval_day_of_week = (
        _safe_int(
            state_settings.get(
                "operate_auto_eval_day_of_week", settings.operate_auto_eval_day_of_week
            ),
            settings.operate_auto_eval_day_of_week,
        )
        % 7
    )
    auto_eval_time = parse_time_hhmm(
        str(state_settings.get("operate_auto_eval_time_ist", settings.operate_auto_eval_time_ist)),
        default=settings.operate_auto_eval_time_ist,
    )
    auto_eval_last_date = state_settings.get("operate_last_auto_eval_date")
    auto_eval_due = False
    if auto_eval_frequency == "DAILY":
        auto_eval_due = True
    else:
        auto_eval_due = _weekly_eval_matches_day(
            day=today,
            day_of_week=auto_eval_day_of_week,
            segment=segment,
            settings=settings,
        )

    if (
        auto_eval_enabled
        and auto_eval_due
        and now.time() >= auto_eval_time
        and not (isinstance(auto_eval_last_date, str) and auto_eval_last_date == today.isoformat())
    ):
        dedupe_key = f"{today.isoformat()}::AUTO_EVAL"
        queued_eval_id: str | None = None
        if (
            isinstance(bundle_id, int)
            and bundle_id > 0
            and (
                (isinstance(policy_id, int) and policy_id > 0)
                or (isinstance(active_ensemble_id, int) and active_ensemble_id > 0)
            )
        ):
            queued_eval_id = _enqueue_job(
                session=session,
                queue=queue,
                settings=settings,
                job_type="auto_eval",
                task_path="app.jobs.tasks.run_auto_eval_job",
                payload={
                    "bundle_id": bundle_id,
                    "active_policy_id": policy_id,
                    "active_ensemble_id": active_ensemble_id,
                    "timeframe": timeframe,
                    "lookback_trading_days": _safe_int(
                        state_settings.get(
                            "operate_auto_eval_lookback_trading_days",
                            settings.operate_auto_eval_lookback_trading_days,
                        ),
                        settings.operate_auto_eval_lookback_trading_days,
                    ),
                    "min_trades": _safe_int(
                        state_settings.get(
                            "operate_auto_eval_min_trades",
                            settings.operate_auto_eval_min_trades,
                        ),
                        settings.operate_auto_eval_min_trades,
                    ),
                    "asof_date": today.isoformat(),
                    "seed": 7,
                },
            )
        else:
            emit_operate_event(
                session,
                severity="WARN",
                category="POLICY",
                message="operate_auto_eval_skipped_missing_context",
                details={
                    "trading_date": today.isoformat(),
                    "dedupe_key": dedupe_key,
                    "bundle_id": bundle_id,
                    "active_policy_id": policy_id,
                    "active_ensemble_id": active_ensemble_id,
                },
                correlation_id=None,
            )
        merged["operate_last_auto_eval_date"] = today.isoformat()
        if queued_eval_id is not None:
            emit_operate_event(
                session,
                severity="INFO",
                category="SYSTEM",
                message="operate_auto_eval_triggered",
                details={
                    "trading_date": today.isoformat(),
                    "dedupe_key": dedupe_key,
                    "bundle_id": bundle_id,
                    "timeframe": timeframe,
                    "active_policy_id": policy_id,
                    "active_ensemble_id": active_ensemble_id,
                    "frequency": auto_eval_frequency,
                    "calendar_segment": segment,
                    "queued_job_id": queued_eval_id,
                },
                correlation_id=queued_eval_id,
            )
        triggered = True

    if not triggered:
        return False
    state.settings_json = merged
    session.add(state)
    session.commit()
    return True


def scheduler_loop(*, stop_event: threading.Event, queue: Queue, poll_seconds: int = 20) -> None:
    settings = get_settings()
    while not stop_event.is_set():
        try:
            with Session(engine) as session:
                run_auto_operate_once(session=session, queue=queue, settings=settings)
        except RedisError:
            pass
        except Exception:  # noqa: BLE001
            pass
        stop_event.wait(max(5, int(poll_seconds)))
