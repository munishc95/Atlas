from __future__ import annotations

from datetime import date, datetime
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
from app.services.trading_calendar import (
    compute_next_scheduled_run_ist as calendar_next_scheduled_run_ist,
    get_session as calendar_get_session,
    is_trading_day,
    parse_time_hhmm,
)

IST_ZONE = ZoneInfo("Asia/Kolkata")


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


def _resolve_scheduler_context(session: Session) -> dict[str, Any]:
    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    state = session.get(PaperState, 1)
    state_settings = dict(state.settings_json or {}) if state is not None else {}

    bundle_id: int | None = None
    timeframe = "1d"
    regime = "TREND_UP"
    policy_id: int | None = None

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

    return {
        "bundle_id": bundle_id,
        "timeframe": timeframe,
        "regime": regime,
        "policy_id": policy_id,
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


def run_auto_operate_once(*, session: Session, queue: Queue, settings: Settings, now_ist: datetime | None = None) -> bool:
    context = _resolve_scheduler_context(session)
    state: PaperState | None = context["state"]
    if state is None:
        return False

    state_settings = context["state_settings"]
    auto_run_enabled = bool(state_settings.get("operate_auto_run_enabled", settings.operate_auto_run_enabled))
    if not auto_run_enabled:
        return False

    run_time = parse_time_hhmm(
        str(state_settings.get("operate_auto_run_time_ist", settings.operate_auto_run_time_ist)),
        default=settings.operate_auto_run_time_ist,
    )
    segment = str(state_settings.get("trading_calendar_segment", settings.trading_calendar_segment))
    now = now_ist or datetime.now(IST_ZONE)
    today = now.date()
    if not is_trading_day(today, segment=segment, settings=settings) or now.time() < run_time:
        return False

    last_run_date = state_settings.get("operate_last_auto_run_date")
    if isinstance(last_run_date, str) and last_run_date == today.isoformat():
        return False

    bundle_id = context["bundle_id"]
    timeframe = str(context["timeframe"] or "1d")
    regime = str(context["regime"] or "TREND_UP")
    policy_id = context["policy_id"]

    queued_jobs: dict[str, str] = {}
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

    merged = dict(state_settings)
    merged["operate_last_auto_run_date"] = today.isoformat()
    state.settings_json = merged
    session.add(state)
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
            "calendar_segment": segment,
            "session": calendar_get_session(today, segment=segment, settings=settings),
            "queued_jobs": queued_jobs,
        },
        correlation_id=queued_jobs.get("paper_step"),
    )
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
