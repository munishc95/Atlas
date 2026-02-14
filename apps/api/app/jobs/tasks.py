from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any

from redis.exceptions import RedisError
from sqlalchemy.exc import OperationalError
from sqlmodel import Session

from app.core.config import get_settings
from app.db.session import engine
from app.services.backtests import execute_backtest
from app.services.data_store import DataStore
from app.services.data_updates import run_data_updates
from app.services.data_quality import run_data_quality_report
from app.services.evaluations import execute_policy_evaluation
from app.services.importer import import_ohlcv_bytes
from app.services.jobs import append_job_log, update_job
from app.services.operate_events import emit_operate_event
from app.services.paper import run_paper_step
from app.services.replay import execute_replay_run
from app.services.reports import generate_daily_report, generate_monthly_report
from app.services.research import execute_research_run
from app.services.walkforward import execute_walkforward
from app.db.models import PaperState


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _set_progress(session: Session, job_id: str, progress: int, message: str | None = None) -> None:
    update_job(session, job_id, progress=progress)
    if message:
        append_job_log(session, job_id, message)


def _emit_job_error_event(
    session: Session,
    *,
    job_id: str,
    job_type: str,
    exc: Exception,
) -> None:
    emit_operate_event(
        session,
        severity="ERROR",
        category="SYSTEM",
        message=f"{job_type} job failed.",
        details={"job_id": job_id, "job_type": job_type, "error": str(exc)},
        correlation_id=job_id,
    )


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (OperationalError, RedisError, ConnectionError, OSError, TimeoutError)):
        return True
    text = str(exc).lower()
    return any(token in text for token in ("temporar", "timeout", "connection", "locked"))


def _run_with_timeout(fn, timeout_seconds: int | None):
    if timeout_seconds is None:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeoutError as exc:  # pragma: no cover - deterministic in runtime path
            future.cancel()
            raise TimeoutError(f"job exceeded max_runtime_seconds={timeout_seconds}") from exc


def _execute_with_retry(
    *,
    fn,
    settings,
    session: Session,
    job_id: str,
    job_name: str,
    max_runtime_seconds: int | None,
):
    attempts = max(0, int(settings.job_retry_max))
    for attempt in range(attempts + 1):
        try:
            started = time.monotonic()
            output = _run_with_timeout(fn, timeout_seconds=max_runtime_seconds)
            elapsed = time.monotonic() - started
            append_job_log(session, job_id, f"{job_name} runtime {elapsed:.2f}s")
            return output
        except Exception as exc:  # noqa: BLE001
            if attempt < attempts and _is_transient_error(exc):
                wait_seconds = int(settings.job_retry_backoff_seconds * (2**attempt))
                append_job_log(
                    session,
                    job_id,
                    f"Transient {job_name} error on attempt {attempt + 1}: {exc}. Retrying in {wait_seconds}s.",
                )
                time.sleep(wait_seconds)
                continue
            raise


def run_backtest_job(
    job_id: str, payload: dict[str, Any], max_runtime_seconds: int | None = None
) -> None:
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            _set_progress(session, job_id, 10, "Backtest job started")
            result = _execute_with_retry(
                fn=lambda: execute_backtest(
                    session=session,
                    store=store,
                    settings=settings,
                    payload=payload,
                    job_id=job_id,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="backtest",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Backtest job finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Backtest failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="backtest", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "backtest_failed", "message": str(exc)}},
            )


def run_walkforward_job(
    job_id: str, payload: dict[str, Any], max_runtime_seconds: int | None = None
) -> None:
    settings = get_settings()
    store = _store()

    def progress_cb(progress: int, message: str | None = None) -> None:
        try:
            with Session(engine) as cb_session:
                _set_progress(cb_session, job_id, progress, message)
        except Exception:  # noqa: BLE001
            return

    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            _set_progress(session, job_id, 10, "Walk-forward job started")
            result = _execute_with_retry(
                fn=lambda: execute_walkforward(
                    session=session,
                    store=store,
                    settings=settings,
                    payload=payload,
                    progress_cb=progress_cb,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="walkforward",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Walk-forward job finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Walk-forward failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="walkforward", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "walkforward_failed", "message": str(exc)}},
            )


def run_import_job(
    job_id: str,
    payload: dict[str, Any],
    content: bytes,
    filename: str | None,
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            _set_progress(session, job_id, 15, f"Import started for {payload['symbol']}")
            result = _execute_with_retry(
                fn=lambda: import_ohlcv_bytes(
                    session=session,
                    store=store,
                    raw=content,
                    filename=filename,
                    symbol=str(payload["symbol"]).upper(),
                    timeframe=str(payload.get("timeframe", "1d")),
                    mapping=payload.get("mapping"),
                    provider=str(payload.get("provider", "csv")),
                    bar_windows=settings.four_hour_bars,
                    instrument_kind=str(payload.get("instrument_kind", "EQUITY_CASH")),
                    underlying=payload.get("underlying"),
                    lot_size=payload.get("lot_size"),
                    tick_size=float(payload.get("tick_size", 0.05)),
                    bundle_id=payload.get("bundle_id"),
                    bundle_name=payload.get("bundle_name"),
                    bundle_description=payload.get("bundle_description"),
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="import",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Import job finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Import failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="data_import", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "import_failed", "message": str(exc)}},
            )


def run_data_quality_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=10)
            append_job_log(session, job_id, "Data quality run started")
            result = _execute_with_retry(
                fn=lambda: _data_quality_result(
                    session=session,
                    settings=settings,
                    store=store,
                    payload=payload,
                    job_id=job_id,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="data_quality",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Data quality run finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Data quality run failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="data_quality", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "data_quality_failed", "message": str(exc)}},
            )


def run_data_updates_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=10)
            append_job_log(session, job_id, "Data updates run started")
            result = _execute_with_retry(
                fn=lambda: _data_updates_result(
                    session=session,
                    settings=settings,
                    store=store,
                    payload=payload,
                    job_id=job_id,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="data_updates",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Data updates run finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Data updates run failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="data_updates", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "data_updates_failed", "message": str(exc)}},
            )


def _data_updates_result(
    *,
    session: Session,
    settings,
    store: DataStore,
    payload: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    bundle_id = int(payload.get("bundle_id") or 0)
    if bundle_id <= 0:
        raise ValueError("bundle_id is required for data updates run")
    timeframe = str(payload.get("timeframe") or "1d")
    max_files_per_run = payload.get("max_files_per_run")
    state = session.get(PaperState, 1)
    overrides = dict(state.settings_json or {}) if state is not None else {}
    row = run_data_updates(
        session=session,
        settings=settings,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
        overrides=overrides,
        max_files_per_run=(int(max_files_per_run) if max_files_per_run is not None else None),
        correlation_id=job_id,
    )
    return {
        "id": int(row.id) if row.id is not None else None,
        "bundle_id": row.bundle_id,
        "timeframe": row.timeframe,
        "status": row.status,
        "inbox_path": row.inbox_path,
        "scanned_files": row.scanned_files,
        "processed_files": row.processed_files,
        "skipped_files": row.skipped_files,
        "rows_ingested": row.rows_ingested,
        "symbols_affected_json": row.symbols_affected_json,
        "warnings_json": row.warnings_json,
        "errors_json": row.errors_json,
        "created_at": row.created_at.isoformat(),
        "ended_at": row.ended_at.isoformat() if row.ended_at is not None else None,
    }


def _data_quality_result(
    *,
    session: Session,
    settings,
    store: DataStore,
    payload: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    bundle_id = int(payload.get("bundle_id") or 0)
    if bundle_id <= 0:
        raise ValueError("bundle_id is required for data quality run")
    timeframe = str(payload.get("timeframe") or "1d")
    state = session.get(PaperState, 1)
    overrides = dict(state.settings_json or {}) if state is not None else {}
    report = run_data_quality_report(
        session=session,
        settings=settings,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
        overrides=overrides,
        reference_ts=datetime.now(timezone.utc),
        correlation_id=job_id,
    )
    return {
        "id": int(report.id) if report.id is not None else None,
        "bundle_id": report.bundle_id,
        "timeframe": report.timeframe,
        "status": report.status,
        "issues_json": report.issues_json,
        "last_bar_ts": report.last_bar_ts.isoformat() if report.last_bar_ts is not None else None,
        "coverage_pct": report.coverage_pct,
        "checked_symbols": report.checked_symbols,
        "total_symbols": report.total_symbols,
        "created_at": report.created_at.isoformat(),
    }


def run_paper_step_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=10)
            append_job_log(session, job_id, "Paper step started")
            result = _execute_with_retry(
                fn=lambda: run_paper_step(session=session, settings=settings, payload=payload),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="paper_step",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Paper step finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Paper step failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="paper_step", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "paper_step_failed", "message": str(exc)}},
            )


def run_research_job(
    job_id: str, payload: dict[str, Any], max_runtime_seconds: int | None = None
) -> None:
    settings = get_settings()
    store = _store()

    def progress_cb(progress: int, message: str | None = None) -> None:
        try:
            with Session(engine) as cb_session:
                _set_progress(cb_session, job_id, progress, message)
        except Exception:  # noqa: BLE001
            return

    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            _set_progress(session, job_id, 10, "Auto Research job started")
            result = _execute_with_retry(
                fn=lambda: execute_research_run(
                    session=session,
                    store=store,
                    settings=settings,
                    payload=payload,
                    progress_cb=progress_cb,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="research",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Auto Research job finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Auto Research failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="research", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "research_failed", "message": str(exc)}},
            )


def run_daily_report_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()

    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=10)
            append_job_log(session, job_id, "Daily report generation started")
            result = _execute_with_retry(
                fn=lambda: _daily_report_result(
                    session=session,
                    settings=settings,
                    payload=payload,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="daily_report",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Daily report generation finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Daily report failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="daily_report", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "daily_report_failed", "message": str(exc)}},
            )


def run_monthly_report_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()

    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=10)
            append_job_log(session, job_id, "Monthly report generation started")
            result = _execute_with_retry(
                fn=lambda: _monthly_report_result(
                    session=session,
                    settings=settings,
                    payload=payload,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="monthly_report",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Monthly report generation finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Monthly report failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="monthly_report", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "monthly_report_failed", "message": str(exc)}},
            )


def _daily_report_result(
    *,
    session: Session,
    settings,
    payload: dict[str, Any],
) -> dict[str, Any]:
    row = generate_daily_report(
        session=session,
        settings=settings,
        report_date=(
            datetime.fromisoformat(str(payload["date"])).date() if payload.get("date") else None
        ),
        bundle_id=payload.get("bundle_id"),
        policy_id=payload.get("policy_id"),
        overwrite=True,
    )
    return {
        "id": int(row.id) if row.id is not None else None,
        "date": row.date.isoformat(),
        "bundle_id": row.bundle_id,
        "policy_id": row.policy_id,
        "content_json": row.content_json,
        "created_at": row.created_at.isoformat(),
    }


def _monthly_report_result(
    *,
    session: Session,
    settings,
    payload: dict[str, Any],
) -> dict[str, Any]:
    row = generate_monthly_report(
        session=session,
        settings=settings,
        month=str(payload["month"]) if payload.get("month") else None,
        bundle_id=payload.get("bundle_id"),
        policy_id=payload.get("policy_id"),
        overwrite=True,
    )
    return {
        "id": int(row.id) if row.id is not None else None,
        "month": row.month,
        "bundle_id": row.bundle_id,
        "policy_id": row.policy_id,
        "content_json": row.content_json,
        "created_at": row.created_at.isoformat(),
    }


def run_evaluation_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()

    def progress_cb(progress: int, message: str | None = None) -> None:
        try:
            with Session(engine) as cb_session:
                _set_progress(cb_session, job_id, progress, message)
        except Exception:  # noqa: BLE001
            return

    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            _set_progress(session, job_id, 10, "Policy evaluation started")
            result = _execute_with_retry(
                fn=lambda: execute_policy_evaluation(
                    session=session,
                    store=store,
                    settings=settings,
                    payload=payload,
                    progress_cb=progress_cb,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="evaluation",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Policy evaluation finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Policy evaluation failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="evaluation", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "evaluation_failed", "message": str(exc)}},
            )


def run_replay_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()

    def progress_cb(progress: int, message: str | None = None) -> None:
        try:
            with Session(engine) as cb_session:
                _set_progress(cb_session, job_id, progress, message)
        except Exception:  # noqa: BLE001
            return

    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            _set_progress(session, job_id, 10, "Replay run started")
            result = _execute_with_retry(
                fn=lambda: execute_replay_run(
                    session=session,
                    store=store,
                    settings=settings,
                    payload=payload,
                    progress_cb=progress_cb,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="replay",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Replay run finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Replay run failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="replay", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "replay_failed", "message": str(exc)}},
            )
