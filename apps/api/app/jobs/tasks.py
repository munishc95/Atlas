from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
from typing import Any

from redis.exceptions import RedisError
from sqlalchemy.exc import OperationalError
from sqlmodel import Session
from sqlmodel import select

from app.core.config import Settings, get_settings
from app.db.session import engine
from app.services.backtests import execute_backtest
from app.services.auto_evaluation import execute_auto_evaluation
from app.services.data_store import DataStore
from app.services.fast_mode import prefer_sample_bundle_id
from app.services.data_updates import run_data_updates
from app.services.provider_updates import run_provider_updates
from app.services.data_quality import run_data_quality_report
from app.services.evaluations import execute_policy_evaluation
from app.services.importer import import_ohlcv_bytes
from app.services.jobs import append_job_log, update_job
from app.services.operate_events import emit_operate_event
from app.services.paper import run_paper_step
from app.services.replay import execute_replay_run
from app.services.reports import generate_daily_report, generate_monthly_report
from app.services.research import execute_research_run
from app.services.upstox_auth import token_status as upstox_token_status
from app.services.upstox_token_request import request_token_run, serialize_request_run
from app.services.walkforward import execute_walkforward
from app.db.models import DatasetBundle, PaperRun, PaperState, ProviderUpdateItem


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


def _emit_job_duration_event(
    session: Session,
    *,
    job_id: str,
    job_kind: str,
    duration_seconds: float,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    details = {
        "job_id": job_id,
        "job_kind": str(job_kind),
        "duration_seconds": round(float(duration_seconds), 3),
        "status": str(status).upper(),
    }
    if isinstance(extra, dict):
        details.update(extra)
    emit_operate_event(
        session,
        severity="INFO" if str(status).upper() in {"SUCCEEDED", "DONE"} else "WARN",
        category="SYSTEM",
        message="job_duration_recorded",
        details=details,
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
    started = time.perf_counter()
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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="data_quality",
                duration_seconds=time.perf_counter() - started,
                status="SUCCEEDED",
            )
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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="data_quality",
                duration_seconds=time.perf_counter() - started,
                status="FAILED",
            )


def run_data_updates_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()
    started = time.perf_counter()
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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="data_updates",
                duration_seconds=time.perf_counter() - started,
                status="SUCCEEDED",
            )
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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="data_updates",
                duration_seconds=time.perf_counter() - started,
                status="FAILED",
            )


def run_provider_updates_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()
    started = time.perf_counter()
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=10)
            append_job_log(session, job_id, "Provider updates run started")
            result = _execute_with_retry(
                fn=lambda: _provider_updates_result(
                    session=session,
                    settings=settings,
                    store=store,
                    payload=payload,
                    job_id=job_id,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="provider_updates",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Provider updates run finished")
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="provider_updates",
                duration_seconds=time.perf_counter() - started,
                status="SUCCEEDED",
            )
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Provider updates run failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="provider_updates", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "provider_updates_failed", "message": str(exc)}},
            )
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="provider_updates",
                duration_seconds=time.perf_counter() - started,
                status="FAILED",
            )


def run_upstox_token_request_job(
    job_id: str,
    payload: dict[str, Any] | None = None,
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    started = time.perf_counter()
    safe_payload = payload if isinstance(payload, dict) else {}
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=10)
            append_job_log(session, job_id, "Upstox token request started")
            run, deduped = _execute_with_retry(
                fn=lambda: request_token_run(
                    session,
                    settings=settings,
                    correlation_id=job_id,
                    source=str(safe_payload.get("source") or "scheduler"),
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="upstox_token_request",
                max_runtime_seconds=max_runtime_seconds,
            )
            result_payload = {
                "run": serialize_request_run(run),
                "deduplicated": bool(deduped),
            }
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result_payload)
            append_job_log(session, job_id, "Upstox token request finished")
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="upstox_token_request",
                duration_seconds=time.perf_counter() - started,
                status="SUCCEEDED",
            )
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Upstox token request failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="upstox_token_request", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "upstox_token_request_failed", "message": str(exc)}},
            )
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="upstox_token_request",
                duration_seconds=time.perf_counter() - started,
                status="FAILED",
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


def _provider_updates_result(
    *,
    session: Session,
    settings,
    store: DataStore,
    payload: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    bundle_id = int(payload.get("bundle_id") or 0)
    if bundle_id <= 0:
        raise ValueError("bundle_id is required for provider updates run")
    timeframe = str(payload.get("timeframe") or "1d")
    state = session.get(PaperState, 1)
    overrides = dict(state.settings_json or {}) if state is not None else {}
    if payload.get("provider_mode") is not None:
        overrides["data_updates_provider_mode"] = payload.get("provider_mode")
    if payload.get("provider_priority_order") is not None:
        overrides["data_updates_provider_priority_order"] = payload.get("provider_priority_order")
    row = run_provider_updates(
        session=session,
        settings=settings,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
        overrides=overrides,
        provider_kind=payload.get("provider_kind"),
        max_symbols_per_run=payload.get("max_symbols_per_run"),
        max_calls_per_run=payload.get("max_calls_per_run"),
        start=payload.get("start"),
        end=payload.get("end"),
        correlation_id=job_id,
    )
    bars_updated_total = 0
    if row.id is not None:
        items = session.exec(
            select(ProviderUpdateItem).where(ProviderUpdateItem.run_id == int(row.id))
        ).all()
        bars_updated_total = int(sum(int(item.bars_updated or 0) for item in items))
    return {
        "id": int(row.id) if row.id is not None else None,
        "bundle_id": row.bundle_id,
        "timeframe": row.timeframe,
        "provider_kind": row.provider_kind,
        "provider_mode": row.provider_mode,
        "provider_priority_json": row.provider_priority_json,
        "status": row.status,
        "symbols_attempted": row.symbols_attempted,
        "symbols_succeeded": row.symbols_succeeded,
        "symbols_failed": row.symbols_failed,
        "bars_added": row.bars_added,
        "bars_updated": bars_updated_total,
        "repaired_days_used": row.repaired_days_used,
        "missing_days_detected": row.missing_days_detected,
        "backfill_truncated": bool(row.backfill_truncated),
        "api_calls": row.api_calls,
        "coverage_before_pct": float(row.coverage_before_pct or 0.0),
        "coverage_after_pct": float(row.coverage_after_pct or 0.0),
        "by_provider_count_json": row.by_provider_count_json,
        "confidence_distribution_json": row.confidence_distribution_json,
        "continuity_met": bool(row.continuity_met),
        "duration_seconds": row.duration_seconds,
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
        "coverage_by_source_json": report.coverage_by_source_json,
        "low_confidence_days_count": int(report.low_confidence_days_count or 0),
        "low_confidence_symbols_count": int(report.low_confidence_symbols_count or 0),
        "created_at": report.created_at.isoformat(),
    }


def run_paper_step_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    started = time.perf_counter()
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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="paper_step",
                duration_seconds=time.perf_counter() - started,
                status="SUCCEEDED",
            )
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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="paper_step",
                duration_seconds=time.perf_counter() - started,
                status="FAILED",
            )


def _parse_iso_date(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    return None


def _resolve_operate_context(
    *,
    session: Session,
    payload: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    state = session.get(PaperState, 1)
    state_settings = dict(state.settings_json or {}) if state is not None else {}

    bundle_id: int | None = None
    raw_bundle = payload.get("bundle_id")
    if isinstance(raw_bundle, int) and raw_bundle > 0:
        bundle_id = int(raw_bundle)
    elif settings.fast_mode_enabled:
        bundle_id = prefer_sample_bundle_id(session, settings=settings)
    elif latest_run is not None and latest_run.bundle_id is not None:
        bundle_id = int(latest_run.bundle_id)
    else:
        latest_bundle = session.exec(
            select(DatasetBundle).order_by(DatasetBundle.created_at.desc())
        ).first()
        if latest_bundle is not None and latest_bundle.id is not None:
            bundle_id = int(latest_bundle.id)

    timeframe = str(payload.get("timeframe") or "").strip()
    if not timeframe and latest_run is not None:
        summary = latest_run.summary_json if isinstance(latest_run.summary_json, dict) else {}
        tfs = summary.get("timeframes", [])
        if isinstance(tfs, list) and tfs:
            timeframe = str(tfs[0] or "").strip()
    if not timeframe:
        timeframe = "1d"

    regime = str(payload.get("regime") or "").strip()
    if not regime:
        regime = str(latest_run.regime) if latest_run is not None else "TREND_UP"

    policy_id: int | None = None
    raw_policy = payload.get("policy_id")
    if isinstance(raw_policy, int) and raw_policy > 0:
        policy_id = int(raw_policy)
    else:
        try:
            active = state_settings.get("active_policy_id")
            if active is not None:
                policy_id = int(active)
        except (TypeError, ValueError):
            policy_id = None

    include_updates_raw = payload.get("include_data_updates")
    if isinstance(include_updates_raw, bool):
        include_updates = include_updates_raw
    else:
        include_updates = bool(
            state_settings.get(
                "operate_auto_run_include_data_updates",
                settings.operate_auto_run_include_data_updates,
            )
        )

    asof_dt = _parse_iso_date(payload.get("asof")) or datetime.now(timezone.utc)
    return {
        "bundle_id": bundle_id,
        "timeframe": timeframe,
        "regime": regime,
        "policy_id": policy_id,
        "state_settings": state_settings,
        "include_data_updates": include_updates,
        "asof_dt": asof_dt,
    }


def _operate_run_result(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    payload: dict[str, Any],
    job_id: str,
) -> dict[str, Any]:
    context = _resolve_operate_context(session=session, payload=payload, settings=settings)
    bundle_id = context["bundle_id"]
    timeframe = context["timeframe"]
    regime = context["regime"]
    policy_id = context["policy_id"]
    include_data_updates = context["include_data_updates"]
    asof_dt: datetime = context["asof_dt"]
    state_settings: dict[str, Any] = context["state_settings"]
    provider_enabled = bool(
        state_settings.get("data_updates_provider_enabled", settings.data_updates_provider_enabled)
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
    provider_stage_enabled = include_data_updates and (
        (str(timeframe).strip().lower() in provider_timeframes) if provider_enabled else False
    )
    step_order = ["data_updates", "data_quality", "paper_step", "daily_report"]
    if provider_stage_enabled:
        step_order = ["provider_updates", *step_order]

    summary: dict[str, Any] = {
        "bundle_id": bundle_id,
        "timeframe": timeframe,
        "policy_id": policy_id,
        "regime": regime,
        "step_order": step_order,
        "provider_stage_status": ("PENDING" if provider_stage_enabled else "NOT_ENABLED"),
        "steps": [],
    }

    update_job(session, job_id, progress=10)
    provider_payload: dict[str, Any] | None = None
    if provider_stage_enabled:
        step_started = time.perf_counter()
        provider_kind = str(
            payload.get("provider_kind")
            or state_settings.get("data_updates_provider_kind", settings.data_updates_provider_kind)
            or "UPSTOX"
        ).strip().upper()
        provider_invalid_mode = str(
            state_settings.get(
                "operate_provider_stage_on_token_invalid",
                settings.operate_provider_stage_on_token_invalid,
            )
        ).strip().upper() or "SKIP"
        token_invalid_reason: str | None = None
        token_invalid_context: dict[str, Any] | None = None
        if provider_kind == "UPSTOX":
            token_meta = upstox_token_status(
                session,
                settings=settings,
                allow_env_fallback=True,
            )
            if not bool(token_meta.get("connected")):
                token_invalid_reason = "provider_token_missing"
            elif bool(token_meta.get("is_expired")):
                token_invalid_reason = "provider_token_expired"
            if token_invalid_reason is not None:
                token_invalid_context = {
                    "connected": bool(token_meta.get("connected")),
                    "is_expired": bool(token_meta.get("is_expired")),
                    "expires_at": token_meta.get("expires_at"),
                }
        if isinstance(bundle_id, int) and bundle_id > 0:
            if token_invalid_reason is not None and provider_invalid_mode != "FAIL":
                append_job_log(
                    session,
                    job_id,
                    f"Operate run: provider updates skipped ({token_invalid_reason})",
                )
                provider_payload = {
                    "status": "SKIPPED",
                    "reason": "provider_token_invalid",
                    "token_reason": token_invalid_reason,
                    "provider_kind": provider_kind,
                    "provider_stage_status": "SKIPPED_TOKEN_INVALID",
                    "token_context": token_invalid_context or {},
                }
                emit_operate_event(
                    session,
                    severity="WARN",
                    category="SYSTEM",
                    message="operate_provider_stage_skipped_token_invalid",
                    details={
                        "job_id": job_id,
                        "bundle_id": bundle_id,
                        "timeframe": timeframe,
                        "provider_kind": provider_kind,
                        "reason": token_invalid_reason,
                        "mode": provider_invalid_mode,
                    },
                    correlation_id=job_id,
                )
            elif token_invalid_reason is not None and provider_invalid_mode == "FAIL":
                append_job_log(
                    session,
                    job_id,
                    f"Operate run: provider updates failing due to invalid token ({token_invalid_reason})",
                )
                emit_operate_event(
                    session,
                    severity="WARN",
                    category="SYSTEM",
                    message="operate_provider_stage_failed_token_invalid",
                    details={
                        "job_id": job_id,
                        "bundle_id": bundle_id,
                        "timeframe": timeframe,
                        "provider_kind": provider_kind,
                        "reason": token_invalid_reason,
                        "mode": provider_invalid_mode,
                    },
                    correlation_id=job_id,
                )
                raise RuntimeError(
                    f"provider stage failed due to invalid token ({token_invalid_reason})"
                )
            else:
                append_job_log(session, job_id, "Operate run: provider updates started")
                provider_row = run_provider_updates(
                    session=session,
                    settings=settings,
                    store=store,
                    bundle_id=bundle_id,
                    timeframe=timeframe,
                    overrides=state_settings,
                    provider_kind=payload.get("provider_kind"),
                    max_symbols_per_run=payload.get("provider_max_symbols_per_run"),
                    max_calls_per_run=payload.get("provider_max_calls_per_run"),
                    start=payload.get("provider_start"),
                    end=payload.get("provider_end"),
                    correlation_id=job_id,
                )
                provider_items = (
                    session.exec(
                        select(ProviderUpdateItem).where(
                            ProviderUpdateItem.run_id == int(provider_row.id)
                        )
                    ).all()
                    if provider_row.id is not None
                    else []
                )
                bars_updated_total = int(sum(int(item.bars_updated or 0) for item in provider_items))
                provider_payload = {
                    "status": str(provider_row.status),
                    "run_id": int(provider_row.id) if provider_row.id is not None else None,
                    "provider_kind": provider_row.provider_kind,
                    "symbols_attempted": int(provider_row.symbols_attempted),
                    "symbols_succeeded": int(provider_row.symbols_succeeded),
                    "symbols_failed": int(provider_row.symbols_failed),
                    "bars_added": int(provider_row.bars_added),
                    "bars_updated": bars_updated_total,
                    "repaired_days_used": int(provider_row.repaired_days_used),
                    "missing_days_detected": int(provider_row.missing_days_detected),
                    "backfill_truncated": bool(provider_row.backfill_truncated),
                    "api_calls": int(provider_row.api_calls),
                    "provider_stage_status": (
                        "SUCCEEDED"
                        if str(provider_row.status).upper() in {"SUCCEEDED", "DONE"}
                        else str(provider_row.status).upper()
                    ),
                }
        else:
            provider_payload = {
                "status": "SKIPPED",
                "reason": "no_bundle",
                "provider_stage_status": "SKIPPED_NO_BUNDLE",
            }
        provider_payload["duration_seconds"] = round(time.perf_counter() - step_started, 3)
        if "provider_stage_status" not in provider_payload:
            provider_payload["provider_stage_status"] = str(provider_payload.get("status", "UNKNOWN")).upper()
        summary["provider_stage_status"] = provider_payload["provider_stage_status"]
        summary["provider_updates"] = provider_payload
        summary["steps"].append({"name": "provider_updates", **provider_payload})
        emit_operate_event(
            session,
            severity="INFO",
            category="SYSTEM",
            message="job_duration_recorded",
            details={
                "job_id": job_id,
                "job_kind": "provider_updates",
                "duration_seconds": provider_payload["duration_seconds"],
                "status": provider_payload.get("status"),
                "mode": "operate_run",
            },
            correlation_id=job_id,
        )

    update_job(session, job_id, progress=25 if provider_stage_enabled else 15)
    step_started = time.perf_counter()
    if include_data_updates and isinstance(bundle_id, int) and bundle_id > 0:
        append_job_log(session, job_id, "Operate run: data updates started")
        update_row = run_data_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe=timeframe,
            overrides=state_settings,
            max_files_per_run=(
                int(payload["max_files_per_run"])
                if payload.get("max_files_per_run") is not None
                else None
            ),
            correlation_id=job_id,
        )
        step_payload = {
            "status": str(update_row.status),
            "run_id": int(update_row.id) if update_row.id is not None else None,
            "rows_ingested": int(update_row.rows_ingested),
            "processed_files": int(update_row.processed_files),
            "skipped_files": int(update_row.skipped_files),
        }
        summary["data_updates"] = step_payload
    else:
        step_payload = {"status": "SKIPPED", "reason": "disabled_or_no_bundle"}
        summary["data_updates"] = step_payload
    step_payload["duration_seconds"] = round(time.perf_counter() - step_started, 3)
    summary["steps"].append({"name": "data_updates", **step_payload})
    emit_operate_event(
        session,
        severity="INFO",
        category="SYSTEM",
        message="job_duration_recorded",
        details={
            "job_id": job_id,
            "job_kind": "data_updates",
            "duration_seconds": step_payload["duration_seconds"],
            "status": step_payload.get("status"),
            "mode": "operate_run",
        },
        correlation_id=job_id,
    )

    update_job(session, job_id, progress=45 if provider_stage_enabled else 40)
    step_started = time.perf_counter()
    if isinstance(bundle_id, int) and bundle_id > 0:
        append_job_log(session, job_id, "Operate run: data quality started")
        quality = run_data_quality_report(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe=timeframe,
            overrides=state_settings,
            reference_ts=asof_dt,
            correlation_id=job_id,
        )
        quality_payload = {
            "status": str(quality.status),
            "report_id": int(quality.id) if quality.id is not None else None,
            "coverage_pct": float(quality.coverage_pct),
            "issues_count": len(quality.issues_json or []),
        }
    else:
        quality_payload = {"status": "SKIPPED", "reason": "no_bundle"}
    summary["data_quality"] = quality_payload
    quality_payload["duration_seconds"] = round(time.perf_counter() - step_started, 3)
    summary["steps"].append({"name": "data_quality", **quality_payload})
    emit_operate_event(
        session,
        severity="INFO",
        category="SYSTEM",
        message="job_duration_recorded",
        details={
            "job_id": job_id,
            "job_kind": "data_quality",
            "duration_seconds": quality_payload["duration_seconds"],
            "status": quality_payload.get("status"),
            "mode": "operate_run",
        },
        correlation_id=job_id,
    )

    update_job(session, job_id, progress=70 if provider_stage_enabled else 65)
    step_started = time.perf_counter()
    append_job_log(session, job_id, "Operate run: paper step started")
    paper_payload = {
        "regime": regime,
        "bundle_id": bundle_id,
        "auto_generate_signals": True,
        "signals": [],
        "mark_prices": {},
        "timeframes": [timeframe],
        "asof": asof_dt.isoformat(),
    }
    if isinstance(policy_id, int) and policy_id > 0:
        paper_payload["policy_id"] = int(policy_id)
    paper_result = run_paper_step(
        session=session,
        settings=settings,
        payload=paper_payload,
        store=store,
    )
    paper_summary = {
        "status": str(paper_result.get("status", "ok")),
        "paper_run_id": paper_result.get("paper_run_id"),
        "mode": (
            "SHADOW"
            if str(paper_result.get("execution_mode", "")).upper() == "SHADOW"
            else ("SAFE" if bool((paper_result.get("safe_mode") or {}).get("active")) else "NORMAL")
        ),
        "selected_signals_count": int(paper_result.get("selected_signals_count", 0)),
        "generated_signals_count": int(paper_result.get("generated_signals_count", 0)),
        "safe_mode": paper_result.get("safe_mode", {}),
        "confidence_gate": paper_result.get("confidence_gate", {}),
        "scan_truncated": bool(paper_result.get("scan_truncated", False)),
        "risk_overlay": paper_result.get("risk_overlay", {}),
    }
    summary["paper"] = paper_summary
    paper_summary["duration_seconds"] = round(time.perf_counter() - step_started, 3)
    summary["steps"].append({"name": "paper_step", **paper_summary})
    emit_operate_event(
        session,
        severity="INFO",
        category="SYSTEM",
        message="job_duration_recorded",
        details={
            "job_id": job_id,
            "job_kind": "paper_step",
            "duration_seconds": paper_summary["duration_seconds"],
            "status": paper_summary.get("status"),
            "mode": "operate_run",
        },
        correlation_id=job_id,
    )

    update_job(session, job_id, progress=88 if provider_stage_enabled else 85)
    step_started = time.perf_counter()
    append_job_log(session, job_id, "Operate run: daily report generation started")
    report_date = payload.get("date")
    if isinstance(report_date, str) and report_date.strip():
        try:
            report_day = datetime.fromisoformat(report_date).date()
        except ValueError:
            report_day = asof_dt.date()
    else:
        report_day = asof_dt.date()
    report_row = generate_daily_report(
        session=session,
        settings=settings,
        report_date=report_day,
        bundle_id=bundle_id if isinstance(bundle_id, int) and bundle_id > 0 else None,
        policy_id=(int(policy_id) if isinstance(policy_id, int) and policy_id > 0 else None),
        overwrite=True,
    )
    report_payload = {
        "status": "SUCCEEDED",
        "id": int(report_row.id) if report_row.id is not None else None,
        "date": report_row.date.isoformat(),
    }
    summary["daily_report"] = report_payload
    report_payload["duration_seconds"] = round(time.perf_counter() - step_started, 3)
    summary["steps"].append({"name": "daily_report", **report_payload})
    emit_operate_event(
        session,
        severity="INFO",
        category="SYSTEM",
        message="job_duration_recorded",
        details={
            "job_id": job_id,
            "job_kind": "daily_report",
            "duration_seconds": report_payload["duration_seconds"],
            "status": report_payload.get("status"),
            "mode": "operate_run",
        },
        correlation_id=job_id,
    )

    summary["mode"] = paper_summary["mode"]
    summary["quality_status"] = quality_payload.get("status")
    summary["update_status"] = step_payload.get("status")
    summary["durations_seconds"] = {
        "provider_updates": float(provider_payload.get("duration_seconds", 0.0))
        if isinstance(provider_payload, dict)
        else 0.0,
        "data_updates": float(step_payload.get("duration_seconds", 0.0)),
        "data_quality": float(quality_payload.get("duration_seconds", 0.0)),
        "paper_step": float(paper_summary.get("duration_seconds", 0.0)),
        "daily_report": float(report_payload.get("duration_seconds", 0.0)),
    }
    return {
        "status": "ok",
        "summary": summary,
    }


def run_operate_run_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            append_job_log(session, job_id, "Operate run started")
            result = _execute_with_retry(
                fn=lambda: _operate_run_result(
                    session=session,
                    settings=settings,
                    store=store,
                    payload=payload,
                    job_id=job_id,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="operate_run",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Operate run finished")
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Operate run failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="operate_run", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "operate_run_failed", "message": str(exc)}},
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
    started = time.perf_counter()

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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="daily_report",
                duration_seconds=time.perf_counter() - started,
                status="SUCCEEDED",
            )
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
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="daily_report",
                duration_seconds=time.perf_counter() - started,
                status="FAILED",
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


def run_auto_eval_job(
    job_id: str,
    payload: dict[str, Any],
    max_runtime_seconds: int | None = None,
) -> None:
    settings = get_settings()
    store = _store()
    started = time.perf_counter()

    def progress_cb(progress: int, message: str | None = None) -> None:
        try:
            with Session(engine) as cb_session:
                _set_progress(cb_session, job_id, progress, message)
        except Exception:  # noqa: BLE001
            return

    with Session(engine) as session:
        try:
            update_job(session, job_id, status="RUNNING", progress=5)
            _set_progress(session, job_id, 10, "Auto evaluation started")
            result = _execute_with_retry(
                fn=lambda: execute_auto_evaluation(
                    session=session,
                    store=store,
                    settings=settings,
                    payload=payload,
                    progress_cb=progress_cb,
                ),
                settings=settings,
                session=session,
                job_id=job_id,
                job_name="auto_eval",
                max_runtime_seconds=max_runtime_seconds,
            )
            update_job(session, job_id, status="SUCCEEDED", progress=100, result=result)
            append_job_log(session, job_id, "Auto evaluation finished")
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="auto_eval",
                duration_seconds=time.perf_counter() - started,
                status="SUCCEEDED",
            )
        except Exception as exc:  # noqa: BLE001
            append_job_log(session, job_id, f"Auto evaluation failed: {exc}")
            _emit_job_error_event(session, job_id=job_id, job_type="auto_eval", exc=exc)
            update_job(
                session,
                job_id,
                status="FAILED",
                progress=100,
                result={"error": {"code": "auto_eval_failed", "message": str(exc)}},
            )
            _emit_job_duration_event(
                session,
                job_id=job_id,
                job_kind="auto_eval",
                duration_seconds=time.perf_counter() - started,
                status="FAILED",
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
