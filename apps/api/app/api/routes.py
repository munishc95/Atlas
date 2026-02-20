from __future__ import annotations

import csv
import io
import json
import secrets
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError
from rq import Retry
from sqlalchemy import func
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.core.exceptions import APIError
from app.db.models import (
    Backtest,
    PolicySwitchEvent,
    Policy,
    PaperRun,
    ResearchRun,
    Strategy,
    Symbol,
    Trade,
    UpstoxTokenRequestRun,
    WalkForwardFold,
    WalkForwardRun,
)
from app.db.session import engine, get_session
from app.jobs.queue import get_queue
from app.jobs.tasks import (
    run_auto_eval_job,
    run_backtest_job,
    run_provider_updates_job,
    run_data_updates_job,
    run_data_quality_job,
    run_import_job,
    run_evaluation_job,
    run_paper_step_job,
    run_daily_report_job,
    run_monthly_report_job,
    run_operate_run_job,
    run_replay_job,
    run_research_job,
    run_walkforward_job,
)
from app.schemas.api import (
    AutoEvalRunRequest,
    BacktestRunRequest,
    CreatePolicyEnsembleRequest,
    CreatePolicyRequest,
    DataQualityRunRequest,
    DataUpdatesRunRequest,
    ProviderUpdatesRunRequest,
    DailyReportGenerateRequest,
    MonthlyReportGenerateRequest,
    OperateRunRequest,
    PaperSignalsPreviewRequest,
    PaperRunStepRequest,
    PolicyEnsembleMembersRequest,
    PolicyEnsembleRegimeWeightsRequest,
    PolicyEvaluationRunRequest,
    PromoteStrategyRequest,
    ReplayRunRequest,
    ResearchRunRequest,
    RuntimeSettingsRequest,
    UpstoxTokenExchangeRequest,
    UpstoxTokenRequestCreateRequest,
    UpstoxMappingImportRequest,
    WalkForwardRunRequest,
)
from app.services.data_store import DataStore
from app.services.auto_evaluation import (
    get_auto_eval_run,
    list_auto_eval_runs,
    list_policy_switch_events,
)
from app.services.ensembles import (
    create_policy_ensemble,
    get_active_policy_ensemble,
    get_policy_ensemble,
    list_policy_ensemble_members,
    list_policy_ensembles,
    serialize_policy_ensemble,
    set_active_policy_ensemble,
    upsert_policy_ensemble_regime_weights,
    upsert_policy_ensemble_members,
)
from app.services.data_updates import (
    compute_data_coverage,
    get_latest_data_update_run,
    list_data_update_history,
)
from app.services.provider_updates import (
    get_latest_provider_update_run,
    list_provider_update_history,
)
from app.services.provider_mapping import (
    get_upstox_mapping_status,
    import_upstox_mapping_file,
    list_upstox_missing_symbols,
)
from app.services.upstox_auth import (
    build_fake_access_token,
    build_authorization_url,
    delete_provider_credential,
    get_provider_access_token,
    get_provider_credential,
    mark_verified_now,
    save_provider_credential,
    store_oauth_state,
    token_status,
    token_timestamps_from_payload,
    exchange_authorization_code,
    extract_access_token,
    mask_token,
    persist_access_token,
    resolve_client_id,
    resolve_client_secret,
    resolve_redirect_uri,
    validate_oauth_state,
    verify_access_token,
)
from app.services.upstox_token_request import (
    auto_renew_meta,
    check_notifier_rate_limit,
    create_notifier_ping,
    ensure_test_pending_run,
    get_notifier_secret,
    legacy_notifier_endpoint,
    latest_request_run,
    list_notifier_events,
    list_request_runs,
    notifier_health_payload,
    notifier_status_payload,
    notifier_ping_status,
    process_notifier_payload,
    receive_notifier_ping,
    recommended_notifier_endpoint,
    renew_token_run,
    serialize_ping_event,
    request_token_run,
    serialize_request_run,
    serialize_notifier_event,
    sweep_expired_request_runs,
)
from app.services.data_quality import (
    get_latest_data_quality_report,
    list_data_quality_history,
)
from app.services.jobs import create_job, get_job, job_event_stream, list_recent_jobs, update_job
from app.services.jobs import find_job_by_idempotency, hash_payload
from app.services.operate_events import (
    emit_operate_event,
    get_operate_health_summary,
    list_operate_events,
)
from app.services.fast_mode import clamp_job_timeout_seconds, fast_mode_enabled
from app.services.paper import (
    activate_policy_mode,
    get_orders,
    get_paper_state_payload,
    get_positions,
    get_or_create_paper_state,
    preview_policy_signals,
    update_runtime_settings,
)
from app.services.evaluations import (
    get_policy_evaluation,
    get_policy_evaluation_details,
    list_policy_evaluations,
)
from app.services.policy_health import (
    get_policy_health_snapshot,
    latest_policy_health_snapshots,
)
from app.services.replay import get_replay_run, list_replay_runs
from app.services.reports import (
    get_daily_report,
    get_monthly_report,
    list_daily_reports,
    list_monthly_reports,
    render_daily_report_pdf,
    render_monthly_report_pdf,
)
from app.services.research import (
    create_policy_from_research_run,
    list_research_candidates,
    list_research_runs,
)
from app.services.regime import current_regime_payload
from app.strategies.templates import default_template_payload

router = APIRouter(prefix="/api", tags=["atlas"])


def _data(data: Any, meta: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"data": data}
    if meta:
        payload["meta"] = meta
    return payload


def get_store(settings: Settings = Depends(get_settings)) -> DataStore:
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _enqueue_or_inline(
    *,
    session: Session,
    settings: Settings,
    job_type: str,
    task_path: str,
    task_args: list[Any],
    idempotency_key: str | None,
    request_hash: str,
    inline_runner,
) -> dict[str, Any]:
    if idempotency_key:
        existing = find_job_by_idempotency(
            session,
            job_type=job_type,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
        )
        if existing is not None:
            return _data({"job_id": existing.id, "status": existing.status, "deduplicated": True})

    job = create_job(
        session,
        job_type,
        idempotency_key=idempotency_key,
        request_hash=request_hash,
    )
    max_runtime_seconds = int(settings.job_default_timeout_seconds)
    if task_args and isinstance(task_args[0], dict):
        cfg = task_args[0].get("config")
        if isinstance(cfg, dict) and cfg.get("max_runtime_seconds") is not None:
            max_runtime_seconds = max(1, int(cfg["max_runtime_seconds"]))
    max_runtime_seconds = clamp_job_timeout_seconds(
        settings=settings,
        requested=max_runtime_seconds,
    )
    retry = None
    if settings.job_retry_max > 0:
        intervals = [
            settings.job_retry_backoff_seconds * (2**attempt)
            for attempt in range(settings.job_retry_max)
        ]
        retry = Retry(max=settings.job_retry_max, interval=intervals)

    try:
        if settings.jobs_inline or fast_mode_enabled(settings):
            inline_runner(job.id, *task_args, max_runtime_seconds=max_runtime_seconds)
        else:
            queue = get_queue(settings)
            queue.enqueue(
                task_path,
                job.id,
                *task_args,
                max_runtime_seconds=max_runtime_seconds,
                job_timeout=max_runtime_seconds,
                retry=retry,
            )
    except RedisError as exc:
        # Local-first fallback: if queue dispatch fails (e.g. Redis unavailable),
        # run inline so operator flows/tests remain functional.
        try:
            inline_runner(job.id, *task_args, max_runtime_seconds=max_runtime_seconds)
        except Exception as inline_exc:  # noqa: BLE001
            update_job(
                session,
                job.id,
                status="FAILED",
                progress=100,
                result={
                    "error": {
                        "code": "queue_error",
                        "message": (
                            f"Queue dispatch failed: {exc}. Inline fallback failed: {inline_exc}"
                        ),
                    }
                },
            )
    except Exception as exc:  # noqa: BLE001
        update_job(
            session,
            job.id,
            status="FAILED",
            progress=100,
            result={"error": {"code": "job_dispatch_failed", "message": str(exc)}},
        )

    session.expire_all()
    current = get_job(session, job.id)
    status = current.status if current is not None else job.status
    return _data({"job_id": job.id, "status": status})


def _parse_report_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise APIError(
            code="invalid_date",
            message="Date must be in YYYY-MM-DD format.",
        ) from exc


def _parse_report_month(value: str | None) -> str | None:
    if value is None:
        return None
    if len(value) == 7 and value[4] == "-":
        return value
    raise APIError(code="invalid_month", message="Month must be in YYYY-MM format.")


@router.get("/health")
def health() -> dict[str, Any]:
    return _data(
        {
            "status": "ok",
            "service": "atlas-api",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer": (
                "This tool is for research and paper trading. Not financial advice. "
                "Trading involves risk. Past performance does not guarantee future results."
            ),
        }
    )


@router.get("/universe")
def universe(
    session: Session = Depends(get_session),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    return _data({"symbols": store.list_symbols(session)})


@router.get("/universes")
def universes(
    session: Session = Depends(get_session),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    return _data(store.list_bundles(session))


@router.get("/strategies")
def strategy_list(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = session.exec(select(Strategy).order_by(Strategy.created_at.desc())).all()
    return _data([row.model_dump() for row in rows])


@router.get("/strategies/templates")
def strategy_templates() -> dict[str, Any]:
    return _data(default_template_payload())


@router.post("/data/import")
def import_data(
    symbol: str = Form(...),
    timeframe: str = Form("1d"),
    provider: str = Form("csv"),
    instrument_kind: str = Form("EQUITY_CASH"),
    underlying: str | None = Form(default=None),
    lot_size: int | None = Form(default=None),
    tick_size: float = Form(0.05),
    bundle_id: int | None = Form(default=None),
    bundle_name: str | None = Form(default=None),
    bundle_description: str | None = Form(default=None),
    mapping_json: str | None = Form(default=None),
    file: UploadFile = File(...),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    content = file.file.read()
    if not content:
        raise APIError(code="empty_file", message="Uploaded file is empty")

    mapping = json.loads(mapping_json) if mapping_json else None
    payload = {
        "symbol": symbol.upper(),
        "timeframe": timeframe,
        "provider": provider,
        "instrument_kind": instrument_kind,
        "underlying": underlying,
        "lot_size": lot_size,
        "tick_size": tick_size,
        "bundle_id": bundle_id,
        "bundle_name": bundle_name,
        "bundle_description": bundle_description,
        "mapping": mapping,
    }
    request_hash = hash_payload(
        {
            "payload": payload,
            "filename": file.filename,
            "content_sha256": hash_payload(content),
        }
    )

    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="data_import",
        task_path="app.jobs.tasks.run_import_job",
        task_args=[payload, content, file.filename],
        idempotency_key=idempotency_key,
        request_hash=request_hash,
        inline_runner=run_import_job,
    )


@router.get("/data/status")
def data_status(
    session: Session = Depends(get_session),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    return _data(store.data_status(session))


@router.post("/data/updates/run")
def run_data_updates(
    payload: DataUpdatesRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="data_updates",
        task_path="app.jobs.tasks.run_data_updates_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_data_updates_job,
    )


@router.get("/data/updates/latest")
def latest_data_update(
    bundle_id: int = Query(..., ge=1),
    timeframe: str = Query(default="1d"),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_latest_data_update_run(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe,
    )
    if row is None:
        raise APIError(code="not_found", message="No data update run found", status_code=404)
    return _data(row.model_dump())


@router.get("/data/updates/history")
def data_updates_history(
    bundle_id: int | None = Query(default=None, ge=1),
    timeframe: str | None = Query(default=None),
    days: int = Query(default=7, ge=1, le=365),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = list_data_update_history(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe,
        days=days,
    )
    return _data([row.model_dump() for row in rows])


@router.post("/data/provider-updates/run")
def run_provider_updates(
    payload: ProviderUpdatesRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="provider_updates",
        task_path="app.jobs.tasks.run_provider_updates_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_provider_updates_job,
    )


@router.get("/data/provider-updates/latest")
def latest_provider_update(
    bundle_id: int | None = Query(default=None, ge=1),
    timeframe: str | None = Query(default=None),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_latest_provider_update_run(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe,
    )
    if row is None:
        raise APIError(code="not_found", message="No provider update run found", status_code=404)
    return _data(row.model_dump())


@router.get("/data/provider-updates/history")
def provider_updates_history(
    bundle_id: int | None = Query(default=None, ge=1),
    timeframe: str | None = Query(default=None),
    days: int = Query(default=7, ge=1, le=365),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = list_provider_update_history(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe,
        days=days,
    )
    return _data([row.model_dump() for row in rows])


@router.get("/providers/upstox/auth-url")
def upstox_auth_url(
    redirect_uri: str | None = Query(default=None),
    state: str | None = Query(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    if settings.fast_mode_enabled:
        client_id = str(settings.upstox_client_id or settings.upstox_api_key or "ATLAS_FASTMODE")
    else:
        client_id = resolve_client_id(settings)
    resolved_redirect = resolve_redirect_uri(settings=settings, redirect_uri=redirect_uri)
    resolved_state = str(state).strip() if state else secrets.token_urlsafe(16)
    expires_at = store_oauth_state(
        session,
        settings=settings,
        state=resolved_state,
        redirect_uri=resolved_redirect,
        ttl_seconds=settings.upstox_oauth_state_ttl_seconds,
    )
    auth_url = build_authorization_url(
        client_id=client_id,
        redirect_uri=resolved_redirect,
        state=resolved_state,
        base_url=settings.upstox_base_url,
    )
    return _data(
        {
            "auth_url": auth_url,
            "state": resolved_state,
            "redirect_uri": resolved_redirect,
            "state_expires_at": expires_at.isoformat(),
            "client_id_hint": f"{client_id[:6]}...",
        }
    )


@router.post("/providers/upstox/token/exchange")
def upstox_token_exchange(
    payload: UpstoxTokenExchangeRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    resolved_redirect = resolve_redirect_uri(
        settings=settings,
        redirect_uri=payload.redirect_uri,
    )
    validate_oauth_state(
        session,
        settings=settings,
        state=payload.state,
        redirect_uri=resolved_redirect,
    )

    is_fast_stub = (
        settings.fast_mode_enabled
        and str(payload.code).strip() == str(settings.upstox_e2e_fake_code).strip()
    )
    if is_fast_stub:
        fake_token = build_fake_access_token()
        raw = {
            "status": "success",
            "data": {
                "access_token": fake_token,
                "issued_at": datetime.now(timezone.utc).isoformat(),
            },
        }
    else:
        client_id = resolve_client_id(settings)
        client_secret = resolve_client_secret(settings)
        raw = exchange_authorization_code(
            code=payload.code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=resolved_redirect,
            base_url=settings.upstox_base_url,
            timeout_seconds=settings.upstox_timeout_seconds,
        )
    token = extract_access_token(raw)
    issued_at, expires_at = token_timestamps_from_payload(token=token, payload=raw)
    saved = save_provider_credential(
        session,
        settings=settings,
        access_token=token,
        issued_at=issued_at,
        expires_at=expires_at,
    )
    persisted_paths: list[str] = []
    if payload.persist_token and bool(settings.upstox_persist_env_fallback):
        persisted_paths = persist_access_token(access_token=token)
        get_settings.cache_clear()
    verification = verify_access_token(
        access_token=token,
        base_url=settings.upstox_base_url,
        timeout_seconds=settings.upstox_timeout_seconds,
    )
    if bool(verification.get("valid")):
        mark_verified_now(session, settings=settings)
    return _data(
        {
            "token_masked": mask_token(token),
            "connected": True,
            "expires_at": saved.expires_at.isoformat() if saved.expires_at is not None else None,
            "last_verified_at": saved.last_verified_at.isoformat()
            if saved.last_verified_at is not None
            else None,
            "persisted_paths": persisted_paths,
            "verification": verification,
            "token_source": "encrypted_store",
            "note": (
                "Upstox access tokens still rotate periodically. Atlas stores token encrypted locally. "
                "Env persistence is optional and disabled by default."
            ),
        }
    )


@router.get("/providers/upstox/token/status")
def upstox_token_status(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    sweep_expired_request_runs(session, settings=settings, correlation_id=None)
    state = get_or_create_paper_state(session, settings)
    state_settings = dict(state.settings_json or {})
    status_payload = token_status(
        session,
        settings=settings,
        allow_env_fallback=True,
    )
    latest = latest_request_run(session)
    status_payload["auto_renew"] = auto_renew_meta(
        settings=settings,
        state_settings=state_settings,
        expires_at=(status_payload.get("expires_at") if isinstance(status_payload, dict) else None),
    )
    status_payload["token_request_latest"] = (
        serialize_request_run(latest) if latest is not None else None
    )
    status_payload["notifier_health"] = notifier_health_payload(session, settings=settings)
    return _data(status_payload)


@router.post("/providers/upstox/token/request")
def upstox_token_request(
    payload: UpstoxTokenRequestCreateRequest | None = None,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    sweep_expired_request_runs(session, settings=settings, correlation_id=None)
    run, deduped = request_token_run(
        session,
        settings=settings,
        correlation_id=None,
        source=str((payload.source if payload is not None else None) or "manual"),
    )
    secret = get_notifier_secret(session, settings=settings)
    guidance = {
        "notifier_url_in_myapps": (
            "Set your notifier webhook in Upstox My Apps to the recommended notifier URL."
        ),
        "recommended_notifier_endpoint": recommended_notifier_endpoint(
            settings=settings,
            session=session,
            nonce=str(run.correlation_nonce or ""),
            include_nonce_query=False,
        ),
        "legacy_notifier_endpoint": legacy_notifier_endpoint(settings=settings, nonce=None),
        "nonce_hint": str(run.correlation_nonce or ""),
        "secret_hint": ("set" if secret else "missing"),
    }
    return _data(
        {
            "run": serialize_request_run(run),
            "deduplicated": bool(deduped),
            "guidance": guidance,
        }
    )


@router.post("/providers/upstox/token/renew")
def upstox_token_renew(
    payload: UpstoxTokenRequestCreateRequest | None = None,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    sweep_expired_request_runs(session, settings=settings, correlation_id=None)
    source = str((payload.source if payload is not None else None) or "ops_manual_renew")
    run, reused = renew_token_run(
        session,
        settings=settings,
        correlation_id=None,
        source=source,
    )
    status = "reused_pending" if reused else "new_pending"
    instructions = {
        "steps": [
            "Open Upstox My Apps and set notifier URL to the secret-path URL below.",
            "Approve the token request in Upstox.",
            "Return to Atlas and click Verify, or wait for status auto-refresh.",
        ],
        "status": status,
    }
    return _data(
        {
            "run": serialize_request_run(run),
            "reused": bool(reused),
            "approval_instructions": instructions,
            "recommended_notifier_url": recommended_notifier_endpoint(
                settings=settings,
                session=session,
                nonce=str(run.correlation_nonce or ""),
                include_nonce_query=False,
            ),
        }
    )


async def _parse_upstox_notifier_payload(request: Request) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        raw = await request.body()
        if raw:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
            if isinstance(parsed, dict):
                payload = parsed
    except Exception:  # noqa: BLE001
        payload = {}
    return payload


def _request_client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("x-forwarded-for")
    if isinstance(forwarded, str) and forwarded.strip():
        return forwarded.split(",")[0].strip() or None
    if request.client is not None:
        return str(request.client.host or "").strip() or None
    return None


def _notifier_response(result: dict[str, Any]) -> dict[str, Any]:
    return _data(
        {
            "acknowledged": True,
            "matched": bool(result.get("matched")),
            "accepted": bool(result.get("accepted")),
            "reason": result.get("reason"),
            "run_id": result.get("run_id"),
            "event_id": result.get("event_id"),
            "deduplicated": bool(result.get("deduplicated", False)),
        }
    )


@router.post("/providers/upstox/notifier")
async def upstox_notifier_legacy(
    request: Request,
    nonce: str | None = Query(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload = await _parse_upstox_notifier_payload(request)
    check_notifier_rate_limit(
        session,
        ip=_request_client_ip(request),
        source="webhook_legacy",
        correlation_id=None,
    )
    try:
        result = process_notifier_payload(
            session,
            settings=settings,
            payload=payload,
            nonce=nonce,
            headers=dict(request.headers),
            secret_valid=True,
            correlation_id=None,
            source="webhook_legacy",
        )
    except Exception:  # noqa: BLE001
        result = {"matched": False, "accepted": False, "reason": "handler_error"}
    return _notifier_response(result)


@router.get("/providers/upstox/notifier/status")
def upstox_notifier_status(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return _data(notifier_status_payload(session, settings=settings))


@router.post("/providers/upstox/notifier/ping")
def upstox_notifier_ping_create(
    source: str = Query(default="settings"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    row = create_notifier_ping(
        session,
        settings=settings,
        source=str(source or "settings").strip() or "settings",
    )
    payload = serialize_ping_event(row, settings=settings)
    return _data(
        {
            **payload,
            "ping_url": payload.get("ping_url"),
        }
    )


@router.get("/providers/upstox/notifier/ping/{ping_id}")
def upstox_notifier_ping_receive(
    ping_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    result = receive_notifier_ping(session, ping_id=ping_id, settings=settings)
    return _data(result)


@router.get("/providers/upstox/notifier/ping/{ping_id}/status")
def upstox_notifier_ping_get_status(
    ping_id: str,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return _data(notifier_ping_status(session, ping_id=ping_id, settings=settings))


@router.post("/providers/upstox/notifier/test")
def upstox_notifier_test(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    run = ensure_test_pending_run(session, settings=settings, source="notifier_test")
    preserve_existing_credential = not bool(settings.fast_mode_enabled)
    existing_credential = (
        get_provider_credential(session) if preserve_existing_credential else None
    )
    credential_snapshot = None
    if existing_credential is not None:
        credential_snapshot = {
            "access_token_encrypted": existing_credential.access_token_encrypted,
            "user_id": existing_credential.user_id,
            "issued_at": existing_credential.issued_at,
            "expires_at": existing_credential.expires_at,
            "last_verified_at": existing_credential.last_verified_at,
            "metadata_json": dict(existing_credential.metadata_json or {}),
        }
    now_utc = (
        datetime(2026, 1, 1, 3, 30, tzinfo=timezone.utc)
        if settings.fast_mode_enabled
        else datetime.now(timezone.utc)
    )
    payload = {
        "client_id": str(settings.upstox_client_id or settings.upstox_api_key or "ATLAS_FASTMODE"),
        "user_id": "ATLAS_TEST",
        "access_token": build_fake_access_token(issued_at=now_utc),
        "token_type": "Bearer",
        "issued_at": now_utc.isoformat(),
        "expires_at": (now_utc + timedelta(hours=12)).isoformat(),
        "message_type": "access_token",
    }
    result = process_notifier_payload(
        session,
        settings=settings,
        payload=payload,
        nonce=str(run.correlation_nonce or ""),
        headers={"x-atlas-notifier-test": "1"},
        secret_valid=True,
        correlation_id=None,
        source="notifier_test",
    )
    if preserve_existing_credential:
        current_credential = get_provider_credential(session)
        if credential_snapshot is not None and current_credential is not None:
            current_credential.access_token_encrypted = str(
                credential_snapshot["access_token_encrypted"]
            )
            current_credential.user_id = (
                str(credential_snapshot["user_id"])
                if credential_snapshot["user_id"] is not None
                else None
            )
            current_credential.issued_at = credential_snapshot["issued_at"]
            current_credential.expires_at = credential_snapshot["expires_at"]
            current_credential.last_verified_at = credential_snapshot["last_verified_at"]
            current_credential.metadata_json = dict(
                credential_snapshot["metadata_json"] or {}
            )
            session.add(current_credential)
            session.commit()
        elif credential_snapshot is None and current_credential is not None:
            metadata = (
                current_credential.metadata_json
                if isinstance(current_credential.metadata_json, dict)
                else {}
            )
            if str(metadata.get("source") or "").strip() == "notifier_test":
                session.delete(current_credential)
                session.commit()
    health_after = notifier_health_payload(session, settings=settings)
    return _data(
        {
            "created_event_id": result.get("event_id"),
            "result": {
                "matched": bool(result.get("matched")),
                "accepted": bool(result.get("accepted")),
                "reason": result.get("reason"),
                "run_id": result.get("run_id"),
            },
            "webhook_health_after": health_after,
        }
    )


@router.get("/providers/upstox/notifier/events")
def upstox_notifier_events(
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows, total = list_notifier_events(session, limit=limit, offset=offset)
    run_ids = {
        str(row.correlated_request_run_id)
        for row in rows
        if row.correlated_request_run_id is not None
    }
    run_lookup: dict[str, UpstoxTokenRequestRun] = {}
    if run_ids:
        run_rows = session.exec(
            select(UpstoxTokenRequestRun).where(UpstoxTokenRequestRun.id.in_(run_ids))
        ).all()
        run_lookup = {str(row.id): row for row in run_rows}
    return _data(
        [serialize_notifier_event(row, run_lookup=run_lookup) for row in rows],
        meta={"limit": limit, "offset": offset, "total": total},
    )


@router.post("/providers/upstox/notifier/{secret}")
async def upstox_notifier_secure(
    secret: str,
    request: Request,
    nonce: str | None = Query(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload = await _parse_upstox_notifier_payload(request)
    expected_secret = get_notifier_secret(session, settings=settings)
    secret_valid = secrets.compare_digest(str(secret).strip(), str(expected_secret).strip())
    check_notifier_rate_limit(
        session,
        ip=_request_client_ip(request),
        source="webhook_secret",
        correlation_id=None,
    )
    try:
        result = process_notifier_payload(
            session,
            settings=settings,
            payload=payload,
            nonce=nonce,
            headers=dict(request.headers),
            secret_valid=secret_valid,
            correlation_id=None,
            source="webhook_secret",
        )
    except Exception:  # noqa: BLE001
        result = {"matched": False, "accepted": False, "reason": "handler_error"}
    return _notifier_response(result)


@router.get("/providers/upstox/token/requests/latest")
def upstox_token_request_latest(
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    sweep_expired_request_runs(session, settings=settings, correlation_id=None)
    row = latest_request_run(session)
    if row is None:
        raise APIError(code="not_found", message="No Upstox token requests found", status_code=404)
    return _data(serialize_request_run(row))


@router.get("/providers/upstox/token/requests/history")
def upstox_token_request_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    sweep_expired_request_runs(session, settings=settings, correlation_id=None)
    rows, total = list_request_runs(
        session,
        page=page,
        page_size=page_size,
    )
    return _data(
        [serialize_request_run(row) for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": page * page_size < total,
        },
    )


@router.get("/providers/upstox/token/verify")
def upstox_token_verify(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    token = get_provider_access_token(
        session,
        settings=settings,
        allow_env_fallback=True,
    )
    if not token:
        raise APIError(
            code="upstox_token_missing",
            message="Connect Upstox first before verification.",
            status_code=400,
        )
    verification = verify_access_token(
        access_token=str(token),
        base_url=settings.upstox_base_url,
        timeout_seconds=settings.upstox_timeout_seconds,
    )
    if bool(verification.get("valid")):
        row = get_provider_credential(session)
        if row is not None:
            mark_verified_now(session, settings=settings)
            session.refresh(row)
    status_payload = token_status(
        session,
        settings=settings,
        allow_env_fallback=True,
    )
    return _data(
        {
            "token_configured": bool(status_payload.get("connected")),
            "token_masked": status_payload.get("token_masked"),
            "expires_at": status_payload.get("expires_at"),
            "is_expired": status_payload.get("is_expired"),
            "expires_soon": status_payload.get("expires_soon"),
            "last_verified_at": status_payload.get("last_verified_at"),
            "verification": verification,
        }
    )


@router.post("/providers/upstox/disconnect")
def upstox_disconnect(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    deleted = delete_provider_credential(session)
    emit_operate_event(
        session,
        severity="WARN",
        category="SYSTEM",
        message="upstox_disconnected",
        details={"provider_kind": "UPSTOX", "deleted": bool(deleted)},
        commit=True,
    )
    return _data({"disconnected": bool(deleted)})


@router.post("/providers/upstox/mapping/import")
def import_upstox_mapping(
    payload: UpstoxMappingImportRequest,
    bundle_id: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    row = import_upstox_mapping_file(
        session=session,
        settings=settings,
        store=store,
        path=payload.path,
        mode=payload.mode,
        bundle_id=bundle_id,
    )
    status = get_upstox_mapping_status(
        session=session,
        store=store,
        bundle_id=bundle_id,
        timeframe="1d",
    )
    return _data(
        {
            "run": row.model_dump(),
            "status": status,
        }
    )


@router.get("/providers/upstox/mapping/status")
def upstox_mapping_status(
    bundle_id: int | None = Query(default=None, ge=1),
    timeframe: str = Query(default="1d"),
    sample_limit: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    return _data(
        get_upstox_mapping_status(
            session=session,
            store=store,
            bundle_id=bundle_id,
            timeframe=timeframe,
            sample_limit=sample_limit,
        )
    )


@router.get("/providers/upstox/mapping/missing")
def upstox_mapping_missing(
    limit: int = Query(default=50, ge=1, le=500),
    bundle_id: int | None = Query(default=None, ge=1),
    timeframe: str = Query(default="1d"),
    session: Session = Depends(get_session),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    symbols = list_upstox_missing_symbols(
        session=session,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
        limit=limit,
    )
    return _data({"symbols": symbols, "count": len(symbols)})


@router.get("/data/coverage")
def data_coverage(
    bundle_id: int = Query(..., ge=1),
    timeframe: str = Query(default="1d"),
    top_n: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    overrides = dict(state.settings_json or {})
    return _data(
        compute_data_coverage(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe=timeframe,
            overrides=overrides,
            top_n=top_n,
        )
    )


@router.post("/data/quality/run")
def run_data_quality(
    payload: DataQualityRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="data_quality",
        task_path="app.jobs.tasks.run_data_quality_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_data_quality_job,
    )


@router.get("/data/quality/latest")
def latest_data_quality(
    bundle_id: int = Query(..., ge=1),
    timeframe: str = Query(default="1d"),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_latest_data_quality_report(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe,
    )
    if row is None:
        raise APIError(code="not_found", message="No data quality report found", status_code=404)
    return _data(row.model_dump())


@router.get("/data/quality/history")
def data_quality_history(
    bundle_id: int | None = Query(default=None, ge=1),
    timeframe: str | None = Query(default=None),
    days: int = Query(default=7, ge=1, le=365),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = list_data_quality_history(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe,
        days=days,
    )
    return _data([row.model_dump() for row in rows])


@router.post("/backtests/run")
def run_backtest(
    payload: BacktestRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="backtest",
        task_path="app.jobs.tasks.run_backtest_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_backtest_job,
    )


@router.get("/backtests")
def list_backtests(
    template: str | None = None,
    timeframe: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    sort_by: str = Query(default="created_at"),
    sort_dir: str = Query(default="desc"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = session.exec(select(Backtest).order_by(Backtest.created_at.desc())).all()
    filtered: list[Backtest] = []
    for row in rows:
        if timeframe and row.timeframe != timeframe:
            continue
        strategy_template = str(row.config_json.get("strategy_template", ""))
        if template and strategy_template != template:
            continue
        if start_date and str(row.start_date) < start_date:
            continue
        if end_date and str(row.end_date) > end_date:
            continue
        filtered.append(row)

    reverse = sort_dir.lower() != "asc"
    if sort_by == "created_at":
        filtered.sort(key=lambda row: row.created_at, reverse=reverse)
    else:
        filtered.sort(key=lambda row: float(row.metrics_json.get(sort_by, 0.0)), reverse=reverse)

    start = (page - 1) * page_size
    end = start + page_size
    items = filtered[start:end]
    payload = [
        {
            "id": row.id,
            "strategy_template": row.config_json.get("strategy_template"),
            "symbol": row.symbol,
            "timeframe": row.timeframe,
            "start_date": row.start_date.isoformat(),
            "end_date": row.end_date.isoformat(),
            "created_at": row.created_at.isoformat(),
            "metrics": row.metrics_json,
            "status": "SUCCEEDED",
        }
        for row in items
    ]
    return _data(
        payload,
        meta={
            "page": page,
            "page_size": page_size,
            "total": len(filtered),
            "has_next": end < len(filtered),
        },
    )


@router.get("/backtests/compare")
def compare_backtests(
    ids: str = Query(..., description="Comma separated backtest IDs"),
    max_points: int = Query(default=1000, ge=100, le=20_000),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    try:
        backtest_ids = [int(item.strip()) for item in ids.split(",") if item.strip()]
    except ValueError as exc:  # noqa: PERF203
        raise APIError(
            code="invalid_ids", message="Backtest IDs must be comma separated integers"
        ) from exc
    if len(backtest_ids) < 2 or len(backtest_ids) > 3:
        raise APIError(code="invalid_count", message="Select 2 to 3 backtest IDs for comparison")

    rows: list[dict[str, Any]] = []
    for backtest_id in backtest_ids:
        backtest = session.get(Backtest, backtest_id)
        if backtest is None:
            raise APIError(
                code="not_found", message=f"Backtest {backtest_id} not found", status_code=404
            )
        equity = list(backtest.config_json.get("equity_curve", []))
        if len(equity) > max_points:
            step = max(1, len(equity) // max_points)
            sampled = equity[::step]
            if sampled[-1] != equity[-1]:
                sampled.append(equity[-1])
            equity = sampled
        rows.append(
            {
                "id": backtest.id,
                "label": f"{backtest.config_json.get('strategy_template', 'strategy')} #{backtest.id}",
                "metrics": backtest.metrics_json,
                "equity_curve": equity,
            }
        )
    return _data(rows)


@router.get("/backtests/{backtest_id}")
def get_backtest(backtest_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise APIError(code="not_found", message="Backtest not found", status_code=404)
    return _data(backtest.model_dump())


@router.get("/backtests/{backtest_id}/trades")
def get_backtest_trades(
    backtest_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=500),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    total = int(
        session.exec(
            select(func.count()).select_from(Trade).where(Trade.backtest_id == backtest_id)
        ).one()
    )
    statement = (
        select(Trade)
        .where(Trade.backtest_id == backtest_id)
        .order_by(Trade.entry_dt)
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = session.exec(statement).all()
    start = (page - 1) * page_size
    end = start + page_size
    return _data(
        [row.model_dump() for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": end < total,
        },
    )


@router.get("/backtests/{backtest_id}/equity")
def get_backtest_equity(
    backtest_id: int,
    max_points: int | None = Query(default=None, ge=50, le=20_000),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise APIError(code="not_found", message="Backtest not found", status_code=404)
    equity = list(backtest.config_json.get("equity_curve", []))
    if max_points and len(equity) > max_points:
        step = max(1, len(equity) // max_points)
        sampled = equity[::step]
        if sampled[-1] != equity[-1]:
            sampled.append(equity[-1])
        equity = sampled
    return _data(equity)


@router.get("/backtests/{backtest_id}/trades/export.csv")
def export_backtest_trades_csv(backtest_id: int, session: Session = Depends(get_session)):
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise APIError(code="not_found", message="Backtest not found", status_code=404)
    rows = session.exec(
        select(Trade).where(Trade.backtest_id == backtest_id).order_by(Trade.entry_dt)
    ).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id",
            "symbol",
            "entry_dt",
            "exit_dt",
            "qty",
            "entry_px",
            "exit_px",
            "pnl",
            "r_multiple",
            "reason",
        ]
    )
    for row in rows:
        writer.writerow(
            [
                row.id,
                row.symbol,
                row.entry_dt.isoformat(),
                row.exit_dt.isoformat(),
                row.qty,
                row.entry_px,
                row.exit_px,
                row.pnl,
                row.r_multiple,
                row.reason,
            ]
        )
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="backtest_{backtest_id}_trades.csv"'
        },
    )


@router.get("/backtests/{backtest_id}/summary/export.json")
def export_backtest_summary_json(backtest_id: int, session: Session = Depends(get_session)):
    backtest = session.get(Backtest, backtest_id)
    if backtest is None:
        raise APIError(code="not_found", message="Backtest not found", status_code=404)
    payload = {
        "id": backtest.id,
        "symbol": backtest.symbol,
        "timeframe": backtest.timeframe,
        "start_date": backtest.start_date.isoformat(),
        "end_date": backtest.end_date.isoformat(),
        "metrics": backtest.metrics_json,
        "config": backtest.config_json,
        "created_at": backtest.created_at.isoformat(),
    }
    return _data(payload)


@router.post("/walkforward/run")
def run_walkforward(
    payload: WalkForwardRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="walkforward",
        task_path="app.jobs.tasks.run_walkforward_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_walkforward_job,
    )


@router.get("/walkforward/{run_id}")
def get_walkforward(run_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    run = session.get(WalkForwardRun, run_id)
    if run is None:
        raise APIError(code="not_found", message="Walk-forward run not found", status_code=404)
    return _data(run.model_dump())


@router.get("/walkforward/{run_id}/folds")
def get_walkforward_folds(
    run_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    run = session.get(WalkForwardRun, run_id)
    if run is None:
        raise APIError(code="not_found", message="Walk-forward run not found", status_code=404)
    statement = (
        select(WalkForwardFold)
        .where(WalkForwardFold.run_id == run_id)
        .order_by(WalkForwardFold.id.asc())
    )
    rows = session.exec(statement).all()
    start = (page - 1) * page_size
    end = start + page_size
    items = rows[start:end]
    return _data(
        [row.model_dump() for row in items],
        meta={
            "page": page,
            "page_size": page_size,
            "total": len(rows),
            "has_next": end < len(rows),
        },
    )


@router.post("/research/run")
def run_research(
    payload: ResearchRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    force_inline = bool((payload_dict.get("config") or {}).get("force_inline", False))
    effective_settings = (
        settings.model_copy(update={"jobs_inline": True}) if force_inline else settings
    )
    return _enqueue_or_inline(
        session=session,
        settings=effective_settings,
        job_type="research",
        task_path="app.jobs.tasks.run_research_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_research_job,
    )


@router.get("/research/runs")
def get_research_runs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows, total = list_research_runs(session, page=page, page_size=page_size)
    end = page * page_size
    return _data(
        [row.model_dump() for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": end < total,
        },
    )


@router.get("/research/runs/{run_id}")
def get_research_run(run_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    run = session.get(ResearchRun, run_id)
    if run is None:
        raise APIError(code="not_found", message="Research run not found", status_code=404)
    return _data(run.model_dump())


@router.get("/research/runs/{run_id}/candidates")
def get_research_run_candidates(
    run_id: int,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=25, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    run = session.get(ResearchRun, run_id)
    if run is None:
        raise APIError(code="not_found", message="Research run not found", status_code=404)
    rows, total = list_research_candidates(session, run_id=run_id, page=page, page_size=page_size)
    end = page * page_size
    return _data(
        [row.model_dump() for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": end < total,
        },
    )


@router.post("/evaluations/run")
def run_evaluation(
    payload: PolicyEvaluationRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="evaluation",
        task_path="app.jobs.tasks.run_evaluation_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_evaluation_job,
    )


@router.get("/evaluations")
def get_evaluations(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows, total = list_policy_evaluations(session, page=page, page_size=page_size)
    end = page * page_size
    return _data(
        [row.model_dump() for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": end < total,
        },
    )


@router.get("/evaluations/{evaluation_id}")
def get_evaluation_by_id(
    evaluation_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_policy_evaluation(session, evaluation_id)
    return _data(row.model_dump())


@router.get("/evaluations/{evaluation_id}/details")
def get_evaluation_details(
    evaluation_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = get_policy_evaluation_details(session, evaluation_id=evaluation_id)
    return _data([row.model_dump() for row in rows])


@router.post("/policies")
def create_policy(
    payload: CreatePolicyRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    policy = create_policy_from_research_run(
        session,
        run_id=payload.research_run_id,
        name=payload.name,
    )
    return _data(policy.model_dump())


@router.post("/ensembles")
def create_ensemble(
    payload: CreatePolicyEnsembleRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    ensemble = create_policy_ensemble(
        session,
        name=payload.name,
        bundle_id=int(payload.bundle_id),
        is_active=bool(payload.is_active),
    )
    if payload.is_active:
        state = get_or_create_paper_state(session, settings)
        merged = dict(state.settings_json or {})
        merged["paper_mode"] = "policy"
        merged["active_policy_id"] = None
        merged["active_policy_name"] = None
        merged["active_ensemble_id"] = int(ensemble.id or 0)
        merged["active_ensemble_name"] = ensemble.name
        state.settings_json = merged
        session.add(state)
        session.commit()
    return _data(serialize_policy_ensemble(session, ensemble, include_members=True))


@router.get("/ensembles")
def list_ensembles(
    bundle_id: int | None = Query(default=None, ge=1),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = list_policy_ensembles(session, bundle_id=bundle_id)
    start = (page - 1) * page_size
    end = start + page_size
    items = rows[start:end]
    return _data(
        [serialize_policy_ensemble(session, row, include_members=True) for row in items],
        meta={
            "page": page,
            "page_size": page_size,
            "total": len(rows),
            "has_next": end < len(rows),
        },
    )


@router.get("/ensembles/{ensemble_id}")
def get_ensemble(
    ensemble_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_policy_ensemble(session, ensemble_id)
    return _data(serialize_policy_ensemble(session, row, include_members=True))


@router.post("/ensembles/{ensemble_id}/members")
def upsert_ensemble_members(
    ensemble_id: int,
    payload: PolicyEnsembleMembersRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    members = upsert_policy_ensemble_members(
        session,
        ensemble_id=ensemble_id,
        members=[item.model_dump() for item in payload.members],
    )
    row = get_policy_ensemble(session, ensemble_id)
    return _data(
        {
            **serialize_policy_ensemble(session, row, include_members=False),
            "members": members,
        }
    )


@router.put("/ensembles/{ensemble_id}/regime-weights")
def put_ensemble_regime_weights(
    ensemble_id: int,
    payload: PolicyEnsembleRegimeWeightsRequest,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    regime_weights = upsert_policy_ensemble_regime_weights(
        session,
        ensemble_id=ensemble_id,
        payload=payload.root,
    )
    row = get_policy_ensemble(session, ensemble_id)
    return _data(
        {
            **serialize_policy_ensemble(session, row, include_members=False),
            "members": list_policy_ensemble_members(
                session,
                ensemble_id=ensemble_id,
                enabled_only=False,
            ),
            "regime_weights": regime_weights,
        }
    )


@router.post("/ensembles/{ensemble_id}/set-active")
def set_active_ensemble(
    ensemble_id: int,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    row = set_active_policy_ensemble(session, ensemble_id=ensemble_id)
    state = get_or_create_paper_state(session, settings)
    merged = dict(state.settings_json or {})
    merged["paper_mode"] = "policy"
    merged["active_policy_id"] = None
    merged["active_policy_name"] = None
    merged["active_ensemble_id"] = int(row.id or 0)
    merged["active_ensemble_name"] = row.name
    state.settings_json = merged
    session.add(state)
    session.commit()
    return _data(
        {
            "status": "active_ensemble_set",
            "ensemble_id": int(row.id or 0),
            "ensemble_name": row.name,
        }
    )


@router.get("/policies")
def list_policies(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = session.exec(select(Policy).order_by(Policy.created_at.desc())).all()
    start = (page - 1) * page_size
    end = start + page_size
    items = rows[start:end]
    return _data(
        [row.model_dump() for row in items],
        meta={
            "page": page,
            "page_size": page_size,
            "total": len(rows),
            "has_next": end < len(rows),
        },
    )


@router.get("/policies/health")
def list_policy_health(session: Session = Depends(get_session)) -> dict[str, Any]:
    rows = latest_policy_health_snapshots(session)
    return _data([row.model_dump() for row in rows])


@router.get("/policies/{policy_id}/health")
def policy_health(
    policy_id: int,
    window_days: int = Query(default=20, ge=5, le=365),
    refresh: bool = Query(default=True),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    policy = session.get(Policy, policy_id)
    if policy is None:
        raise APIError(code="not_found", message="Policy not found", status_code=404)
    paper_state = get_or_create_paper_state(session, settings)
    state_settings = paper_state.settings_json or {}
    snapshot = get_policy_health_snapshot(
        session,
        settings=settings,
        policy=policy,
        window_days=window_days,
        refresh=refresh,
        overrides=state_settings,
    )
    return _data(snapshot.model_dump())


@router.get("/policies/{policy_id}")
def get_policy(policy_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    policy = session.get(Policy, policy_id)
    if policy is None:
        raise APIError(code="not_found", message="Policy not found", status_code=404)
    return _data(policy.model_dump())


@router.post("/policies/{policy_id}/promote-to-paper")
def promote_policy_to_paper(
    policy_id: int,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    policy = session.get(Policy, policy_id)
    if policy is None:
        raise APIError(code="not_found", message="Policy not found", status_code=404)
    payload = activate_policy_mode(session=session, settings=settings, policy=policy)
    return _data({"policy_id": policy.id, "status": "promoted_to_paper", **payload})


@router.post("/policies/{policy_id}/set-active")
def set_active_policy(
    policy_id: int,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    policy = session.get(Policy, policy_id)
    if policy is None:
        raise APIError(code="not_found", message="Policy not found", status_code=404)
    state = get_or_create_paper_state(session, settings)
    before_settings = dict(state.settings_json or {})
    from_policy_id = before_settings.get("active_policy_id")
    payload = activate_policy_mode(session=session, settings=settings, policy=policy)
    try:
        from_id = int(from_policy_id) if from_policy_id is not None else 0
    except (TypeError, ValueError):
        from_id = 0
    if from_id > 0 and from_id != int(policy.id):
        session.add(
            PolicySwitchEvent(
                from_policy_id=from_id,
                to_policy_id=int(policy.id),
                reason="manual_set_active",
                auto_eval_id=None,
                cooldown_state_json={},
                mode="MANUAL",
            )
        )
        session.commit()
    return _data({"policy_id": policy.id, "status": "active_policy_set", **payload})


@router.post("/strategies/promote")
def promote_strategy(
    payload: PromoteStrategyRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    strategy = Strategy(
        name=payload.strategy_name,
        template=payload.template,
        params_json=payload.params_json,
        enabled=True,
        promoted_at=datetime.now(timezone.utc),
    )
    session.add(strategy)
    session.commit()
    session.refresh(strategy)
    return _data({"strategy_id": strategy.id, "status": "promoted"})


@router.get("/regime/current")
def regime_current(
    symbol: str | None = None,
    store: DataStore = Depends(get_store),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    symbols = [] if symbol else [s.symbol for s in session.exec(select(Symbol)).all()]
    selected = (symbol or (symbols[0] if symbols else "NIFTY500")).upper()
    return _data(current_regime_payload(store=store, symbol=selected))


@router.get("/paper/state")
def paper_state(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return _data(get_paper_state_payload(session, settings))


@router.get("/paper/positions")
def paper_positions(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    get_or_create_paper_state(session, settings)
    return _data([row.model_dump() for row in get_positions(session)])


@router.get("/paper/orders")
def paper_orders(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    get_or_create_paper_state(session, settings)
    return _data([row.model_dump() for row in get_orders(session)])


@router.get("/operate/status")
def operate_status(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    health_summary = get_operate_health_summary(session, settings)
    state_payload = get_paper_state_payload(session, settings)
    state = state_payload["state"]
    state_settings = state.get("settings_json", {})
    active_policy_id = state_settings.get("active_policy_id")
    policy: Policy | None = None
    if isinstance(active_policy_id, int):
        policy = session.get(Policy, active_policy_id)
    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    latest_summary = (
        latest_run.summary_json
        if latest_run is not None and isinstance(latest_run.summary_json, dict)
        else {}
    )
    active_bundle_id = (
        latest_run.bundle_id
        if latest_run is not None and latest_run.bundle_id is not None
        else health_summary.get("active_bundle_id")
    )
    preferred_ensemble_id: int | None = None
    try:
        if state_settings.get("active_ensemble_id") is not None:
            preferred_ensemble_id = int(state_settings.get("active_ensemble_id"))
    except (TypeError, ValueError):
        preferred_ensemble_id = None
    active_ensemble = get_active_policy_ensemble(
        session,
        bundle_id=int(active_bundle_id) if isinstance(active_bundle_id, int) else None,
        preferred_ensemble_id=preferred_ensemble_id,
    )
    active_ensemble_payload = (
        serialize_policy_ensemble(session, active_ensemble, include_members=True)
        if active_ensemble is not None
        else None
    )
    health_short = None
    health_long = None
    if policy is not None:
        short_window = int(
            state_settings.get("health_window_days_short", settings.health_window_days_short)
        )
        long_window = int(
            state_settings.get("health_window_days_long", settings.health_window_days_long)
        )
        health_short = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=policy,
            window_days=short_window,
            refresh=False,
            overrides=state_settings,
        ).model_dump()
        health_long = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=policy,
            window_days=long_window,
            refresh=False,
            overrides=state_settings,
        ).model_dump()
    return _data(
        {
            "mode": health_summary.get("mode"),
            "mode_reason": health_summary.get("mode_reason"),
            "safe_mode_on_fail": health_summary.get("safe_mode_on_fail"),
            "safe_mode_action": health_summary.get("safe_mode_action"),
            "operate_mode": health_summary.get("operate_mode"),
            "calendar_segment": health_summary.get("calendar_segment"),
            "calendar_today_ist": health_summary.get("calendar_today_ist"),
            "calendar_is_trading_day_today": health_summary.get("calendar_is_trading_day_today"),
            "calendar_session_today": health_summary.get("calendar_session_today"),
            "calendar_next_trading_day": health_summary.get("calendar_next_trading_day"),
            "calendar_previous_trading_day": health_summary.get("calendar_previous_trading_day"),
            "auto_run_enabled": health_summary.get("auto_run_enabled"),
            "auto_run_time_ist": health_summary.get("auto_run_time_ist"),
            "auto_run_include_data_updates": health_summary.get("auto_run_include_data_updates"),
            "last_auto_run_date": health_summary.get("last_auto_run_date"),
            "next_scheduled_run_ist": health_summary.get("next_scheduled_run_ist"),
            "auto_eval_enabled": health_summary.get("auto_eval_enabled"),
            "auto_eval_frequency": health_summary.get("auto_eval_frequency"),
            "auto_eval_day_of_week": health_summary.get("auto_eval_day_of_week"),
            "auto_eval_time_ist": health_summary.get("auto_eval_time_ist"),
            "last_auto_eval_date": health_summary.get("last_auto_eval_date"),
            "next_auto_eval_run_ist": health_summary.get("next_auto_eval_run_ist"),
            "active_policy_id": policy.id if policy is not None else None,
            "active_policy_name": policy.name if policy is not None else None,
            "active_ensemble_id": (
                int(active_ensemble.id)
                if active_ensemble is not None and active_ensemble.id is not None
                else None
            ),
            "active_ensemble_name": active_ensemble.name if active_ensemble is not None else None,
            "active_ensemble": active_ensemble_payload,
            "active_bundle_id": active_bundle_id,
            "current_regime": latest_run.regime if latest_run is not None else None,
            "no_trade": latest_summary.get("no_trade", {}),
            "no_trade_triggered": bool(latest_summary.get("no_trade_triggered", False)),
            "no_trade_reasons": latest_summary.get("no_trade_reasons", []),
            "ensemble_weights_source": latest_summary.get("ensemble_weights_source"),
            "ensemble_regime_used": latest_summary.get("ensemble_regime_used"),
            "last_run_step_at": latest_run.asof_ts.isoformat() if latest_run is not None else None,
            "latest_run": latest_run.model_dump() if latest_run is not None else None,
            "latest_data_quality": health_summary.get("latest_data_quality"),
            "latest_data_update": health_summary.get("latest_data_update"),
            "latest_provider_update": health_summary.get("latest_provider_update"),
            "upstox_token_status": health_summary.get("upstox_token_status"),
            "upstox_token_request_latest": health_summary.get("upstox_token_request_latest"),
            "upstox_notifier_health": health_summary.get("upstox_notifier_health"),
            "upstox_auto_renew_enabled": health_summary.get("upstox_auto_renew_enabled"),
            "upstox_auto_renew_time_ist": health_summary.get("upstox_auto_renew_time_ist"),
            "upstox_auto_renew_if_expires_within_hours": health_summary.get(
                "upstox_auto_renew_if_expires_within_hours"
            ),
            "upstox_auto_renew_lead_hours_before_open": health_summary.get(
                "upstox_auto_renew_lead_hours_before_open"
            ),
            "operate_provider_stage_on_token_invalid": health_summary.get(
                "operate_provider_stage_on_token_invalid"
            ),
            "operate_last_upstox_auto_renew_date": health_summary.get(
                "operate_last_upstox_auto_renew_date"
            ),
            "next_upstox_auto_renew_ist": health_summary.get("next_upstox_auto_renew_ist"),
            "upstox_token_expires_within_hours": health_summary.get(
                "upstox_token_expires_within_hours"
            ),
            "provider_stage_status": health_summary.get("provider_stage_status"),
            "recent_event_counts_24h": health_summary.get("recent_event_counts_24h"),
            "fast_mode_enabled": health_summary.get("fast_mode_enabled"),
            "last_job_durations": health_summary.get("last_job_durations"),
            "health_short": health_short,
            "health_long": health_long,
            "paper_state": state,
        }
    )


@router.get("/operate/events")
def operate_events(
    since: str | None = Query(default=None),
    severity: str | None = Query(default=None),
    category: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=500),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError as exc:
            raise APIError(
                code="invalid_since", message="since must be an ISO datetime string"
            ) from exc
    rows = list_operate_events(
        session,
        since=since_dt,
        severity=severity,
        category=category,
        limit=limit,
    )
    return _data([row.model_dump() for row in rows])


@router.get("/operate/health")
def operate_health(
    bundle_id: int | None = Query(default=None, ge=1),
    timeframe: str | None = Query(default=None),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    return _data(
        get_operate_health_summary(
            session,
            settings,
            bundle_id=bundle_id,
            timeframe=timeframe,
        )
    )


@router.post("/operate/run")
def operate_run(
    payload: OperateRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="operate_run",
        task_path="app.jobs.tasks.run_operate_run_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_operate_run_job,
    )


@router.post("/operate/auto-eval/run")
def operate_auto_eval_run(
    payload: AutoEvalRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="auto_eval",
        task_path="app.jobs.tasks.run_auto_eval_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_auto_eval_job,
    )


@router.get("/operate/auto-eval/history")
def operate_auto_eval_history(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    bundle_id: int | None = Query(default=None, ge=1),
    policy_id: int | None = Query(default=None, ge=1),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows, total = list_auto_eval_runs(
        session,
        page=page,
        page_size=page_size,
        bundle_id=bundle_id,
        policy_id=policy_id,
    )
    return _data(
        [row.model_dump() for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": page * page_size < total,
        },
    )


@router.get("/operate/auto-eval/{auto_eval_id}")
def operate_auto_eval_by_id(
    auto_eval_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_auto_eval_run(session, auto_eval_id)
    return _data(row.model_dump())


@router.get("/operate/policy-switches")
def operate_policy_switches(
    limit: int = Query(default=10, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = list_policy_switch_events(session, limit=limit)
    return _data([row.model_dump() for row in rows])


@router.post("/paper/run-step")
def paper_run_step(
    payload: PaperRunStepRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="paper_step",
        task_path="app.jobs.tasks.run_paper_step_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_paper_step_job,
    )


@router.post("/paper/signals/preview")
def paper_signals_preview(
    payload: PaperSignalsPreviewRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    store: DataStore = Depends(get_store),
) -> dict[str, Any]:
    return _data(
        preview_policy_signals(
            session=session,
            settings=settings,
            payload=payload.model_dump(),
            store=store,
        )
    )


@router.post("/replay/run")
def replay_run(
    payload: ReplayRunRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="replay",
        task_path="app.jobs.tasks.run_replay_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_replay_job,
    )


@router.get("/replay/runs")
def replay_runs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows, total = list_replay_runs(session, page=page, page_size=page_size)
    end = page * page_size
    return _data(
        [row.model_dump() for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": end < total,
        },
    )


@router.get("/replay/runs/{replay_id}")
def replay_run_by_id(
    replay_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_replay_run(session, replay_id)
    return _data(row.model_dump())


@router.get("/replay/runs/{replay_id}/export.json")
def replay_run_export_json(
    replay_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_replay_run(session, replay_id)
    return _data(row.summary_json if isinstance(row.summary_json, dict) else {})


@router.post("/reports/daily/generate")
def generate_daily_report_job(
    payload: DailyReportGenerateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="daily_report",
        task_path="app.jobs.tasks.run_daily_report_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_daily_report_job,
    )


@router.get("/reports/daily")
def get_daily_reports(
    date_value: str | None = Query(default=None, alias="date"),
    bundle_id: int | None = Query(default=None),
    policy_id: int | None = Query(default=None),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    report_date = _parse_report_date(date_value)
    rows = list_daily_reports(
        session,
        report_date=report_date,
        bundle_id=bundle_id,
        policy_id=policy_id,
    )
    return _data([row.model_dump() for row in rows])


@router.get("/reports/daily/{report_id}")
def get_daily_report_by_id(
    report_id: int, session: Session = Depends(get_session)
) -> dict[str, Any]:
    row = get_daily_report(session, report_id)
    return _data(row.model_dump())


@router.get("/reports/daily/{report_id}/export.json")
def export_daily_report_json(
    report_id: int, session: Session = Depends(get_session)
) -> dict[str, Any]:
    row = get_daily_report(session, report_id)
    return _data(row.content_json)


@router.get("/reports/daily/{report_id}/export.csv")
def export_daily_report_csv(report_id: int, session: Session = Depends(get_session)):
    row = get_daily_report(session, report_id)
    content = row.content_json if isinstance(row.content_json, dict) else {}
    summary = content.get("summary", {})
    explainability = content.get("explainability", {})
    risk = content.get("risk", {})

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["section", "key", "value"])
    for key, value in summary.items():
        writer.writerow(["summary", key, value])
    for key, value in risk.items():
        writer.writerow(["risk", key, value])
    for key, value in (explainability.get("selected_reason_histogram", {}) or {}).items():
        writer.writerow(["selected_reasons", key, value])
    for key, value in (explainability.get("skipped_reason_histogram", {}) or {}).items():
        writer.writerow(["skipped_reasons", key, value])

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="daily_report_{report_id}.csv"'},
    )


@router.get("/reports/daily/{report_id}/export.pdf")
def export_daily_report_pdf(report_id: int, session: Session = Depends(get_session)):
    row = get_daily_report(session, report_id)
    content = render_daily_report_pdf(session, report=row)
    return StreamingResponse(
        iter([content]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="daily_report_{report_id}.pdf"'},
    )


@router.post("/reports/monthly/generate")
def generate_monthly_report_job(
    payload: MonthlyReportGenerateRequest,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    payload_dict = payload.model_dump()
    return _enqueue_or_inline(
        session=session,
        settings=settings,
        job_type="monthly_report",
        task_path="app.jobs.tasks.run_monthly_report_job",
        task_args=[payload_dict],
        idempotency_key=idempotency_key,
        request_hash=hash_payload(payload_dict),
        inline_runner=run_monthly_report_job,
    )


@router.get("/reports/monthly")
def get_monthly_reports(
    month: str | None = Query(default=None),
    bundle_id: int | None = Query(default=None),
    policy_id: int | None = Query(default=None),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    rows = list_monthly_reports(
        session,
        month=_parse_report_month(month),
        bundle_id=bundle_id,
        policy_id=policy_id,
    )
    return _data([row.model_dump() for row in rows])


@router.get("/reports/monthly/{report_id}")
def get_monthly_report_by_id(
    report_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_monthly_report(session, report_id)
    return _data(row.model_dump())


@router.get("/reports/monthly/{report_id}/export.json")
def export_monthly_report_json(
    report_id: int,
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    row = get_monthly_report(session, report_id)
    return _data(row.content_json)


@router.get("/reports/monthly/{report_id}/export.pdf")
def export_monthly_report_pdf(report_id: int, session: Session = Depends(get_session)):
    row = get_monthly_report(session, report_id)
    content = render_monthly_report_pdf(session, report=row)
    return StreamingResponse(
        iter([content]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="monthly_report_{report_id}.pdf"'},
    )


@router.get("/settings")
def get_settings_payload(
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    defaults = {
        "risk_per_trade": settings.risk_per_trade,
        "max_positions": settings.max_positions,
        "kill_switch_dd": settings.kill_switch_drawdown,
        "cooldown_days": settings.kill_switch_cooldown_days,
        "commission_bps": settings.commission_bps,
        "slippage_base_bps": settings.slippage_base_bps,
        "slippage_vol_factor": settings.slippage_vol_factor,
        "allowed_sides": settings.allowed_sides,
        "paper_short_squareoff_time": settings.paper_short_squareoff_time,
        "cost_model_enabled": settings.cost_model_enabled,
        "cost_mode": settings.cost_mode,
        "brokerage_bps": settings.brokerage_bps,
        "stt_delivery_buy_bps": settings.stt_delivery_buy_bps,
        "stt_delivery_sell_bps": settings.stt_delivery_sell_bps,
        "stt_intraday_buy_bps": settings.stt_intraday_buy_bps,
        "stt_intraday_sell_bps": settings.stt_intraday_sell_bps,
        "exchange_txn_bps": settings.exchange_txn_bps,
        "sebi_bps": settings.sebi_bps,
        "stamp_delivery_buy_bps": settings.stamp_delivery_buy_bps,
        "stamp_intraday_buy_bps": settings.stamp_intraday_buy_bps,
        "gst_rate": settings.gst_rate,
        "futures_brokerage_bps": settings.futures_brokerage_bps,
        "futures_stt_sell_bps": settings.futures_stt_sell_bps,
        "futures_exchange_txn_bps": settings.futures_exchange_txn_bps,
        "futures_stamp_buy_bps": settings.futures_stamp_buy_bps,
        "futures_initial_margin_pct": settings.futures_initial_margin_pct,
        "futures_symbol_mapping_strategy": settings.futures_symbol_mapping_strategy,
        "paper_use_simulator_engine": settings.paper_use_simulator_engine,
        "trading_calendar_segment": settings.trading_calendar_segment,
        "operate_safe_mode_on_fail": settings.operate_safe_mode_on_fail,
        "operate_safe_mode_action": settings.operate_safe_mode_action,
        "operate_mode": settings.operate_mode,
        "data_quality_stale_severity": settings.data_quality_stale_severity,
        "data_quality_stale_severity_override": False,
        "data_quality_max_stale_minutes_1d": settings.data_quality_max_stale_minutes_1d,
        "data_quality_max_stale_minutes_intraday": settings.data_quality_max_stale_minutes_intraday,
        "operate_auto_run_enabled": settings.operate_auto_run_enabled,
        "operate_auto_run_time_ist": settings.operate_auto_run_time_ist,
        "operate_auto_run_include_data_updates": settings.operate_auto_run_include_data_updates,
        "operate_last_auto_run_date": None,
        "operate_auto_eval_enabled": settings.operate_auto_eval_enabled,
        "operate_auto_eval_frequency": settings.operate_auto_eval_frequency,
        "operate_auto_eval_day_of_week": settings.operate_auto_eval_day_of_week,
        "operate_auto_eval_time_ist": settings.operate_auto_eval_time_ist,
        "operate_auto_eval_lookback_trading_days": settings.operate_auto_eval_lookback_trading_days,
        "operate_auto_eval_min_trades": settings.operate_auto_eval_min_trades,
        "operate_auto_eval_cooldown_trading_days": settings.operate_auto_eval_cooldown_trading_days,
        "operate_auto_eval_max_switches_per_30d": settings.operate_auto_eval_max_switches_per_30d,
        "operate_auto_eval_auto_switch": settings.operate_auto_eval_auto_switch,
        "operate_auto_eval_shadow_only_gate": settings.operate_auto_eval_shadow_only_gate,
        "operate_last_auto_eval_date": None,
        "operate_max_stale_minutes_1d": settings.operate_max_stale_minutes_1d,
        "operate_max_stale_minutes_4h_ish": settings.operate_max_stale_minutes_4h_ish,
        "operate_max_gap_bars": settings.operate_max_gap_bars,
        "operate_outlier_zscore": settings.operate_outlier_zscore,
        "operate_cost_ratio_spike_threshold": settings.operate_cost_ratio_spike_threshold,
        "operate_cost_ratio_spike_days": settings.operate_cost_ratio_spike_days,
        "operate_cost_spike_risk_scale": settings.operate_cost_spike_risk_scale,
        "operate_scan_truncated_warn_days": settings.operate_scan_truncated_warn_days,
        "operate_scan_truncated_reduce_to": settings.operate_scan_truncated_reduce_to,
        "data_updates_inbox_enabled": settings.data_updates_inbox_enabled,
        "data_updates_max_files_per_run": settings.data_updates_max_files_per_run,
        "data_updates_provider_enabled": settings.data_updates_provider_enabled,
        "data_updates_provider_kind": settings.data_updates_provider_kind,
        "data_updates_provider_max_symbols_per_run": settings.data_updates_provider_max_symbols_per_run,
        "data_updates_provider_max_calls_per_run": settings.data_updates_provider_max_calls_per_run,
        "data_updates_provider_timeframe_enabled": settings.data_updates_provider_timeframe_enabled,
        "data_updates_provider_timeframes": settings.data_updates_provider_timeframes,
        "data_updates_provider_repair_last_n_trading_days": settings.data_updates_provider_repair_last_n_trading_days,
        "data_updates_provider_backfill_max_days": settings.data_updates_provider_backfill_max_days,
        "data_updates_provider_allow_partial_4h_ish": settings.data_updates_provider_allow_partial_4h_ish,
        "upstox_persist_env_fallback": settings.upstox_persist_env_fallback,
        "upstox_auto_renew_enabled": settings.upstox_auto_renew_enabled,
        "upstox_auto_renew_time_ist": settings.upstox_auto_renew_time_ist,
        "upstox_auto_renew_if_expires_within_hours": settings.upstox_auto_renew_if_expires_within_hours,
        "upstox_auto_renew_lead_hours_before_open": settings.upstox_auto_renew_lead_hours_before_open,
        "upstox_auto_renew_only_when_provider_enabled": settings.upstox_auto_renew_only_when_provider_enabled,
        "operate_provider_stage_on_token_invalid": settings.operate_provider_stage_on_token_invalid,
        "upstox_notifier_pending_no_callback_minutes": settings.upstox_notifier_pending_no_callback_minutes,
        "upstox_notifier_stale_hours": settings.upstox_notifier_stale_hours,
        "operate_last_upstox_auto_renew_date": None,
        "coverage_missing_latest_warn_pct": settings.coverage_missing_latest_warn_pct,
        "coverage_missing_latest_fail_pct": settings.coverage_missing_latest_fail_pct,
        "coverage_inactive_after_missing_days": settings.coverage_inactive_after_missing_days,
        "risk_overlay_enabled": (
            True if str(settings.operate_mode).strip().lower() == "live" else False
        ),
        "risk_overlay_target_vol_annual": settings.risk_overlay_target_vol_annual,
        "risk_overlay_lookback_days": settings.risk_overlay_lookback_days,
        "risk_overlay_min_scale": settings.risk_overlay_min_scale,
        "risk_overlay_max_scale": settings.risk_overlay_max_scale,
        "risk_overlay_max_gross_exposure_pct": settings.risk_overlay_max_gross_exposure_pct,
        "risk_overlay_max_single_name_exposure_pct": settings.risk_overlay_max_single_name_exposure_pct,
        "risk_overlay_max_sector_exposure_pct": settings.risk_overlay_max_sector_exposure_pct,
        "risk_overlay_corr_clamp_enabled": settings.risk_overlay_corr_clamp_enabled,
        "risk_overlay_corr_threshold": settings.risk_overlay_corr_threshold,
        "risk_overlay_corr_reduce_factor": settings.risk_overlay_corr_reduce_factor,
        "no_trade_enabled": settings.no_trade_enabled,
        "no_trade_regimes": settings.no_trade_regimes,
        "no_trade_max_realized_vol_annual": settings.no_trade_max_realized_vol_annual,
        "no_trade_min_breadth_pct": settings.no_trade_min_breadth_pct,
        "no_trade_min_trend_strength": settings.no_trade_min_trend_strength,
        "no_trade_cooldown_trading_days": settings.no_trade_cooldown_trading_days,
        "max_position_value_pct_adv": settings.max_position_value_pct_adv,
        "diversification_corr_threshold": settings.diversification_corr_threshold,
        "autopilot_max_symbols_scan": settings.autopilot_max_symbols_scan,
        "autopilot_max_runtime_seconds": settings.autopilot_max_runtime_seconds,
        "reports_auto_generate_daily": settings.reports_auto_generate_daily,
        "health_window_days_short": settings.health_window_days_short,
        "health_window_days_long": settings.health_window_days_long,
        "drift_maxdd_multiplier": settings.drift_maxdd_multiplier,
        "drift_negative_return_cost_ratio_threshold": settings.drift_negative_return_cost_ratio_threshold,
        "drift_win_rate_drop_pct": settings.drift_win_rate_drop_pct,
        "drift_return_delta_threshold": settings.drift_return_delta_threshold,
        "drift_warning_risk_scale": settings.drift_warning_risk_scale,
        "drift_degraded_risk_scale": settings.drift_degraded_risk_scale,
        "drift_degraded_action": settings.drift_degraded_action,
        "four_hour_bars": settings.four_hour_bars,
        "paper_mode": "strategy",
        "active_policy_id": None,
        "active_ensemble_id": None,
        "active_ensemble_name": None,
    }
    merged = {**defaults, **(state.settings_json or {})}
    return _data(merged)


@router.put("/settings")
def put_settings_payload(
    payload: RuntimeSettingsRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> dict[str, Any]:
    clean = payload.model_dump(exclude_unset=True)
    if not clean:
        raise APIError(code="invalid_payload", message="No settings fields provided")
    return _data(update_runtime_settings(session=session, settings=settings, payload=clean))


@router.get("/jobs")
def jobs_list(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=200),
    limit: int | None = Query(default=None, ge=1, le=200),
    session: Session = Depends(get_session),
) -> dict[str, Any]:
    if limit is not None:
        page = 1
        page_size = limit
    rows, total = list_recent_jobs(session, page=page, page_size=page_size)
    return _data(
        [row.model_dump() for row in rows],
        meta={
            "page": page,
            "page_size": page_size,
            "total": total,
            "has_next": page * page_size < total,
        },
    )


@router.get("/jobs/{job_id}")
def job_status(job_id: str, session: Session = Depends(get_session)) -> dict[str, Any]:
    job = get_job(session, job_id)
    if job is None:
        raise APIError(code="not_found", message="Job not found", status_code=404)
    return _data(job.model_dump())


@router.get("/jobs/{job_id}/stream")
def stream_job(job_id: str, settings: Settings = Depends(get_settings)):
    def session_factory() -> Session:
        return Session(engine)

    poll_seconds = (
        float(settings.fast_mode_job_poll_seconds) if fast_mode_enabled(settings) else 1.0
    )
    return StreamingResponse(
        job_event_stream(
            session_factory=session_factory,
            job_id=job_id,
            poll_seconds=max(0.1, poll_seconds),
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
