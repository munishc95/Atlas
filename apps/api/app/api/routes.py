from __future__ import annotations

import csv
import io
import json
from datetime import date, datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile
from fastapi.responses import StreamingResponse
from redis.exceptions import RedisError
from rq import Retry
from sqlalchemy import func
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.core.exceptions import APIError
from app.db.models import (
    Backtest,
    DailyReport,
    Policy,
    PolicyHealthSnapshot,
    PaperRun,
    ResearchRun,
    Strategy,
    Symbol,
    Trade,
    WalkForwardFold,
    WalkForwardRun,
)
from app.db.session import engine, get_session
from app.jobs.queue import get_queue
from app.jobs.tasks import (
    run_backtest_job,
    run_import_job,
    run_paper_step_job,
    run_daily_report_job,
    run_research_job,
    run_walkforward_job,
)
from app.schemas.api import (
    BacktestRunRequest,
    CreatePolicyRequest,
    DailyReportGenerateRequest,
    PaperSignalsPreviewRequest,
    PaperRunStepRequest,
    PromoteStrategyRequest,
    ResearchRunRequest,
    RuntimeSettingsRequest,
    WalkForwardRunRequest,
)
from app.services.data_store import DataStore
from app.services.jobs import create_job, get_job, job_event_stream, list_recent_jobs, update_job
from app.services.jobs import find_job_by_idempotency, hash_payload
from app.services.paper import (
    activate_policy_mode,
    get_orders,
    get_paper_state_payload,
    get_positions,
    get_or_create_paper_state,
    preview_policy_signals,
    update_runtime_settings,
)
from app.services.policy_health import (
    get_policy_health_snapshot,
    latest_policy_health_snapshots,
)
from app.services.reports import generate_daily_report, get_daily_report, list_daily_reports
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
    max_runtime_seconds = settings.job_default_timeout_seconds
    if task_args and isinstance(task_args[0], dict):
        cfg = task_args[0].get("config")
        if isinstance(cfg, dict) and cfg.get("max_runtime_seconds") is not None:
            max_runtime_seconds = max(1, int(cfg["max_runtime_seconds"]))
    retry = None
    if settings.job_retry_max > 0:
        intervals = [
            settings.job_retry_backoff_seconds * (2**attempt)
            for attempt in range(settings.job_retry_max)
        ]
        retry = Retry(max=settings.job_retry_max, interval=intervals)

    try:
        if settings.jobs_inline:
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
        update_job(
            session,
            job.id,
            status="FAILED",
            progress=100,
            result={"error": {"code": "queue_error", "message": str(exc)}},
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
    state_payload = get_paper_state_payload(session, settings)
    state = state_payload["state"]
    active_policy_id = state.get("settings_json", {}).get("active_policy_id")
    policy: Policy | None = None
    if isinstance(active_policy_id, int):
        policy = session.get(Policy, active_policy_id)
    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    health_short = None
    health_long = None
    if policy is not None:
        state_settings = state.get("settings_json", {})
        short_window = int(
            state_settings.get(
                "health_window_days_short", settings.health_window_days_short
            )
        )
        long_window = int(
            state_settings.get(
                "health_window_days_long", settings.health_window_days_long
            )
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
            "active_policy_id": policy.id if policy is not None else None,
            "active_policy_name": policy.name if policy is not None else None,
            "active_bundle_id": latest_run.bundle_id if latest_run is not None else None,
            "current_regime": latest_run.regime if latest_run is not None else None,
            "last_run_step_at": latest_run.asof_ts.isoformat() if latest_run is not None else None,
            "latest_run": latest_run.model_dump() if latest_run is not None else None,
            "health_short": health_short,
            "health_long": health_long,
            "paper_state": state,
        }
    )


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
def get_daily_report_by_id(report_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
    row = get_daily_report(session, report_id)
    return _data(row.model_dump())


@router.get("/reports/daily/{report_id}/export.json")
def export_daily_report_json(report_id: int, session: Session = Depends(get_session)) -> dict[str, Any]:
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
def stream_job(job_id: str):
    def session_factory() -> Session:
        return Session(engine)

    return StreamingResponse(
        job_event_stream(session_factory=session_factory, job_id=job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
