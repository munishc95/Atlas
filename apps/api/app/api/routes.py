from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
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
    Policy,
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
    run_research_job,
    run_walkforward_job,
)
from app.schemas.api import (
    BacktestRunRequest,
    CreatePolicyRequest,
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
    update_runtime_settings,
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
    return DataStore(parquet_root=settings.parquet_root, duckdb_path=settings.duckdb_path)


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
        "max_position_value_pct_adv": settings.max_position_value_pct_adv,
        "diversification_corr_threshold": settings.diversification_corr_threshold,
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
