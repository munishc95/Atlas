from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import Policy, ReplayRun
from app.services.data_store import DataStore
from app.services.policy_simulation import simulate_policy_on_bundle


ProgressCallback = Callable[[int, str], None]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: str | None, *, field_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise APIError(
            code="invalid_date",
            message=f"{field_name} must be YYYY-MM-DD.",
        ) from exc


def _utc_datetime(value: date, *, end: bool = False) -> datetime:
    if end:
        return datetime.combine(value, time.max, tzinfo=timezone.utc)
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _stable_hash(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _engine_version_hash() -> str:
    root = Path(__file__).resolve().parent.parent
    candidates = [
        root / "services" / "policy_simulation.py",
        root / "services" / "paper.py",
        root / "engine" / "signal_engine.py",
        root / "engine" / "backtester.py",
    ]
    sha = hashlib.sha256()
    for path in candidates:
        if not path.exists():
            continue
        sha.update(path.name.encode("utf-8"))
        sha.update(path.read_bytes())
    return sha.hexdigest()[:16]


def _resolve_window(payload: dict[str, Any]) -> tuple[date, date]:
    start_date = _parse_date(payload.get("start_date"), field_name="start_date")
    end_date = _parse_date(payload.get("end_date"), field_name="end_date")
    if start_date is None and end_date is None:
        window_days = max(1, _safe_int(payload.get("window_days"), 20))
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=window_days - 1)
    elif start_date is None:
        raise APIError(code="invalid_window", message="start_date is required when end_date is set.")
    elif end_date is None:
        raise APIError(code="invalid_window", message="end_date is required when start_date is set.")
    if start_date > end_date:
        raise APIError(code="invalid_window", message="start_date must be <= end_date.")
    return start_date, end_date


def _resolve_timeframe(policy: Policy) -> str:
    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    timeframes = definition.get("timeframes", [])
    if isinstance(timeframes, list) and timeframes:
        timeframe = str(timeframes[0])
        return timeframe or "1d"
    return "1d"


def _trading_days(
    *,
    session: Session,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    start_date: date,
    end_date: date,
) -> list[date]:
    symbols = store.get_bundle_symbols(session, bundle_id, timeframe=timeframe)
    if not symbols:
        raise APIError(
            code="missing_data",
            message=f"No symbols found for bundle_id={bundle_id} timeframe={timeframe}.",
        )
    anchor_symbol = sorted(symbols)[0]
    frame = store.load_ohlcv(
        symbol=anchor_symbol,
        timeframe=timeframe,
        start=_utc_datetime(start_date),
        end=_utc_datetime(end_date, end=True),
    )
    if frame.empty:
        raise APIError(
            code="missing_data",
            message=f"No OHLCV rows available in replay range for {anchor_symbol}.",
        )
    days = pd.to_datetime(frame["datetime"], utc=True).dt.date.unique().tolist()
    return sorted([day for day in days if start_date <= day <= end_date])


def execute_replay_run(
    *,
    session: Session,
    store: DataStore,
    settings: Settings,
    payload: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    bundle_id = _safe_int(payload.get("bundle_id"), 0)
    if bundle_id <= 0:
        raise APIError(code="invalid_payload", message="bundle_id is required.")

    policy_id = _safe_int(payload.get("policy_id"), 0)
    policy = session.get(Policy, policy_id)
    if policy is None:
        raise APIError(code="not_found", message="Policy not found.", status_code=404)

    start_date, end_date = _resolve_window(payload)
    regime = str(payload.get("regime")) if payload.get("regime") else None
    seed = _safe_int(payload.get("seed"), 7)
    timeframe = _resolve_timeframe(policy)

    replay = ReplayRun(
        bundle_id=bundle_id,
        policy_id=policy_id,
        regime=regime,
        start_date=start_date,
        end_date=end_date,
        seed=seed,
        status="RUNNING",
        summary_json={},
    )
    session.add(replay)
    session.commit()
    session.refresh(replay)

    trading_days = _trading_days(
        session=session,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
    )
    if not trading_days:
        raise APIError(code="missing_data", message="No trading days found for replay range.")

    if progress_cb:
        progress_cb(10, f"Replay initialized with {len(trading_days)} trading days")

    daily_rows: list[dict[str, Any]] = []
    final_summary: dict[str, Any] | None = None
    total_days = max(1, len(trading_days))
    for idx, asof_day in enumerate(trading_days, start=1):
        day_summary = simulate_policy_on_bundle(
            session=session,
            store=store,
            settings=settings,
            policy=policy,
            bundle_id=bundle_id,
            start_date=start_date,
            end_date=asof_day,
            regime=regime,
            seed=seed,
        )
        metrics = day_summary.get("metrics", {})
        daily_rows.append(
            {
                "asof_date": asof_day.isoformat(),
                "period_return": float(metrics.get("period_return", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "calmar": float(metrics.get("calmar", 0.0)),
                "cvar_95": float(metrics.get("cvar_95", 0.0)),
                "cost_ratio": float(metrics.get("cost_ratio", 0.0)),
                "score": float(metrics.get("score", 0.0)),
                "digest": str(day_summary.get("digest", "")),
            }
        )
        final_summary = day_summary
        if progress_cb:
            progress = 10 + int((idx / total_days) * 85)
            progress_cb(progress, f"Replay day {idx}/{total_days}: {asof_day.isoformat()}")

    if final_summary is None:
        raise APIError(code="replay_failed", message="Replay produced no summary.")

    summary_json = {
        "bundle_id": bundle_id,
        "policy_id": policy_id,
        "policy_name": policy.name,
        "regime": regime,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timeframe": timeframe,
        "seed": seed,
        "engine_version": _engine_version_hash(),
        "trading_days": [day.isoformat() for day in trading_days],
        "daily": daily_rows,
        "final": final_summary,
    }
    summary_json["digest"] = _stable_hash(summary_json)

    replay.status = "SUCCEEDED"
    replay.summary_json = summary_json
    session.add(replay)
    session.commit()
    session.refresh(replay)

    if progress_cb:
        progress_cb(100, "Replay finished")

    return {
        "replay_run_id": int(replay.id),
        "status": replay.status,
        "summary": summary_json,
    }


def list_replay_runs(
    session: Session,
    *,
    page: int,
    page_size: int,
) -> tuple[list[ReplayRun], int]:
    rows = session.exec(select(ReplayRun).order_by(ReplayRun.created_at.desc())).all()
    total = len(rows)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return rows[start:end], total


def get_replay_run(session: Session, replay_id: int) -> ReplayRun:
    row = session.get(ReplayRun, replay_id)
    if row is None:
        raise APIError(code="not_found", message="Replay run not found", status_code=404)
    return row
