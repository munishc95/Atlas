from __future__ import annotations

from collections import Counter
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from typing import Any, Callable

import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import Policy, ReplayRun
from app.engine.simulator import ENGINE_VERSION
from app.services.data_store import DataStore
from app.services.paper import preview_policy_signals
from app.services.policy_simulation import simulate_policy_on_bundle


ProgressCallback = Callable[[int, str], None]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
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


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


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


def _audit_offsets(payload: dict[str, Any]) -> list[int]:
    raw = payload.get("audit_offsets_days")
    if isinstance(raw, list):
        values = [_safe_int(item, 0) for item in raw]
    else:
        values = [7, 14, 30]
    cleaned = sorted({value for value in values if value > 0})
    return cleaned or [7, 14, 30]


def _checkpoint_days(
    *,
    trading_days: list[date],
    end_date: date,
    offsets_days: list[int],
) -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    seen: set[date] = set()
    for offset in offsets_days:
        target = end_date - timedelta(days=offset)
        candidates = [day for day in trading_days if day <= target and day < end_date]
        if not candidates:
            continue
        asof_day = candidates[-1]
        if asof_day in seen:
            continue
        seen.add(asof_day)
        checkpoints.append(
            {
                "label": f"{offset}d",
                "offset_days": int(offset),
                "target_date": target.isoformat(),
                "asof_date": asof_day.isoformat(),
            }
        )
    return checkpoints


def _signal_day(value: Any, *, fallback: date) -> date:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return fallback
    return parsed.date()


def _first_hit_day(
    frame: pd.DataFrame,
    *,
    side: str,
    price: float,
    condition: str,
) -> str | None:
    if price <= 0:
        return None
    side_norm = side.upper()
    for row in frame.to_dict(orient="records"):
        high = _safe_float(row.get("high"))
        low = _safe_float(row.get("low"))
        hit = False
        if condition == "upside":
            hit = high >= price if side_norm == "BUY" else low <= price
        elif condition == "stop":
            hit = low <= price if side_norm == "BUY" else high >= price
        elif condition == "target":
            hit = high >= price if side_norm == "BUY" else low <= price
        if hit:
            return pd.Timestamp(row["datetime"]).date().isoformat()
    return None


def _evaluate_signal_outcome(
    *,
    session: Session,
    store: DataStore,
    signal: dict[str, Any],
    timeframe: str,
    end_date: date,
    min_upside_pct: float,
) -> dict[str, Any]:
    symbol = str(signal.get("symbol", "")).strip().upper()
    side = str(signal.get("side", "BUY")).strip().upper()
    if not symbol or side not in {"BUY", "SELL"}:
        return {"symbol": symbol, "side": side, "available": False, "reason": "invalid_signal"}

    fill_day = _signal_day(signal.get("fill_at"), fallback=end_date)
    frame = store.load_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start=_utc_datetime(fill_day),
        end=_utc_datetime(end_date, end=True),
        session=session,
    )
    if frame.empty:
        return {
            "symbol": symbol,
            "side": side,
            "available": False,
            "reason": "missing_outcome_bars",
            "fill_date": fill_day.isoformat(),
        }

    frame = frame.copy().sort_values("datetime").reset_index(drop=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    entry_price = _safe_float(signal.get("price"), 0.0)
    if entry_price <= 0:
        entry_price = _safe_float(frame.iloc[0].get("open"), 0.0)
    if entry_price <= 0:
        return {
            "symbol": symbol,
            "side": side,
            "available": False,
            "reason": "invalid_entry_price",
            "fill_date": fill_day.isoformat(),
        }

    highs = pd.to_numeric(frame["high"], errors="coerce")
    lows = pd.to_numeric(frame["low"], errors="coerce")
    closes = pd.to_numeric(frame["close"], errors="coerce")
    last_close = _safe_float(closes.iloc[-1], entry_price)
    last_day = pd.Timestamp(frame.iloc[-1]["datetime"]).date()
    stop_distance = _safe_float(signal.get("stop_distance"), 0.0)
    stop_price = (
        entry_price - stop_distance
        if side == "BUY" and stop_distance > 0
        else entry_price + stop_distance
        if side == "SELL" and stop_distance > 0
        else 0.0
    )
    target_price = _safe_float(signal.get("target_price"), 0.0)
    upside_price = (
        entry_price * (1.0 + min_upside_pct / 100.0)
        if side == "BUY"
        else entry_price * (1.0 - min_upside_pct / 100.0)
    )

    if side == "BUY":
        close_return_pct = (last_close / entry_price - 1.0) * 100.0
        max_favorable_pct = (_safe_float(highs.max(), entry_price) / entry_price - 1.0) * 100.0
        max_adverse_pct = min(0.0, (_safe_float(lows.min(), entry_price) / entry_price - 1.0) * 100.0)
    else:
        close_return_pct = (entry_price / last_close - 1.0) * 100.0 if last_close > 0 else 0.0
        min_low = _safe_float(lows.min(), entry_price)
        max_high = _safe_float(highs.max(), entry_price)
        max_favorable_pct = (entry_price / min_low - 1.0) * 100.0 if min_low > 0 else 0.0
        max_adverse_pct = min(0.0, (entry_price / max_high - 1.0) * 100.0) if max_high > 0 else 0.0

    first_upside_day = _first_hit_day(
        frame,
        side=side,
        price=upside_price,
        condition="upside",
    )
    first_stop_day = (
        _first_hit_day(frame, side=side, price=stop_price, condition="stop")
        if stop_price > 0
        else None
    )
    first_target_day = (
        _first_hit_day(frame, side=side, price=target_price, condition="target")
        if target_price > 0
        else None
    )
    upside_before_stop = bool(first_upside_day) and (
        first_stop_day is None or first_upside_day <= first_stop_day
    )
    positive_close = close_return_pct > 0.0
    stopped_before_upside = bool(first_stop_day) and (
        first_upside_day is None or first_stop_day < first_upside_day
    )
    if upside_before_stop:
        verdict = "WORKED"
    elif stopped_before_upside:
        verdict = "STOPPED"
    elif positive_close:
        verdict = "DRIFT_POSITIVE"
    else:
        verdict = "FAILED"

    return {
        "symbol": symbol,
        "underlying_symbol": str(signal.get("underlying_symbol", symbol)).upper(),
        "side": side,
        "template": str(signal.get("template", "")),
        "instrument_kind": str(signal.get("instrument_kind", "EQUITY_CASH")).upper(),
        "available": True,
        "decision_at": str(signal.get("signal_at", "")),
        "fill_at": str(signal.get("fill_at", "")),
        "fill_date": fill_day.isoformat(),
        "end_date": last_day.isoformat(),
        "bars_held": int(len(frame)),
        "entry_price": float(entry_price),
        "last_close": float(last_close),
        "close_return_pct": float(close_return_pct),
        "max_favorable_pct": float(max_favorable_pct),
        "max_adverse_pct": float(max_adverse_pct),
        "min_upside_pct": float(min_upside_pct),
        "upside_price": float(upside_price),
        "upside_hit": bool(first_upside_day),
        "upside_hit_at": first_upside_day,
        "stop_price": float(stop_price) if stop_price > 0 else None,
        "stop_hit": bool(first_stop_day),
        "stop_hit_at": first_stop_day,
        "target_price": float(target_price) if target_price > 0 else None,
        "target_hit": bool(first_target_day),
        "target_hit_at": first_target_day,
        "worked": bool(upside_before_stop),
        "positive_close": bool(positive_close),
        "stopped_before_upside": bool(stopped_before_upside),
        "verdict": verdict,
        "quality_status": str(signal.get("quality_status", "PASS")).upper(),
        "quality_score": _safe_float(signal.get("quality_score"), 0.0),
        "quality_flags": [str(flag) for flag in (signal.get("quality_flags", []) or [])],
        "quality_metrics": signal.get("quality_metrics", {}),
        "market_context": signal.get("market_context", {}),
    }


def _summarize_signal_outcomes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if bool(row.get("available"))]
    eligible = [row for row in available if str(row.get("quality_status", "PASS")) != "FAIL"]
    verdicts = Counter(str(row.get("verdict", "UNKNOWN")) for row in eligible)
    failed_flags: Counter[str] = Counter()
    for row in eligible:
        if str(row.get("verdict")) in {"FAILED", "STOPPED"}:
            failed_flags.update(str(flag) for flag in (row.get("quality_flags") or []) if str(flag))

    total = len(eligible)
    worked = sum(1 for row in eligible if bool(row.get("worked")))
    positive_close = sum(1 for row in eligible if bool(row.get("positive_close")))
    stopped = sum(1 for row in eligible if bool(row.get("stop_hit")))
    avg_close = (
        sum(_safe_float(row.get("close_return_pct")) for row in eligible) / total if total else 0.0
    )
    avg_mfe = (
        sum(_safe_float(row.get("max_favorable_pct")) for row in eligible) / total if total else 0.0
    )
    avg_mae = (
        sum(_safe_float(row.get("max_adverse_pct")) for row in eligible) / total if total else 0.0
    )
    return {
        "signal_count": len(rows),
        "available_count": len(available),
        "eligible_count": total,
        "quality_fail_count": len(available) - total,
        "worked_count": worked,
        "worked_rate": worked / total if total else 0.0,
        "positive_close_count": positive_close,
        "positive_close_rate": positive_close / total if total else 0.0,
        "stop_hit_count": stopped,
        "stop_hit_rate": stopped / total if total else 0.0,
        "avg_close_return_pct": float(avg_close),
        "avg_max_favorable_pct": float(avg_mfe),
        "avg_max_adverse_pct": float(avg_mae),
        "verdict_counts": dict(verdicts),
        "failed_quality_flags": dict(failed_flags.most_common(12)),
    }


def _build_signal_audit(
    *,
    session: Session,
    store: DataStore,
    settings: Settings,
    payload: dict[str, Any],
    policy: Policy,
    bundle_id: int,
    regime: str | None,
    seed: int,
    timeframe: str,
    trading_days: list[date],
    end_date: date,
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    min_upside_pct = max(0.1, _safe_float(payload.get("audit_min_upside_pct"), 2.0))
    max_signals = max(1, _safe_int(payload.get("audit_max_signals_per_checkpoint"), 25))
    audit_runtime_seconds = max(60, _safe_int(payload.get("audit_max_runtime_seconds"), 180))
    checkpoints = _checkpoint_days(
        trading_days=trading_days,
        end_date=end_date,
        offsets_days=_audit_offsets(payload),
    )
    checkpoint_rows: list[dict[str, Any]] = []
    all_outcomes: list[dict[str, Any]] = []
    for idx, checkpoint in enumerate(checkpoints, start=1):
        asof_day = date.fromisoformat(str(checkpoint["asof_date"]))
        preview = preview_policy_signals(
            session=session,
            settings=settings,
            store=store,
            payload={
                "regime": regime or "TREND_UP",
                "bundle_id": bundle_id,
                "policy_id": int(policy.id or 0),
                "timeframes": [timeframe],
                "seed": int(seed),
                "asof": _utc_datetime(asof_day).isoformat(),
                "max_runtime_seconds": audit_runtime_seconds,
                "runtime_hard_cap_seconds": audit_runtime_seconds,
            },
        )
        signals = list(preview.get("signals", []) or [])[:max_signals]
        outcomes = [
            _evaluate_signal_outcome(
                session=session,
                store=store,
                signal=dict(signal),
                timeframe=timeframe,
                end_date=end_date,
                min_upside_pct=min_upside_pct,
            )
            for signal in signals
        ]
        all_outcomes.extend(outcomes)
        checkpoint_rows.append(
            {
                **checkpoint,
                "generated_signals_count": int(preview.get("generated_signals_count", 0)),
                "audited_signals_count": len(signals),
                "scan_truncated": bool(preview.get("scan_truncated", False)),
                "scanned_symbols": int(preview.get("scanned_symbols", 0)),
                "evaluated_candidates": int(preview.get("evaluated_candidates", 0)),
                "candidate_quality": preview.get("candidate_quality", {}),
                "summary": _summarize_signal_outcomes(outcomes),
                "signals": outcomes,
            }
        )
        if progress_cb:
            progress_cb(95 + int((idx / max(1, len(checkpoints))) * 4), f"Signal audit {idx}/{len(checkpoints)}")

    return {
        "enabled": True,
        "method": "point_in_time_signal_outcome",
        "description": (
            "Signals are generated as-of each checkpoint with next-bar fill timing, then audited "
            "through the replay end date."
        ),
        "min_upside_pct": float(min_upside_pct),
        "max_signals_per_checkpoint": int(max_signals),
        "max_runtime_seconds": int(audit_runtime_seconds),
        "offsets_days": _audit_offsets(payload),
        "checkpoint_count": len(checkpoint_rows),
        "summary": _summarize_signal_outcomes(all_outcomes),
        "checkpoints": checkpoint_rows,
    }


def _trading_days(
    *,
    session: Session,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    start_date: date,
    end_date: date,
) -> list[date]:
    symbols = store.get_bundle_symbols(
        session,
        bundle_id,
        timeframe=timeframe,
        asof_date=end_date,
    )
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
        session=session,
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
    include_signal_audit = _parse_bool(payload.get("include_signal_audit"), default=True)

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

    signal_audit: dict[str, Any] = {"enabled": False}
    if include_signal_audit:
        if progress_cb:
            progress_cb(95, "Signal outcome audit started")
        signal_audit = _build_signal_audit(
            session=session,
            store=store,
            settings=settings,
            payload=payload,
            policy=policy,
            bundle_id=bundle_id,
            regime=regime,
            seed=seed,
            timeframe=timeframe,
            trading_days=trading_days,
            end_date=end_date,
            progress_cb=progress_cb,
        )

    summary_json = {
        "bundle_id": bundle_id,
        "policy_id": policy_id,
        "policy_name": policy.name,
        "regime": regime,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timeframe": timeframe,
        "seed": seed,
        "engine_version": str((final_summary.get("engine_version") or ENGINE_VERSION)),
        "data_digest": str(final_summary.get("data_digest", "")),
        "trading_days": [day.isoformat() for day in trading_days],
        "daily": daily_rows,
        "final": final_summary,
        "signal_audit": signal_audit,
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
