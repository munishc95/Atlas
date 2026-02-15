from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Callable

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import (
    AutoEvalRun,
    DataQualityReport,
    PaperRun,
    Policy,
    PolicySwitchEvent,
)
from app.services.data_store import DataStore
from app.services.fast_mode import resolve_seed
from app.services.operate_events import emit_operate_event
from app.services.paper import get_or_create_paper_state
from app.services.policy_health import DEGRADED, compute_health_metrics, get_policy_health_snapshot
from app.services.policy_simulation import simulate_policy_on_bundle
from app.services.trading_calendar import is_trading_day, list_trading_days, previous_trading_day


ProgressCallback = Callable[[int, str], None]

ACTION_KEEP = "KEEP"
ACTION_SWITCH = "SWITCH"
ACTION_SHADOW_ONLY = "SHADOW_ONLY"


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


def _parse_iso_date(value: Any) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _metrics_score(metrics: dict[str, Any]) -> float:
    calmar = _safe_float(metrics.get("calmar"), 0.0)
    max_dd = abs(_safe_float(metrics.get("max_drawdown"), 0.0))
    profit_factor = _safe_float(metrics.get("profit_factor"), 0.0)
    tail_loss = abs(_safe_float(metrics.get("cvar_95"), 0.0))
    turnover = max(0.0, _safe_float(metrics.get("turnover"), 0.0))
    return float(
        (0.40 * calmar)
        - (0.25 * max_dd)
        + (0.20 * profit_factor)
        - (0.10 * tail_loss)
        - (0.05 * turnover)
    )


def _digest(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _window_start(asof_day: date, lookback_days: int, *, settings: Settings, segment: str) -> date:
    cursor = asof_day
    for _ in range(max(0, lookback_days - 1)):
        cursor = previous_trading_day(cursor, segment=segment, settings=settings)
    return cursor


def _resolve_active_policy(session: Session, payload: dict[str, Any], state_settings: dict[str, Any]) -> Policy:
    from_payload = _safe_int(payload.get("active_policy_id"), 0)
    if from_payload > 0:
        policy_id = from_payload
    else:
        policy_id = _safe_int(state_settings.get("active_policy_id"), 0)
    if policy_id <= 0:
        raise APIError(
            code="invalid_state",
            message="No active policy found for auto evaluation.",
            status_code=409,
        )
    policy = session.get(Policy, policy_id)
    if policy is None:
        raise APIError(code="not_found", message="Active policy not found.", status_code=404)
    return policy


def _resolve_bundle_id(session: Session, *, payload: dict[str, Any], policy: Policy) -> int:
    from_payload = _safe_int(payload.get("bundle_id"), 0)
    if from_payload > 0:
        return from_payload

    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    if latest_run is not None and latest_run.bundle_id is not None:
        return int(latest_run.bundle_id)

    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    universe = definition.get("universe", {})
    if isinstance(universe, dict):
        from_policy = _safe_int(universe.get("bundle_id"), 0)
        if from_policy > 0:
            return from_policy

    raise APIError(
        code="invalid_state",
        message="bundle_id required (payload, active run context, or policy universe).",
        status_code=409,
    )


def _resolve_timeframe(payload: dict[str, Any], policy: Policy) -> str:
    timeframe = str(payload.get("timeframe") or "").strip()
    if timeframe:
        return timeframe
    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    timeframes = definition.get("timeframes", [])
    if isinstance(timeframes, list) and timeframes:
        first = str(timeframes[0]).strip()
        if first:
            return first
    return "1d"


def _resolve_challengers(session: Session, *, payload: dict[str, Any], active_policy_id: int) -> list[Policy]:
    challenger_ids: list[int] = []
    raw = payload.get("challenger_policy_ids")
    if isinstance(raw, list) and raw:
        challenger_ids = [_safe_int(item, 0) for item in raw if _safe_int(item, 0) > 0]
    if challenger_ids:
        rows = session.exec(
            select(Policy).where(Policy.id.in_(challenger_ids)).order_by(Policy.created_at.desc())
        ).all()
        return [row for row in rows if row.id is not None and int(row.id) != active_policy_id]

    rows = session.exec(select(Policy).order_by(Policy.created_at.desc())).all()
    return [row for row in rows if row.id is not None and int(row.id) != active_policy_id]


def _latest_quality(session: Session, *, bundle_id: int, timeframe: str) -> tuple[str | None, float]:
    row = session.exec(
        select(DataQualityReport)
        .where(DataQualityReport.bundle_id == bundle_id)
        .where(DataQualityReport.timeframe == timeframe)
        .order_by(DataQualityReport.created_at.desc())
    ).first()
    if row is None:
        return None, 100.0
    return str(row.status).upper(), float(row.coverage_pct)


def _switch_gates(
    session: Session,
    *,
    asof_day: date,
    settings_map: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    cooldown_trading_days = max(
        1,
        _safe_int(
            settings_map.get(
                "operate_auto_eval_cooldown_trading_days",
                settings.operate_auto_eval_cooldown_trading_days,
            ),
            settings.operate_auto_eval_cooldown_trading_days,
        ),
    )
    max_switches = max(
        1,
        _safe_int(
            settings_map.get(
                "operate_auto_eval_max_switches_per_30d",
                settings.operate_auto_eval_max_switches_per_30d,
            ),
            settings.operate_auto_eval_max_switches_per_30d,
        ),
    )

    since_30d = datetime.combine(asof_day, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=30)
    recent = list(
        session.exec(
            select(PolicySwitchEvent)
            .where(PolicySwitchEvent.ts >= since_30d)
            .order_by(PolicySwitchEvent.ts.desc())
        ).all()
    )
    last_switch = recent[0] if recent else None

    trading_days_since_last = 99_999
    if last_switch is not None:
        last_day = last_switch.ts.astimezone(timezone.utc).date()
        trading_days_since_last = max(
            0,
            len(
                list_trading_days(
                    start_date=last_day,
                    end_date=asof_day,
                    segment=str(settings_map.get("trading_calendar_segment", settings.trading_calendar_segment)),
                    settings=settings,
                )
            )
            - 1,
        )

    return {
        "cooldown_trading_days": cooldown_trading_days,
        "max_switches_per_30d": max_switches,
        "recent_switches_30d": len(recent),
        "trading_days_since_last_switch": trading_days_since_last,
        "cooldown_blocked": trading_days_since_last < cooldown_trading_days,
        "max_switches_blocked": len(recent) >= max_switches,
    }


def _apply_switch(
    session: Session,
    *,
    settings: Settings,
    from_policy: Policy,
    to_policy: Policy,
    auto_eval_id: int,
    reason: str,
    cooldown_state: dict[str, Any],
    mode: str,
) -> None:
    state = get_or_create_paper_state(session, settings)
    settings_map = dict(state.settings_json or {})
    settings_map["paper_mode"] = "policy"
    settings_map["active_policy_id"] = int(to_policy.id)
    settings_map["active_policy_name"] = to_policy.name
    state.settings_json = settings_map
    session.add(state)
    session.add(
        PolicySwitchEvent(
            from_policy_id=int(from_policy.id),
            to_policy_id=int(to_policy.id),
            reason=reason[:512],
            auto_eval_id=auto_eval_id,
            cooldown_state_json=dict(cooldown_state),
            mode=mode.upper(),
        )
    )


def execute_auto_evaluation(
    *,
    session: Session,
    store: DataStore,
    settings: Settings,
    payload: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    settings_map = dict(state.settings_json or {})
    segment = str(settings_map.get("trading_calendar_segment", settings.trading_calendar_segment))

    active_policy = _resolve_active_policy(session, payload, settings_map)
    bundle_id = _resolve_bundle_id(session, payload=payload, policy=active_policy)
    timeframe = _resolve_timeframe(payload, active_policy)

    lookback_days = max(
        5,
        _safe_int(
            payload.get("lookback_trading_days")
            or settings_map.get("operate_auto_eval_lookback_trading_days")
            or settings.operate_auto_eval_lookback_trading_days,
            settings.operate_auto_eval_lookback_trading_days,
        ),
    )
    min_trades = max(
        1,
        _safe_int(
            payload.get("min_trades")
            or settings_map.get("operate_auto_eval_min_trades")
            or settings.operate_auto_eval_min_trades,
            settings.operate_auto_eval_min_trades,
        ),
    )
    seed = resolve_seed(settings=settings, value=payload.get("seed"), default=7)
    asof_day = _parse_iso_date(payload.get("asof_date")) or datetime.now(timezone.utc).date()
    if not is_trading_day(asof_day, segment=segment, settings=settings):
        asof_day = previous_trading_day(asof_day, segment=segment, settings=settings)
    window_start = _window_start(asof_day, lookback_days, settings=settings, segment=segment)

    active_runs = list(
        session.exec(
            select(PaperRun)
            .where(PaperRun.policy_id == int(active_policy.id))
            .where(PaperRun.bundle_id == int(bundle_id))
            .where(PaperRun.asof_ts >= datetime.combine(window_start, datetime.min.time(), tzinfo=timezone.utc))
            .where(PaperRun.asof_ts <= datetime.combine(asof_day, datetime.max.time(), tzinfo=timezone.utc))
            .order_by(PaperRun.asof_ts.asc())
        ).all()
    )
    active_metrics = compute_health_metrics(active_runs, window_days=lookback_days)
    active_score = _metrics_score(active_metrics)
    active_trade_count = int(_safe_float(active_metrics.get("trade_count"), 0.0))
    active_run_count = len(active_runs)

    if progress_cb:
        progress_cb(15, "Auto-eval baseline metrics computed")

    quality_status, quality_coverage = _latest_quality(session, bundle_id=bundle_id, timeframe=timeframe)
    quality_ok = quality_status not in {"FAIL"}
    safe_mode_shadow_gate = bool(
        settings_map.get("operate_auto_eval_shadow_only_gate", settings.operate_auto_eval_shadow_only_gate)
    )
    safe_mode_active = bool(
        str(quality_status or "").upper() == "FAIL"
        and bool(settings_map.get("operate_safe_mode_on_fail", settings.operate_safe_mode_on_fail))
    )
    safe_mode_action = str(settings_map.get("operate_safe_mode_action", settings.operate_safe_mode_action)).lower()

    short_window = max(5, _safe_int(settings_map.get("health_window_days_short"), settings.health_window_days_short))
    long_window = max(short_window, _safe_int(settings_map.get("health_window_days_long"), settings.health_window_days_long))
    health_short = get_policy_health_snapshot(
        session,
        settings=settings,
        policy=active_policy,
        window_days=short_window,
        asof_date=asof_day,
        refresh=True,
        overrides=settings_map,
    )
    health_long = get_policy_health_snapshot(
        session,
        settings=settings,
        policy=active_policy,
        window_days=long_window,
        asof_date=asof_day,
        refresh=True,
        overrides=settings_map,
    )

    degradations: list[str] = []
    if health_short.status == DEGRADED or health_long.status == DEGRADED:
        degradations.append("Active policy health is DEGRADED.")
    baseline = active_policy.definition_json.get("baseline", {}) if isinstance(active_policy.definition_json, dict) else {}
    baseline_tail = abs(_safe_float((baseline or {}).get("cvar_95"), 0.0))
    current_tail = abs(_safe_float(active_metrics.get("cvar_95"), 0.0))
    if baseline_tail > 0 and current_tail > baseline_tail * 1.5:
        degradations.append("Tail loss exceeded baseline guardrail.")

    challengers = _resolve_challengers(session, payload=payload, active_policy_id=int(active_policy.id))
    challenger_rows: list[dict[str, Any]] = []
    total = max(1, len(challengers))
    score_margin = _safe_float(settings_map.get("evaluations_score_margin"), settings.evaluations_score_margin)
    max_dd_multiplier = _safe_float(
        settings_map.get("evaluations_max_dd_multiplier"),
        settings.evaluations_max_dd_multiplier,
    )
    active_max_dd_abs = abs(_safe_float(active_metrics.get("max_drawdown"), 0.0))
    allow_switch_candidates = (
        active_trade_count >= min_trades and active_run_count >= max(5, lookback_days // 2)
    )

    for idx, challenger in enumerate(challengers, start=1):
        summary = simulate_policy_on_bundle(
            session=session,
            store=store,
            settings=settings,
            policy=challenger,
            bundle_id=bundle_id,
            start_date=window_start,
            end_date=asof_day,
            regime=None,
            seed=seed,
        )
        metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
        trade_count = int(
            sum(
                _safe_int(row.get("trade_count"), 0)
                for row in (summary.get("symbol_rows") or [])
                if isinstance(row, dict)
            )
        )
        candidate_score = _metrics_score(metrics)
        max_dd = abs(_safe_float(metrics.get("max_drawdown"), 0.0))
        dd_limit = active_max_dd_abs * max_dd_multiplier if active_max_dd_abs > 0 else settings.kill_switch_drawdown
        dd_pass = max_dd <= dd_limit
        score_pass = candidate_score >= active_score + score_margin
        min_trades_pass = trade_count >= min_trades
        accepted = bool(dd_pass and score_pass and min_trades_pass and allow_switch_candidates)
        reasons: list[str] = []
        if not allow_switch_candidates:
            reasons.append("Active policy does not meet minimum sample/trade requirements.")
        if not dd_pass:
            reasons.append("Rejected: challenger drawdown exceeds threshold.")
        if not score_pass:
            reasons.append("Rejected: score improvement below required margin.")
        if not min_trades_pass:
            reasons.append("Rejected: challenger trade count below minimum.")
        challenger_rows.append(
            {
                "policy_id": int(challenger.id),
                "policy_name": challenger.name,
                "metrics": metrics,
                "trade_count": trade_count,
                "score": candidate_score,
                "passes": accepted,
                "reasons": reasons,
                "engine_version": summary.get("engine_version"),
                "data_digest": summary.get("data_digest"),
            }
        )
        if progress_cb:
            progress_cb(15 + int((idx / total) * 55), f"Auto-eval challenger {idx}/{total}")

    challenger_rows.sort(key=lambda row: float(row.get("score", -1e9)), reverse=True)
    best = challenger_rows[0] if challenger_rows else None
    recommended_action = ACTION_KEEP
    recommended_policy_id: int | None = None
    reasons: list[str] = []

    if active_trade_count < min_trades:
        reasons.append("KEEP: active policy has insufficient recent trades for reliable switching.")
    if active_run_count < max(5, lookback_days // 2):
        reasons.append("KEEP: insufficient active policy sample window.")
    if degradations:
        recommended_action = ACTION_SHADOW_ONLY
        reasons.extend([f"SHADOW_ONLY: {msg}" for msg in degradations])
    elif best is not None and bool(best.get("passes")):
        recommended_action = ACTION_SWITCH
        recommended_policy_id = int(best["policy_id"])
        reasons.append("SWITCH: challenger outperformed by required margin with acceptable drawdown.")
    else:
        reasons.append("KEEP: no challenger passed all deterministic gates.")

    switch_gates = _switch_gates(session, asof_day=asof_day, settings_map=settings_map, settings=settings)
    if not quality_ok:
        switch_gates["data_quality_blocked"] = True
    if safe_mode_shadow_gate and safe_mode_active and safe_mode_action == "shadow_only":
        switch_gates["shadow_only_gate_blocked"] = True
    else:
        switch_gates["shadow_only_gate_blocked"] = False

    auto_switch_enabled = bool(
        payload.get("auto_switch")
        if payload.get("auto_switch") is not None
        else settings_map.get("operate_auto_eval_auto_switch", settings.operate_auto_eval_auto_switch)
    )
    operate_mode = str(settings_map.get("operate_mode", settings.operate_mode)).strip().lower()
    can_auto_switch = (
        auto_switch_enabled
        and recommended_action == ACTION_SWITCH
        and recommended_policy_id is not None
        and operate_mode == "live"
        and quality_ok
        and not bool(switch_gates.get("cooldown_blocked"))
        and not bool(switch_gates.get("max_switches_blocked"))
        and not bool(switch_gates.get("shadow_only_gate_blocked"))
    )

    if not quality_ok:
        reasons.append("Switch blocked: data quality is not OK.")
    if bool(switch_gates.get("cooldown_blocked")):
        reasons.append("Switch blocked: cooldown window not elapsed.")
    if bool(switch_gates.get("max_switches_blocked")):
        reasons.append("Switch blocked: max switches in rolling 30d reached.")
    if bool(switch_gates.get("shadow_only_gate_blocked")):
        reasons.append("Switch blocked: safe mode shadow-only gate is active.")
    if operate_mode != "live" and auto_switch_enabled:
        reasons.append("Auto-switch disabled in offline operate mode.")

    score_table = {
        "active": {
            "policy_id": int(active_policy.id),
            "policy_name": active_policy.name,
            "score": active_score,
            "metrics": active_metrics,
            "trade_count": active_trade_count,
            "run_count": active_run_count,
        },
        "challengers": challenger_rows,
        "gates": {
            "lookback_trading_days": lookback_days,
            "min_trades": min_trades,
            "score_margin": score_margin,
            "max_dd_multiplier": max_dd_multiplier,
            "switch": switch_gates,
            "quality_status": quality_status,
            "quality_coverage_pct": quality_coverage,
            "health_short_status": health_short.status,
            "health_long_status": health_long.status,
        },
    }
    digest = _digest(
        {
            "asof_date": asof_day.isoformat(),
            "bundle_id": bundle_id,
            "active_policy_id": int(active_policy.id),
            "recommended_action": recommended_action,
            "recommended_policy_id": recommended_policy_id,
            "score_table": score_table,
            "seed": seed,
        }
    )

    auto_eval = AutoEvalRun(
        bundle_id=bundle_id,
        active_policy_id=int(active_policy.id),
        recommended_action=recommended_action,
        recommended_policy_id=recommended_policy_id,
        reasons_json=reasons,
        score_table_json=score_table,
        lookback_days=lookback_days,
        digest=digest,
        status="SUCCEEDED",
        auto_switch_attempted=bool(auto_switch_enabled),
        auto_switch_applied=False,
        details_json={
            "timeframe": timeframe,
            "asof_date": asof_day.isoformat(),
            "window_start": window_start.isoformat(),
            "window_end": asof_day.isoformat(),
            "quality_status": quality_status,
            "quality_coverage_pct": quality_coverage,
            "seed": seed,
            "operate_mode": operate_mode,
        },
    )
    session.add(auto_eval)
    session.commit()
    session.refresh(auto_eval)

    switched_to_policy_id: int | None = None
    if can_auto_switch and recommended_policy_id is not None:
        to_policy = session.get(Policy, recommended_policy_id)
        if to_policy is not None:
            _apply_switch(
                session,
                settings=settings,
                from_policy=active_policy,
                to_policy=to_policy,
                auto_eval_id=int(auto_eval.id),
                reason="auto_eval_switch",
                cooldown_state=switch_gates,
                mode="AUTO",
            )
            auto_eval.auto_switch_applied = True
            session.add(auto_eval)
            switched_to_policy_id = int(to_policy.id)
            reasons.append(f"Auto-switch applied: {active_policy.id} -> {to_policy.id}.")
            session.commit()

    emit_operate_event(
        session,
        severity="INFO",
        category="POLICY",
        message="auto_eval_completed",
        details={
            "auto_eval_id": int(auto_eval.id),
            "bundle_id": bundle_id,
            "active_policy_id": int(active_policy.id),
            "recommended_action": recommended_action,
            "recommended_policy_id": recommended_policy_id,
            "auto_switch_attempted": auto_switch_enabled,
            "auto_switch_applied": bool(auto_eval.auto_switch_applied),
        },
        correlation_id=str(auto_eval.id),
    )
    recommendation_event = {
        ACTION_SWITCH: "auto_eval_recommend_switch",
        ACTION_KEEP: "auto_eval_recommend_keep",
        ACTION_SHADOW_ONLY: "auto_eval_recommend_shadow_only",
    }[recommended_action]
    emit_operate_event(
        session,
        severity="WARN" if recommended_action == ACTION_SHADOW_ONLY else "INFO",
        category="POLICY",
        message=recommendation_event,
        details={
            "auto_eval_id": int(auto_eval.id),
            "recommended_policy_id": recommended_policy_id,
            "reasons": reasons,
            "digest": digest,
        },
        correlation_id=str(auto_eval.id),
    )
    session.commit()

    if progress_cb:
        progress_cb(100, "Auto evaluation finished")

    return {
        "auto_eval_id": int(auto_eval.id),
        "recommended_action": recommended_action,
        "recommended_policy_id": recommended_policy_id,
        "auto_switch_applied": bool(auto_eval.auto_switch_applied),
        "switched_to_policy_id": switched_to_policy_id,
        "digest": digest,
        "reasons": reasons,
        "score_table": score_table,
    }


def list_auto_eval_runs(
    session: Session,
    *,
    page: int,
    page_size: int,
    bundle_id: int | None = None,
    policy_id: int | None = None,
) -> tuple[list[AutoEvalRun], int]:
    stmt = select(AutoEvalRun)
    if bundle_id is not None:
        stmt = stmt.where(AutoEvalRun.bundle_id == bundle_id)
    if policy_id is not None:
        stmt = stmt.where(
            (AutoEvalRun.active_policy_id == policy_id)
            | (AutoEvalRun.recommended_policy_id == policy_id)
        )
    rows = list(session.exec(stmt.order_by(AutoEvalRun.ts.desc(), AutoEvalRun.id.desc())).all())
    total = len(rows)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return rows[start:end], total


def get_auto_eval_run(session: Session, auto_eval_id: int) -> AutoEvalRun:
    row = session.get(AutoEvalRun, auto_eval_id)
    if row is None:
        raise APIError(code="not_found", message="Auto evaluation run not found", status_code=404)
    return row


def list_policy_switch_events(session: Session, *, limit: int = 10) -> list[PolicySwitchEvent]:
    return list(
        session.exec(
            select(PolicySwitchEvent)
            .order_by(PolicySwitchEvent.ts.desc(), PolicySwitchEvent.id.desc())
            .limit(max(1, min(int(limit), 200)))
        ).all()
    )
