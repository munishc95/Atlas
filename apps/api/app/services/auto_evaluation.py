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
    PolicyEnsemble,
    PolicyEnsembleMember,
    PolicySwitchEvent,
)
from app.services.data_store import DataStore
from app.services.fast_mode import resolve_seed
from app.services.operate_events import emit_operate_event
from app.services.paper import get_or_create_paper_state
from app.services.policy_health import (
    DEGRADED,
    HEALTHY,
    WARNING,
    compute_health_metrics,
    get_policy_health_snapshot,
)
from app.services.policy_simulation import simulate_policy_on_bundle
from app.services.trading_calendar import is_trading_day, list_trading_days, previous_trading_day


ProgressCallback = Callable[[int, str], None]

ACTION_KEEP = "KEEP"
ACTION_SWITCH = "SWITCH"
ACTION_SHADOW_ONLY = "SHADOW_ONLY"

ENTITY_POLICY = "policy"
ENTITY_ENSEMBLE = "ensemble"


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


def _load_ensemble_members(
    session: Session,
    *,
    ensemble_id: int,
) -> list[tuple[PolicyEnsembleMember, Policy]]:
    rows = session.exec(
        select(PolicyEnsembleMember)
        .where(PolicyEnsembleMember.ensemble_id == int(ensemble_id))
        .where(PolicyEnsembleMember.enabled == True)  # noqa: E712
        .order_by(PolicyEnsembleMember.policy_id.asc(), PolicyEnsembleMember.id.asc())
    ).all()
    out: list[tuple[PolicyEnsembleMember, Policy]] = []
    for row in rows:
        policy = session.get(Policy, int(row.policy_id))
        if policy is None or policy.id is None:
            continue
        out.append((row, policy))
    return out


def _normalize_member_weights(
    rows: list[tuple[PolicyEnsembleMember, Policy]],
) -> list[tuple[PolicyEnsembleMember, Policy, float]]:
    normalized: list[tuple[PolicyEnsembleMember, Policy, float]] = []
    total = 0.0
    for member, policy in rows:
        weight = max(0.0, _safe_float(member.weight, 0.0))
        normalized.append((member, policy, weight))
        total += weight
    if total <= 0:
        return []
    return [(member, policy, weight / total) for member, policy, weight in normalized if weight > 0]


def _resolve_active_candidate(
    session: Session,
    payload: dict[str, Any],
    state_settings: dict[str, Any],
) -> dict[str, Any]:
    payload_ensemble = _safe_int(payload.get("active_ensemble_id"), 0)
    payload_policy = _safe_int(payload.get("active_policy_id"), 0)
    if payload_ensemble > 0 and payload_policy > 0:
        payload_policy = 0

    state_policy = _safe_int(state_settings.get("active_policy_id"), 0)
    state_ensemble = _safe_int(state_settings.get("active_ensemble_id"), 0)

    # Backward-compatible precedence:
    # 1) explicit payload ensemble
    # 2) explicit payload policy
    # 3) state active policy
    # 4) state active ensemble
    if payload_ensemble > 0:
        ensemble = session.get(PolicyEnsemble, payload_ensemble)
        if ensemble is None or ensemble.id is None:
            raise APIError(code="not_found", message="Active ensemble not found.", status_code=404)
        return {
            "entity_type": ENTITY_ENSEMBLE,
            "ensemble": ensemble,
            "policy": None,
            "id": int(ensemble.id),
            "name": ensemble.name,
        }

    if payload_policy > 0:
        policy = session.get(Policy, payload_policy)
        if policy is None or policy.id is None:
            raise APIError(code="not_found", message="Active policy not found.", status_code=404)
        return {
            "entity_type": ENTITY_POLICY,
            "ensemble": None,
            "policy": policy,
            "id": int(policy.id),
            "name": policy.name,
        }

    if state_policy > 0:
        policy = session.get(Policy, state_policy)
        if policy is None or policy.id is None:
            raise APIError(code="not_found", message="Active policy not found.", status_code=404)
        return {
            "entity_type": ENTITY_POLICY,
            "ensemble": None,
            "policy": policy,
            "id": int(policy.id),
            "name": policy.name,
        }

    if state_ensemble > 0:
        ensemble = session.get(PolicyEnsemble, state_ensemble)
        if ensemble is None or ensemble.id is None:
            raise APIError(code="not_found", message="Active ensemble not found.", status_code=404)
        return {
            "entity_type": ENTITY_ENSEMBLE,
            "ensemble": ensemble,
            "policy": None,
            "id": int(ensemble.id),
            "name": ensemble.name,
        }

    raise APIError(
        code="invalid_state",
        message="No active policy/ensemble found for auto evaluation.",
        status_code=409,
    )


def _resolve_bundle_id(session: Session, *, payload: dict[str, Any], active_candidate: dict[str, Any]) -> int:
    from_payload = _safe_int(payload.get("bundle_id"), 0)
    if from_payload > 0:
        return from_payload

    active_ensemble = active_candidate.get("ensemble")
    if isinstance(active_ensemble, PolicyEnsemble) and active_ensemble.id is not None:
        return int(active_ensemble.bundle_id)

    latest_run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
    if latest_run is not None and latest_run.bundle_id is not None:
        return int(latest_run.bundle_id)

    policy = active_candidate.get("policy")
    if not isinstance(policy, Policy):
        raise APIError(
            code="invalid_state",
            message="bundle_id required (payload, active run context, or active policy/ensemble).",
            status_code=409,
        )
    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    universe = definition.get("universe", {})
    if isinstance(universe, dict):
        from_policy = _safe_int(universe.get("bundle_id"), 0)
        if from_policy > 0:
            return from_policy

    raise APIError(
        code="invalid_state",
        message="bundle_id required (payload, active run context, or active policy universe).",
        status_code=409,
    )


def _resolve_timeframe(
    session: Session,
    payload: dict[str, Any],
    active_candidate: dict[str, Any],
) -> str:
    timeframe = str(payload.get("timeframe") or "").strip()
    if timeframe:
        return timeframe
    active_ensemble = active_candidate.get("ensemble")
    if isinstance(active_ensemble, PolicyEnsemble):
        for _member, policy in _load_ensemble_members(
            session=session,
            ensemble_id=int(active_ensemble.id or 0),
        ):
            definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
            timeframes = definition.get("timeframes", [])
            if isinstance(timeframes, list) and timeframes:
                first = str(timeframes[0]).strip()
                if first:
                    return first
        return "1d"
    policy = active_candidate.get("policy")
    if not isinstance(policy, Policy):
        return "1d"
    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    timeframes = definition.get("timeframes", [])
    if isinstance(timeframes, list) and timeframes:
        first = str(timeframes[0]).strip()
        if first:
            return first
    return "1d"


def _candidate_key(candidate: dict[str, Any]) -> tuple[str, int]:
    kind = str(candidate.get("entity_type") or ENTITY_POLICY).lower()
    identifier = _safe_int(candidate.get("id"), 0)
    return kind, max(0, identifier)


def _resolve_challengers(
    session: Session,
    *,
    payload: dict[str, Any],
    active_candidate: dict[str, Any],
    bundle_id: int,
) -> list[dict[str, Any]]:
    active_key = _candidate_key(active_candidate)
    policy_ids = [
        _safe_int(item, 0)
        for item in (payload.get("challenger_policy_ids") if isinstance(payload.get("challenger_policy_ids"), list) else [])
        if _safe_int(item, 0) > 0
    ]
    ensemble_ids = [
        _safe_int(item, 0)
        for item in (
            payload.get("challenger_ensemble_ids")
            if isinstance(payload.get("challenger_ensemble_ids"), list)
            else []
        )
        if _safe_int(item, 0) > 0
    ]

    candidates: list[dict[str, Any]] = []
    if policy_ids:
        for row in session.exec(
            select(Policy).where(Policy.id.in_(policy_ids)).order_by(Policy.created_at.desc())
        ).all():
            if row.id is None:
                continue
            candidate = {
                "entity_type": ENTITY_POLICY,
                "policy": row,
                "ensemble": None,
                "id": int(row.id),
                "name": row.name,
            }
            if _candidate_key(candidate) != active_key:
                candidates.append(candidate)
    if ensemble_ids:
        for row in session.exec(
            select(PolicyEnsemble)
            .where(PolicyEnsemble.id.in_(ensemble_ids))
            .order_by(PolicyEnsemble.created_at.desc())
        ).all():
            if row.id is None:
                continue
            candidate = {
                "entity_type": ENTITY_ENSEMBLE,
                "policy": None,
                "ensemble": row,
                "id": int(row.id),
                "name": row.name,
            }
            if _candidate_key(candidate) != active_key:
                candidates.append(candidate)
    if candidates:
        return candidates

    for row in session.exec(select(Policy).order_by(Policy.created_at.desc())).all():
        if row.id is None:
            continue
        candidate = {
            "entity_type": ENTITY_POLICY,
            "policy": row,
            "ensemble": None,
            "id": int(row.id),
            "name": row.name,
        }
        if _candidate_key(candidate) != active_key:
            candidates.append(candidate)
    for row in session.exec(
        select(PolicyEnsemble)
        .where(PolicyEnsemble.bundle_id == int(bundle_id))
        .order_by(PolicyEnsemble.created_at.desc())
    ).all():
        if row.id is None:
            continue
        candidate = {
            "entity_type": ENTITY_ENSEMBLE,
            "policy": None,
            "ensemble": row,
            "id": int(row.id),
            "name": row.name,
        }
        if _candidate_key(candidate) != active_key:
            candidates.append(candidate)
    return candidates


def _weighted_avg(values: list[tuple[float, float]]) -> float:
    total_weight = sum(max(0.0, weight) for _, weight in values)
    if total_weight <= 0:
        return 0.0
    return float(sum(value * max(0.0, weight) for value, weight in values) / total_weight)


def _combine_ensemble_metrics(member_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not member_rows:
        return {
            "period_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "cvar_95": 0.0,
            "turnover": 0.0,
            "cost_ratio": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "exposure_pct": 0.0,
            "score": -1.0,
            "trade_count": 0,
        }

    def collect(metric: str) -> list[tuple[float, float]]:
        values: list[tuple[float, float]] = []
        for row in member_rows:
            metrics = row.get("metrics", {})
            weight = _safe_float(row.get("weight"), 0.0)
            if not isinstance(metrics, dict) or weight <= 0:
                continue
            values.append((_safe_float(metrics.get(metric), 0.0), weight))
        return values

    trade_count = int(
        round(
            _weighted_avg(
                [(_safe_float(row.get("trade_count"), 0.0), _safe_float(row.get("weight"), 0.0)) for row in member_rows]
            )
        )
    )
    metrics = {
        "period_return": _weighted_avg(collect("period_return")),
        "max_drawdown": _weighted_avg(collect("max_drawdown")),
        "calmar": _weighted_avg(collect("calmar")),
        "cvar_95": _weighted_avg(collect("cvar_95")),
        "turnover": _weighted_avg(collect("turnover")),
        "cost_ratio": _weighted_avg(collect("cost_ratio")),
        "win_rate": _weighted_avg(collect("win_rate")),
        "profit_factor": _weighted_avg(collect("profit_factor")),
        "exposure_pct": _weighted_avg(collect("exposure_pct")),
        "trade_count": max(0, trade_count),
    }
    metrics["score"] = _metrics_score(metrics)
    return metrics


def _simulate_candidate(
    *,
    session: Session,
    store: DataStore,
    settings: Settings,
    candidate: dict[str, Any],
    bundle_id: int,
    start_date: date,
    end_date: date,
    timeframe: str,
    seed: int,
) -> dict[str, Any]:
    entity_type = str(candidate.get("entity_type") or ENTITY_POLICY).lower()
    if entity_type == ENTITY_POLICY:
        policy = candidate.get("policy")
        if not isinstance(policy, Policy):
            raise APIError(code="invalid_state", message="Policy candidate is invalid.")
        summary = simulate_policy_on_bundle(
            session=session,
            store=store,
            settings=settings,
            policy=policy,
            bundle_id=bundle_id,
            start_date=start_date,
            end_date=end_date,
            regime=None,
            seed=seed,
        )
        return {
            "entity_type": ENTITY_POLICY,
            "entity_id": int(policy.id or 0),
            "entity_name": policy.name,
            "policy_id": int(policy.id or 0),
            "policy_name": policy.name,
            "metrics": summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {},
            "trade_count": int(
                sum(
                    _safe_int(row.get("trade_count"), 0)
                    for row in (summary.get("symbol_rows") or [])
                    if isinstance(row, dict)
                )
            ),
            "score": _safe_float((summary.get("metrics") or {}).get("score"), _metrics_score(summary.get("metrics", {}))),
            "engine_version": summary.get("engine_version"),
            "data_digest": summary.get("data_digest"),
            "summary": summary,
            "member_contributions": [],
        }

    ensemble = candidate.get("ensemble")
    if not isinstance(ensemble, PolicyEnsemble):
        raise APIError(code="invalid_state", message="Ensemble candidate is invalid.")
    member_pairs = _normalize_member_weights(
        _load_ensemble_members(session, ensemble_id=int(ensemble.id or 0))
    )
    member_rows: list[dict[str, Any]] = []
    for idx, (_member, policy, weight) in enumerate(member_pairs):
        summary = simulate_policy_on_bundle(
            session=session,
            store=store,
            settings=settings,
            policy=policy,
            bundle_id=bundle_id,
            start_date=start_date,
            end_date=end_date,
            regime=None,
            seed=seed + (idx * 37) + int(policy.id or 0),
            max_symbols=max(1, _safe_int(settings.fast_mode_max_symbols_scan, 10))
            if settings.fast_mode_enabled
            else None,
        )
        metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
        trade_count = int(
            sum(
                _safe_int(row.get("trade_count"), 0)
                for row in (summary.get("symbol_rows") or [])
                if isinstance(row, dict)
            )
        )
        member_rows.append(
            {
                "policy_id": int(policy.id or 0),
                "policy_name": policy.name,
                "weight": float(weight),
                "metrics": metrics,
                "trade_count": trade_count,
                "score": _safe_float(metrics.get("score"), _metrics_score(metrics)),
                "engine_version": summary.get("engine_version"),
                "data_digest": summary.get("data_digest"),
            }
        )

    combined_metrics = _combine_ensemble_metrics(member_rows)
    digest_payload = {
        "ensemble_id": int(ensemble.id or 0),
        "bundle_id": int(bundle_id),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timeframe": timeframe,
        "seed": int(seed),
        "members": [
            {
                "policy_id": int(row.get("policy_id") or 0),
                "weight": float(row.get("weight") or 0.0),
                "data_digest": str(row.get("data_digest") or ""),
            }
            for row in member_rows
        ],
    }
    combined_data_digest = _digest(digest_payload)
    engine_versions = sorted(
        {
            str(row.get("engine_version"))
            for row in member_rows
            if isinstance(row.get("engine_version"), str) and str(row.get("engine_version")).strip()
        }
    )
    return {
        "entity_type": ENTITY_ENSEMBLE,
        "entity_id": int(ensemble.id or 0),
        "entity_name": ensemble.name,
        "ensemble_id": int(ensemble.id or 0),
        "ensemble_name": ensemble.name,
        "metrics": combined_metrics,
        "trade_count": int(combined_metrics.get("trade_count", 0)),
        "score": _safe_float(combined_metrics.get("score"), _metrics_score(combined_metrics)),
        "engine_version": ",".join(engine_versions) if engine_versions else "atlas-sim-v1.8",
        "data_digest": combined_data_digest,
        "summary": {
            "entity_type": ENTITY_ENSEMBLE,
            "ensemble_id": int(ensemble.id or 0),
            "ensemble_name": ensemble.name,
            "bundle_id": int(bundle_id),
            "window_start": start_date.isoformat(),
            "window_end": end_date.isoformat(),
            "timeframe": timeframe,
            "seed": int(seed),
            "member_contributions": member_rows,
            "metrics": combined_metrics,
            "engine_version": ",".join(engine_versions) if engine_versions else "atlas-sim-v1.8",
            "data_digest": combined_data_digest,
            "digest": _digest(
                {
                    "ensemble_id": int(ensemble.id or 0),
                    "seed": int(seed),
                    "metrics": combined_metrics,
                    "members": member_rows,
                }
            ),
        },
        "member_contributions": member_rows,
    }


def _candidate_runs_in_window(
    runs: list[PaperRun],
    *,
    candidate: dict[str, Any],
) -> list[PaperRun]:
    entity_type = str(candidate.get("entity_type") or ENTITY_POLICY).lower()
    if entity_type == ENTITY_POLICY:
        candidate_policy_id = _safe_int(candidate.get("id"), 0)
        return [
            row
            for row in runs
            if row.policy_id is not None and int(row.policy_id) == candidate_policy_id
        ]
    candidate_ensemble_id = _safe_int(candidate.get("id"), 0)
    filtered: list[PaperRun] = []
    for row in runs:
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        ensemble_id = _safe_int(summary.get("ensemble_id"), 0)
        if ensemble_id > 0 and ensemble_id == candidate_ensemble_id:
            filtered.append(row)
    return filtered


def _no_trade_frequency(runs: list[PaperRun]) -> tuple[float, bool]:
    if not runs:
        return 0.0, False
    triggered = 0
    latest_triggered = False
    for index, row in enumerate(runs):
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        flag = bool(summary.get("no_trade_triggered", False))
        if flag:
            triggered += 1
        if index == len(runs) - 1:
            latest_triggered = flag
    return float(triggered / max(1, len(runs))), latest_triggered


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
    recent_policy_switches = list(
        session.exec(
            select(PolicySwitchEvent)
            .where(PolicySwitchEvent.ts >= since_30d)
            .order_by(PolicySwitchEvent.ts.desc())
        ).all()
    )
    recent_auto_switches = list(
        session.exec(
            select(AutoEvalRun)
            .where(AutoEvalRun.ts >= since_30d)
            .where(AutoEvalRun.auto_switch_applied == True)  # noqa: E712
            .order_by(AutoEvalRun.ts.desc())
        ).all()
    )
    recent_switches_30d = len(recent_policy_switches) + len(recent_auto_switches)
    latest_policy_ts = recent_policy_switches[0].ts if recent_policy_switches else None
    latest_auto_ts = recent_auto_switches[0].ts if recent_auto_switches else None
    last_switch_ts = latest_policy_ts
    if latest_auto_ts is not None and (last_switch_ts is None or latest_auto_ts > last_switch_ts):
        last_switch_ts = latest_auto_ts

    trading_days_since_last = 99_999
    if last_switch_ts is not None:
        last_day = last_switch_ts.astimezone(timezone.utc).date()
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
        "recent_switches_30d": recent_switches_30d,
        "recent_policy_switches_30d": len(recent_policy_switches),
        "recent_auto_switches_30d": len(recent_auto_switches),
        "trading_days_since_last_switch": trading_days_since_last,
        "cooldown_blocked": trading_days_since_last < cooldown_trading_days,
        "max_switches_blocked": recent_switches_30d >= max_switches,
    }


def _apply_switch_to_policy(
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


def _apply_switch_to_ensemble(
    session: Session,
    *,
    settings: Settings,
    to_ensemble: PolicyEnsemble,
) -> None:
    state = get_or_create_paper_state(session, settings)
    settings_map = dict(state.settings_json or {})
    settings_map["paper_mode"] = "policy"
    settings_map["active_policy_id"] = None
    settings_map["active_policy_name"] = None
    settings_map["active_ensemble_id"] = int(to_ensemble.id or 0)
    settings_map["active_ensemble_name"] = to_ensemble.name
    state.settings_json = settings_map
    session.add(state)


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

    active_candidate = _resolve_active_candidate(session, payload, settings_map)
    active_entity_type = str(active_candidate.get("entity_type") or ENTITY_POLICY)
    active_policy = active_candidate.get("policy")
    active_ensemble = active_candidate.get("ensemble")
    if not isinstance(active_policy, Policy):
        active_policy = None
    if not isinstance(active_ensemble, PolicyEnsemble):
        active_ensemble = None

    bundle_id = _resolve_bundle_id(session, payload=payload, active_candidate=active_candidate)
    timeframe = _resolve_timeframe(session, payload, active_candidate)

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

    window_start_dt = datetime.combine(window_start, datetime.min.time(), tzinfo=timezone.utc)
    window_end_dt = datetime.combine(asof_day, datetime.max.time(), tzinfo=timezone.utc)
    all_window_runs = list(
        session.exec(
            select(PaperRun)
            .where(PaperRun.bundle_id == int(bundle_id))
            .where(PaperRun.asof_ts >= window_start_dt)
            .where(PaperRun.asof_ts <= window_end_dt)
            .order_by(PaperRun.asof_ts.asc())
        ).all()
    )
    active_runs = _candidate_runs_in_window(all_window_runs, candidate=active_candidate)
    active_metrics = compute_health_metrics(active_runs, window_days=lookback_days)
    active_no_trade_frequency, latest_active_no_trade = _no_trade_frequency(active_runs)
    no_trade_penalty_weight = 0.20
    active_score = _metrics_score(active_metrics) - (active_no_trade_frequency * no_trade_penalty_weight)
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

    short_window = max(
        5,
        _safe_int(settings_map.get("health_window_days_short"), settings.health_window_days_short),
    )
    long_window = max(
        short_window,
        _safe_int(settings_map.get("health_window_days_long"), settings.health_window_days_long),
    )
    health_short_status = HEALTHY
    health_long_status = HEALTHY
    health_reasons: list[str] = []
    degradations: list[str] = []
    if active_policy is not None:
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
        health_short_status = health_short.status
        health_long_status = health_long.status
        health_reasons = list({*health_short.reasons_json, *health_long.reasons_json})
        if health_short.status == DEGRADED or health_long.status == DEGRADED:
            degradations.append("Active policy health is DEGRADED.")
        baseline = (
            active_policy.definition_json.get("baseline", {})
            if isinstance(active_policy.definition_json, dict)
            else {}
        )
        baseline_tail = abs(_safe_float((baseline or {}).get("cvar_95"), 0.0))
        current_tail = abs(_safe_float(active_metrics.get("cvar_95"), 0.0))
        if baseline_tail > 0 and current_tail > baseline_tail * 1.5:
            degradations.append("Tail loss exceeded baseline guardrail.")
    elif active_ensemble is not None:
        member_pairs = _normalize_member_weights(
            _load_ensemble_members(session, ensemble_id=int(active_ensemble.id or 0))
        )
        if not member_pairs:
            degradations.append("Active ensemble has no enabled members.")
        for _member, policy, _weight in member_pairs:
            short = get_policy_health_snapshot(
                session,
                settings=settings,
                policy=policy,
                window_days=short_window,
                asof_date=asof_day,
                refresh=True,
                overrides=settings_map,
            )
            long = get_policy_health_snapshot(
                session,
                settings=settings,
                policy=policy,
                window_days=long_window,
                asof_date=asof_day,
                refresh=True,
                overrides=settings_map,
            )
            if short.status == DEGRADED:
                health_short_status = DEGRADED
            elif short.status == WARNING and health_short_status != DEGRADED:
                health_short_status = WARNING
            if long.status == DEGRADED:
                health_long_status = DEGRADED
            elif long.status == WARNING and health_long_status != DEGRADED:
                health_long_status = WARNING
            if short.status == DEGRADED or long.status == DEGRADED:
                degradations.append(f"Member policy {policy.name} health is DEGRADED.")
            if short.reasons_json:
                health_reasons.extend([f"{policy.name}: {reason}" for reason in short.reasons_json[:2]])
            if long.reasons_json:
                health_reasons.extend([f"{policy.name}: {reason}" for reason in long.reasons_json[:2]])
        health_reasons = list(dict.fromkeys(health_reasons))[:8]

    challengers = _resolve_challengers(
        session,
        payload=payload,
        active_candidate=active_candidate,
        bundle_id=bundle_id,
    )
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
        summary = _simulate_candidate(
            session=session,
            store=store,
            settings=settings,
            candidate=challenger,
            bundle_id=bundle_id,
            start_date=window_start,
            end_date=asof_day,
            timeframe=timeframe,
            seed=seed + idx,
        )
        metrics = summary.get("metrics", {}) if isinstance(summary.get("metrics"), dict) else {}
        trade_count = int(summary.get("trade_count", 0))
        candidate_no_trade_frequency = _safe_float(summary.get("no_trade_frequency"), 0.0)
        candidate_score = _safe_float(summary.get("score"), _metrics_score(metrics)) - (
            candidate_no_trade_frequency * no_trade_penalty_weight
        )
        max_dd = abs(_safe_float(metrics.get("max_drawdown"), 0.0))
        dd_limit = active_max_dd_abs * max_dd_multiplier if active_max_dd_abs > 0 else settings.kill_switch_drawdown
        dd_pass = max_dd <= dd_limit
        score_pass = candidate_score >= active_score + score_margin
        min_trades_pass = trade_count >= min_trades
        accepted = bool(dd_pass and score_pass and min_trades_pass and allow_switch_candidates)
        item_reasons: list[str] = []
        if not allow_switch_candidates:
            item_reasons.append("Active strategy sample is too small for switching.")
        if not dd_pass:
            item_reasons.append("Rejected: challenger drawdown exceeds threshold.")
        if not score_pass:
            item_reasons.append("Rejected: score improvement below required margin.")
        if not min_trades_pass:
            item_reasons.append("Rejected: challenger trade count below minimum.")
        challenger_rows.append(
            {
                "entity_type": str(summary.get("entity_type", ENTITY_POLICY)),
                "entity_id": int(summary.get("entity_id", 0)),
                "entity_name": str(summary.get("entity_name", f"Candidate {idx}")),
                "policy_id": summary.get("policy_id"),
                "ensemble_id": summary.get("ensemble_id"),
                "metrics": metrics,
                "trade_count": trade_count,
                "score": candidate_score,
                "no_trade_frequency": candidate_no_trade_frequency,
                "passes": accepted,
                "reasons": item_reasons,
                "engine_version": summary.get("engine_version"),
                "data_digest": summary.get("data_digest"),
                "member_contributions": summary.get("member_contributions", []),
            }
        )
        if progress_cb:
            progress_cb(15 + int((idx / total) * 55), f"Auto-eval challenger {idx}/{total}")

    challenger_rows.sort(
        key=lambda row: (
            -_safe_float(row.get("score"), -1e9),
            str(row.get("entity_type", ENTITY_POLICY)),
            _safe_int(row.get("entity_id"), 0),
        )
    )
    best = challenger_rows[0] if challenger_rows else None
    recommended_action = ACTION_KEEP
    recommended_entity_type: str | None = None
    recommended_policy_id: int | None = None
    recommended_ensemble_id: int | None = None
    reasons: list[str] = []

    if active_trade_count < min_trades:
        reasons.append("KEEP: active configuration has insufficient recent trades for switching.")
    if active_run_count < max(5, lookback_days // 2):
        reasons.append("KEEP: insufficient active sample window.")
    if degradations:
        recommended_action = ACTION_SHADOW_ONLY
        reasons.extend([f"SHADOW_ONLY: {msg}" for msg in degradations])
    elif best is not None and bool(best.get("passes")):
        recommended_action = ACTION_SWITCH
        recommended_entity_type = str(best.get("entity_type", ENTITY_POLICY))
        if recommended_entity_type == ENTITY_ENSEMBLE:
            recommended_ensemble_id = _safe_int(best.get("entity_id"), 0) or None
            reasons.append(
                "SWITCH: challenger ensemble improved robustness score within drawdown limits."
            )
        else:
            recommended_policy_id = _safe_int(best.get("entity_id"), 0) or None
            reasons.append("SWITCH: challenger policy improved score with acceptable drawdown.")
    else:
        reasons.append("KEEP: no challenger passed all deterministic gates.")
    if latest_active_no_trade and recommended_action == ACTION_SWITCH:
        recommended_action = ACTION_KEEP
        recommended_entity_type = None
        recommended_policy_id = None
        recommended_ensemble_id = None
        reasons.append("KEEP: latest run was no-trade; switch deferred until an executable day.")

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
    switch_target_id = (
        recommended_policy_id if recommended_entity_type == ENTITY_POLICY else recommended_ensemble_id
    )
    can_auto_switch = (
        auto_switch_enabled
        and recommended_action == ACTION_SWITCH
        and switch_target_id is not None
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
            "entity_type": active_entity_type,
            "entity_id": int(active_candidate.get("id") or 0),
            "entity_name": str(active_candidate.get("name") or "Active"),
            "policy_id": int(active_policy.id) if active_policy is not None and active_policy.id is not None else None,
            "ensemble_id": int(active_ensemble.id) if active_ensemble is not None and active_ensemble.id is not None else None,
            "score": active_score,
            "metrics": active_metrics,
            "trade_count": active_trade_count,
            "run_count": active_run_count,
            "no_trade_frequency": active_no_trade_frequency,
            "latest_no_trade": latest_active_no_trade,
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
            "health_short_status": health_short_status,
            "health_long_status": health_long_status,
            "health_reasons": health_reasons,
        },
    }
    digest = _digest(
        {
            "asof_date": asof_day.isoformat(),
            "bundle_id": bundle_id,
            "active_entity_type": active_entity_type,
            "active_entity_id": int(active_candidate.get("id") or 0),
            "recommended_action": recommended_action,
            "recommended_entity_type": recommended_entity_type,
            "recommended_policy_id": recommended_policy_id,
            "recommended_ensemble_id": recommended_ensemble_id,
            "score_table": score_table,
            "seed": seed,
        }
    )

    auto_eval = AutoEvalRun(
        bundle_id=bundle_id,
        active_policy_id=int(active_policy.id) if active_policy is not None and active_policy.id is not None else None,
        active_ensemble_id=(
            int(active_ensemble.id) if active_ensemble is not None and active_ensemble.id is not None else None
        ),
        recommended_action=recommended_action,
        recommended_policy_id=recommended_policy_id,
        recommended_ensemble_id=recommended_ensemble_id,
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
            "active_entity_type": active_entity_type,
            "recommended_entity_type": recommended_entity_type,
        },
    )
    session.add(auto_eval)
    session.commit()
    session.refresh(auto_eval)

    switched_to_policy_id: int | None = None
    switched_to_ensemble_id: int | None = None
    if can_auto_switch and switch_target_id is not None:
        if recommended_entity_type == ENTITY_POLICY and recommended_policy_id is not None:
            to_policy = session.get(Policy, recommended_policy_id)
            if to_policy is not None and to_policy.id is not None and active_policy is not None:
                _apply_switch_to_policy(
                    session,
                    settings=settings,
                    from_policy=active_policy,
                    to_policy=to_policy,
                    auto_eval_id=int(auto_eval.id),
                    reason="auto_eval_switch_policy",
                    cooldown_state=switch_gates,
                    mode="AUTO",
                )
                switched_to_policy_id = int(to_policy.id)
        elif recommended_entity_type == ENTITY_ENSEMBLE and recommended_ensemble_id is not None:
            to_ensemble = session.get(PolicyEnsemble, recommended_ensemble_id)
            if to_ensemble is not None and to_ensemble.id is not None:
                _apply_switch_to_ensemble(
                    session,
                    settings=settings,
                    to_ensemble=to_ensemble,
                )
                switched_to_ensemble_id = int(to_ensemble.id)
        auto_eval.auto_switch_applied = bool(
            switched_to_policy_id is not None or switched_to_ensemble_id is not None
        )
        session.add(auto_eval)
        if auto_eval.auto_switch_applied:
            reasons.append(
                "Auto-switch applied: "
                + (
                    f"policy -> {switched_to_policy_id}"
                    if switched_to_policy_id is not None
                    else f"ensemble -> {switched_to_ensemble_id}"
                )
                + "."
            )
        session.commit()

    emit_operate_event(
        session,
        severity="INFO",
        category="POLICY",
        message="auto_eval_completed",
        details={
            "auto_eval_id": int(auto_eval.id),
            "bundle_id": bundle_id,
            "active_entity_type": active_entity_type,
            "active_policy_id": int(active_policy.id) if active_policy is not None and active_policy.id is not None else None,
            "active_ensemble_id": int(active_ensemble.id) if active_ensemble is not None and active_ensemble.id is not None else None,
            "recommended_action": recommended_action,
            "recommended_entity_type": recommended_entity_type,
            "recommended_policy_id": recommended_policy_id,
            "recommended_ensemble_id": recommended_ensemble_id,
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
            "recommended_entity_type": recommended_entity_type,
            "recommended_policy_id": recommended_policy_id,
            "recommended_ensemble_id": recommended_ensemble_id,
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
        "recommended_entity_type": recommended_entity_type,
        "recommended_policy_id": recommended_policy_id,
        "recommended_ensemble_id": recommended_ensemble_id,
        "auto_switch_applied": bool(auto_eval.auto_switch_applied),
        "switched_to_policy_id": switched_to_policy_id,
        "switched_to_ensemble_id": switched_to_ensemble_id,
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
