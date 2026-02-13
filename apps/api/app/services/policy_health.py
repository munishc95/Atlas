from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import PaperRun, Policy, PolicyHealthSnapshot
from app.services.operate_events import emit_operate_event


HEALTHY = "HEALTHY"
WARNING = "WARNING"
DEGRADED = "DEGRADED"
PAUSED = "PAUSED"
RETIRED = "RETIRED"


def _coalesce_date(value: date | None) -> date:
    return value or datetime.now(timezone.utc).date()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cfg_float(
    settings: Settings,
    overrides: dict[str, Any] | None,
    key: str,
    fallback: float,
) -> float:
    if isinstance(overrides, dict):
        value = overrides.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(getattr(settings, key, fallback))


def _cfg_str(
    settings: Settings,
    overrides: dict[str, Any] | None,
    key: str,
    fallback: str,
) -> str:
    if isinstance(overrides, dict):
        value = overrides.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raw = getattr(settings, key, fallback)
    return str(raw)


def policy_status(policy: Policy) -> str:
    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    raw = str(definition.get("status", "ACTIVE")).upper()
    return raw


def _baseline_metrics(policy: Policy) -> dict[str, float]:
    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    baseline = definition.get("baseline", {})
    if not isinstance(baseline, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in baseline.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _window_runs(
    session: Session,
    *,
    policy_id: int,
    asof_date: date,
    window_days: int,
) -> list[PaperRun]:
    start_date = asof_date - timedelta(days=max(0, window_days - 1))
    rows = session.exec(
        select(PaperRun)
        .where(PaperRun.policy_id == policy_id)
        .where(PaperRun.asof_ts >= datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc))
        .where(PaperRun.asof_ts <= datetime.combine(asof_date, datetime.max.time(), tzinfo=timezone.utc))
        .order_by(PaperRun.asof_ts.asc())
    ).all()
    return list(rows)


def _cvar_95(returns: list[float]) -> float:
    if not returns:
        return 0.0
    arr = np.array(returns, dtype=float)
    threshold = np.quantile(arr, 0.05)
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return 0.0
    return float(np.mean(tail))


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak <= 0:
            continue
        dd = value / peak - 1.0
        max_dd = min(max_dd, dd)
    return float(max_dd)


def compute_health_metrics(runs: list[PaperRun], window_days: int) -> dict[str, float]:
    if not runs:
        return {
            "window_days": float(window_days),
            "run_count": 0.0,
            "period_return": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_hold": 0.0,
            "turnover": 0.0,
            "exposure": 0.0,
            "cost_ratio": 0.0,
            "cvar_95": 0.0,
            "gross_pnl": 0.0,
            "costs": 0.0,
            "trade_count": 0.0,
        }

    summaries = [row.summary_json if isinstance(row.summary_json, dict) else {} for row in runs]
    first_eq = _safe_float(summaries[0].get("equity_before"), 0.0)
    last_eq = _safe_float(summaries[-1].get("equity_after"), first_eq)
    period_return = (last_eq / first_eq - 1.0) if first_eq > 0 else 0.0
    equity_curve = [
        _safe_float(summary.get("equity_after"), _safe_float(summary.get("equity_before"), 0.0))
        for summary in summaries
    ]
    returns = [
        _safe_float(summary.get("net_pnl"), 0.0)
        / max(1e-9, _safe_float(summary.get("equity_before"), 1.0))
        for summary in summaries
    ]
    gross_pnl_values = [_safe_float(summary.get("gross_pnl"), 0.0) for summary in summaries]
    net_pnl_values = [_safe_float(summary.get("net_pnl"), 0.0) for summary in summaries]
    cost_values = [_safe_float(summary.get("total_cost"), 0.0) for summary in summaries]
    wins = [value for value in net_pnl_values if value > 0]
    losses = [abs(value) for value in net_pnl_values if value < 0]

    mean_ret = float(mean(returns)) if returns else 0.0
    std_ret = float(pstdev(returns)) if len(returns) > 1 else 0.0
    downside = [min(0.0, value) for value in returns]
    downside_std = float(np.std(downside)) if downside else 0.0
    sharpe = mean_ret / std_ret if std_ret > 1e-9 else 0.0
    sortino = mean_ret / downside_std if downside_std > 1e-9 else 0.0
    max_dd = _max_drawdown(equity_curve)
    calmar = period_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
    gross_sum = float(sum(max(0.0, value) for value in gross_pnl_values))
    cost_sum = float(sum(cost_values))
    trade_count = float(sum(_safe_float(summary.get("trade_count"), 0.0) for summary in summaries))

    return {
        "window_days": float(window_days),
        "run_count": float(len(runs)),
        "period_return": float(period_return),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(len(wins) / max(1, len(net_pnl_values))),
        "profit_factor": float(sum(wins) / max(1e-9, sum(losses))),
        "avg_hold": float(
            mean(
                [
                    _safe_float(summary.get("avg_holding_days"), 0.0)
                    for summary in summaries
                ]
            )
            if summaries
            else 0.0
        ),
        "turnover": float(
            mean([_safe_float(summary.get("turnover"), 0.0) for summary in summaries]) if summaries else 0.0
        ),
        "exposure": float(
            mean([_safe_float(summary.get("exposure"), 0.0) for summary in summaries]) if summaries else 0.0
        ),
        "cost_ratio": float(cost_sum / max(1e-9, gross_sum)),
        "cvar_95": float(_cvar_95(returns)),
        "gross_pnl": float(sum(gross_pnl_values)),
        "costs": float(cost_sum),
        "trade_count": trade_count,
    }


def evaluate_policy_health(
    *,
    metrics: dict[str, float],
    baseline: dict[str, float],
    settings: Settings,
    window_days: int,
    overrides: dict[str, Any] | None = None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    warnings: list[str] = []
    degraded: list[str] = []

    max_dd = abs(_safe_float(metrics.get("max_drawdown"), 0.0))
    baseline_maxdd = abs(_safe_float(baseline.get("max_drawdown"), 0.0))
    drift_maxdd_multiplier = _cfg_float(
        settings,
        overrides,
        "drift_maxdd_multiplier",
        settings.drift_maxdd_multiplier,
    )
    if baseline_maxdd > 0:
        maxdd_limit = baseline_maxdd * drift_maxdd_multiplier
        if max_dd > maxdd_limit:
            warnings.append(
                f"Rolling max drawdown {max_dd:.3f} exceeds baseline-adjusted limit {maxdd_limit:.3f}."
            )

    period_return = _safe_float(metrics.get("period_return"), 0.0)
    cost_ratio = _safe_float(metrics.get("cost_ratio"), 0.0)
    drift_negative_return_cost_ratio_threshold = _cfg_float(
        settings,
        overrides,
        "drift_negative_return_cost_ratio_threshold",
        settings.drift_negative_return_cost_ratio_threshold,
    )
    if (
        window_days >= 60
        and period_return < 0
        and cost_ratio > drift_negative_return_cost_ratio_threshold
    ):
        warnings.append(
            f"60d return is negative ({period_return:.3f}) with elevated cost ratio ({cost_ratio:.3f})."
        )

    baseline_win_rate = _safe_float(baseline.get("win_rate"), -1.0)
    drift_win_rate_drop_pct = _cfg_float(
        settings,
        overrides,
        "drift_win_rate_drop_pct",
        settings.drift_win_rate_drop_pct,
    )
    if baseline_win_rate >= 0:
        min_win_rate = baseline_win_rate - drift_win_rate_drop_pct
        current_win_rate = _safe_float(metrics.get("win_rate"), 0.0)
        if current_win_rate < min_win_rate:
            warnings.append(
                f"Win rate dropped to {current_win_rate:.3f}, below baseline floor {min_win_rate:.3f}."
            )

    kill_switch_drawdown = _cfg_float(
        settings,
        overrides,
        "kill_switch_dd",
        settings.kill_switch_drawdown,
    )
    if max_dd > kill_switch_drawdown:
        degraded.append(
            f"Max drawdown {max_dd:.3f} exceeds kill-switch threshold {kill_switch_drawdown:.3f}."
        )

    baseline_period_60 = _safe_float(
        baseline.get("period_return_60d", baseline.get("period_return", 0.0)),
        0.0,
    )
    drift_return_delta_threshold = _cfg_float(
        settings,
        overrides,
        "drift_return_delta_threshold",
        settings.drift_return_delta_threshold,
    )
    if window_days >= 60 and period_return < (baseline_period_60 - drift_return_delta_threshold):
        degraded.append(
            "60d return underperforms baseline by configured drift delta."
        )

    if degraded:
        reasons.extend(degraded)
        reasons.extend(warnings)
        return DEGRADED, reasons
    if warnings:
        reasons.extend(warnings)
        return WARNING, reasons
    return HEALTHY, ["Policy metrics are within configured drift bounds."]


def create_policy_health_snapshot(
    session: Session,
    *,
    settings: Settings,
    policy: Policy,
    asof_date: date | None,
    window_days: int,
    overrides: dict[str, Any] | None = None,
) -> PolicyHealthSnapshot:
    resolved_date = _coalesce_date(asof_date)
    runs = _window_runs(
        session,
        policy_id=int(policy.id),
        asof_date=resolved_date,
        window_days=max(1, int(window_days)),
    )
    metrics = compute_health_metrics(runs, window_days=max(1, int(window_days)))
    baseline = _baseline_metrics(policy)
    status, reasons = evaluate_policy_health(
        metrics=metrics,
        baseline=baseline,
        settings=settings,
        window_days=max(1, int(window_days)),
        overrides=overrides,
    )
    snapshot = PolicyHealthSnapshot(
        policy_id=int(policy.id),
        asof_date=resolved_date,
        window_days=max(1, int(window_days)),
        metrics_json=metrics,
        status=status,
        reasons_json=reasons,
    )
    session.add(snapshot)
    session.commit()
    session.refresh(snapshot)
    return snapshot


def get_policy_health_snapshot(
    session: Session,
    *,
    settings: Settings,
    policy: Policy,
    window_days: int,
    asof_date: date | None = None,
    refresh: bool = False,
    overrides: dict[str, Any] | None = None,
) -> PolicyHealthSnapshot:
    resolved_date = _coalesce_date(asof_date)
    if not refresh:
        existing = session.exec(
            select(PolicyHealthSnapshot)
            .where(PolicyHealthSnapshot.policy_id == int(policy.id))
            .where(PolicyHealthSnapshot.window_days == int(window_days))
            .where(PolicyHealthSnapshot.asof_date == resolved_date)
            .order_by(PolicyHealthSnapshot.id.desc())
        ).first()
        if existing is not None:
            return existing
    return create_policy_health_snapshot(
        session,
        settings=settings,
        policy=policy,
        asof_date=resolved_date,
        window_days=window_days,
        overrides=overrides,
    )


def latest_policy_health_snapshots(session: Session) -> list[PolicyHealthSnapshot]:
    rows = session.exec(
        select(PolicyHealthSnapshot).order_by(PolicyHealthSnapshot.asof_date.desc(), PolicyHealthSnapshot.id.desc())
    ).all()
    latest: dict[tuple[int, int], PolicyHealthSnapshot] = {}
    for row in rows:
        key = (int(row.policy_id), int(row.window_days))
        if key not in latest:
            latest[key] = row
    return list(latest.values())


def latest_policy_health_snapshot_for_policy(
    session: Session,
    *,
    policy_id: int,
    window_days: int | None = None,
    asof_date: date | None = None,
) -> PolicyHealthSnapshot | None:
    stmt = (
        select(PolicyHealthSnapshot)
        .where(PolicyHealthSnapshot.policy_id == policy_id)
        .order_by(PolicyHealthSnapshot.asof_date.desc(), PolicyHealthSnapshot.id.desc())
    )
    if window_days is not None:
        stmt = stmt.where(PolicyHealthSnapshot.window_days == int(window_days))
    if asof_date is not None:
        stmt = stmt.where(PolicyHealthSnapshot.asof_date <= asof_date)
    return session.exec(stmt).first()


def apply_policy_health_actions(
    session: Session,
    *,
    settings: Settings,
    policy: Policy,
    snapshot: PolicyHealthSnapshot,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    definition = dict(policy.definition_json or {})
    result: dict[str, Any] = {
        "policy_id": policy.id,
        "snapshot_status": snapshot.status,
        "risk_scale_override": 1.0,
        "policy_status": str(definition.get("status", "ACTIVE")).upper(),
        "action": "NONE",
    }
    if snapshot.status == WARNING:
        result["risk_scale_override"] = _cfg_float(
            settings,
            overrides,
            "drift_warning_risk_scale",
            settings.drift_warning_risk_scale,
        )
        result["action"] = "RISK_SCALE_WARNING"
        emit_operate_event(
            session,
            severity="WARN",
            category="POLICY",
            message="Policy health warning triggered risk scaling.",
            details={
                "policy_id": int(policy.id),
                "snapshot_id": int(snapshot.id) if snapshot.id is not None else None,
                "window_days": int(snapshot.window_days),
                "status": snapshot.status,
                "action": result["action"],
                "risk_scale_override": result["risk_scale_override"],
                "reasons": list(snapshot.reasons_json or []),
            },
        )
        return result

    if snapshot.status != DEGRADED:
        return result

    result["risk_scale_override"] = _cfg_float(
        settings,
        overrides,
        "drift_degraded_risk_scale",
        settings.drift_degraded_risk_scale,
    )
    action = _cfg_str(settings, overrides, "drift_degraded_action", settings.drift_degraded_action).upper()
    if action == "RETIRED":
        definition["status"] = RETIRED
    elif action == "PAUSE":
        definition["status"] = PAUSED
    else:
        definition["status"] = str(definition.get("status", "ACTIVE")).upper()
    policy.definition_json = definition
    session.add(policy)
    session.commit()
    session.refresh(policy)
    result["policy_status"] = str(policy.definition_json.get("status", "ACTIVE")).upper()
    result["action"] = f"DEGRADED_{action}"
    emit_operate_event(
        session,
        severity="ERROR" if action in {"PAUSE", "RETIRED"} else "WARN",
        category="POLICY",
        message="Policy health degraded action applied.",
        details={
            "policy_id": int(policy.id),
            "snapshot_id": int(snapshot.id) if snapshot.id is not None else None,
            "window_days": int(snapshot.window_days),
            "status": snapshot.status,
            "action": result["action"],
            "policy_status": result["policy_status"],
            "risk_scale_override": result["risk_scale_override"],
            "reasons": list(snapshot.reasons_json or []),
        },
    )
    return result


def select_fallback_policy(
    session: Session,
    *,
    current_policy_id: int,
    regime: str,
) -> Policy | None:
    policies = session.exec(select(Policy).order_by(Policy.created_at.desc())).all()
    ranked_regime: list[tuple[float, Policy]] = []
    ranked_risk_off: list[tuple[float, Policy]] = []
    for policy in policies:
        if policy.id == current_policy_id:
            continue
        definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
        status = str(definition.get("status", "ACTIVE")).upper()
        if status in {PAUSED, RETIRED}:
            continue
        regime_map = definition.get("regime_map", {})
        if not isinstance(regime_map, dict):
            continue
        baseline = definition.get("baseline", {})
        score = _safe_float(
            baseline.get("oos_score", baseline.get("calmar", 0.0)) if isinstance(baseline, dict) else 0.0
        )
        regime_cfg = regime_map.get(regime)
        if isinstance(regime_cfg, dict):
            strategy_key = regime_cfg.get("strategy_key")
            allowed = regime_cfg.get("allowed_templates")
            if strategy_key or allowed:
                ranked_regime.append((score, policy))
                continue
        risk_off_cfg = regime_map.get("RISK_OFF")
        if isinstance(risk_off_cfg, dict):
            strategy_key = risk_off_cfg.get("strategy_key")
            allowed = risk_off_cfg.get("allowed_templates")
            if strategy_key or allowed:
                ranked_risk_off.append((score, policy))

    ranked_regime.sort(key=lambda item: item[0], reverse=True)
    if ranked_regime:
        return ranked_regime[0][1]
    ranked_risk_off.sort(key=lambda item: item[0], reverse=True)
    return ranked_risk_off[0][1] if ranked_risk_off else None
