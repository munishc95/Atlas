from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session

from app.core.config import Settings
from app.db.models import PaperState
from app.services.operate_events import get_operate_health_summary

STATUS_PASS = "PASS"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"


def _check(
    *,
    check_id: str,
    category: str,
    label: str,
    status: str,
    detail: str,
    action: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "category": category,
        "label": label,
        "status": status,
        "detail": detail,
        "action": action,
        "metadata": dict(metadata or {}),
    }


def _setting_bool(state_settings: dict[str, Any], settings: Settings, key: str) -> bool:
    return bool(state_settings.get(key, getattr(settings, key)))


def _setting_float(
    state_settings: dict[str, Any],
    settings: Settings,
    key: str,
    fallback_key: str | None = None,
) -> float:
    raw = state_settings.get(key, getattr(settings, fallback_key or key))
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(getattr(settings, fallback_key or key))


def _setting_int(
    state_settings: dict[str, Any],
    settings: Settings,
    key: str,
    fallback_key: str | None = None,
) -> int:
    raw = state_settings.get(key, getattr(settings, fallback_key or key))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(getattr(settings, fallback_key or key))


def _setting_str(state_settings: dict[str, Any], settings: Settings, key: str) -> str:
    return str(state_settings.get(key, getattr(settings, key))).strip()


def _worst_status(checks: list[dict[str, Any]]) -> str:
    statuses = {str(item.get("status", STATUS_PASS)).upper() for item in checks}
    if STATUS_FAIL in statuses:
        return "BLOCKED"
    if STATUS_WARN in statuses:
        return "ATTENTION"
    return "READY"


def _status_counts(checks: list[dict[str, Any]]) -> dict[str, int]:
    counts = {STATUS_PASS: 0, STATUS_WARN: 0, STATUS_FAIL: 0}
    for item in checks:
        token = str(item.get("status", STATUS_PASS)).upper()
        counts[token] = counts.get(token, 0) + 1
    return counts


def evaluate_operate_readiness(
    session: Session,
    settings: Settings,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = None,
) -> dict[str, Any]:
    """Evaluate whether Atlas is safe to run in production paper/shadow mode.

    This is an operator gate, not a live-money permission switch. It composes
    existing health, paper-state, data-quality, confidence, and scheduler signals
    into a single explainable verdict.
    """

    state = session.get(PaperState, 1)
    state_settings = dict(state.settings_json or {}) if state is not None else {}
    health = get_operate_health_summary(
        session,
        settings,
        bundle_id=bundle_id,
        timeframe=timeframe,
    )
    active_bundle_id = health.get("active_bundle_id")
    active_timeframe = str(health.get("active_timeframe") or timeframe or "1d")
    checks: list[dict[str, Any]] = []

    if isinstance(active_bundle_id, int) and active_bundle_id > 0:
        checks.append(
            _check(
                check_id="active_bundle",
                category="CONFIG",
                label="Active universe bundle",
                status=STATUS_PASS,
                detail=f"Bundle {active_bundle_id} is selected for operate flows.",
            )
        )
    else:
        checks.append(
            _check(
                check_id="active_bundle",
                category="CONFIG",
                label="Active universe bundle",
                status=STATUS_FAIL,
                detail="No active dataset bundle is resolved.",
                action="Select or import a universe bundle before production paper runs.",
            )
        )

    paper_mode = str(state_settings.get("paper_mode", "strategy")).strip().lower() or "strategy"
    active_policy_id = state_settings.get("active_policy_id")
    active_ensemble_id = state_settings.get("active_ensemble_id")
    if paper_mode == "policy" and active_policy_id is None and active_ensemble_id is None:
        checks.append(
            _check(
                check_id="policy_selection",
                category="CONFIG",
                label="Paper policy selection",
                status=STATUS_FAIL,
                detail="Policy mode is enabled but no policy or ensemble is active.",
                action="Activate a validated policy or switch paper mode back to strategy.",
            )
        )
    elif paper_mode == "policy":
        checks.append(
            _check(
                check_id="policy_selection",
                category="CONFIG",
                label="Paper policy selection",
                status=STATUS_PASS,
                detail=(
                    f"Policy mode has active policy {active_policy_id} "
                    f"and ensemble {active_ensemble_id}."
                ),
            )
        )
    else:
        checks.append(
            _check(
                check_id="policy_selection",
                category="CONFIG",
                label="Paper policy selection",
                status=STATUS_WARN,
                detail="Strategy mode is active; production paper can run, but it is not tied to a promoted policy.",
                action="Promote a walk-forward validated policy when you want governed paper operation.",
            )
        )

    if state is None:
        checks.append(
            _check(
                check_id="paper_state",
                category="SAFETY",
                label="Paper account state",
                status=STATUS_FAIL,
                detail="Paper account state has not been initialized.",
                action="Open Paper Trading or call /api/paper/state once to initialize state.",
            )
        )
    else:
        if bool(state.kill_switch_active):
            checks.append(
                _check(
                    check_id="paper_state",
                    category="SAFETY",
                    label="Paper account state",
                    status=STATUS_FAIL,
                    detail="Kill-switch is active.",
                    action="Review drawdown and reset only after an operator review.",
                    metadata={"cooldown_days_left": int(state.cooldown_days_left)},
                )
            )
        elif int(state.cooldown_days_left or 0) > 0:
            checks.append(
                _check(
                    check_id="paper_state",
                    category="SAFETY",
                    label="Paper account state",
                    status=STATUS_WARN,
                    detail=f"Cooldown has {int(state.cooldown_days_left)} day(s) remaining.",
                    action="Keep new entries disabled until cooldown ends.",
                )
            )
        else:
            checks.append(
                _check(
                    check_id="paper_state",
                    category="SAFETY",
                    label="Paper account state",
                    status=STATUS_PASS,
                    detail="Kill-switch is clear and no cooldown is active.",
                    metadata={
                        "equity": float(state.equity),
                        "drawdown": float(state.drawdown),
                    },
                )
            )

    safe_mode_on_fail = bool(health.get("safe_mode_on_fail"))
    safe_mode_action = str(health.get("safe_mode_action") or "").strip().lower()
    if safe_mode_on_fail and safe_mode_action in {"exits_only", "shadow_only"}:
        checks.append(
            _check(
                check_id="safe_mode",
                category="SAFETY",
                label="Data-fail safe mode",
                status=STATUS_PASS,
                detail=f"Data-quality failures route to {safe_mode_action}.",
            )
        )
    else:
        checks.append(
            _check(
                check_id="safe_mode",
                category="SAFETY",
                label="Data-fail safe mode",
                status=STATUS_FAIL,
                detail="Data-quality failure handling is not conservative.",
                action="Enable operate_safe_mode_on_fail and use exits_only or shadow_only.",
            )
        )

    risk_per_trade = _setting_float(
        state_settings,
        settings,
        "risk_per_trade",
        fallback_key="risk_per_trade",
    )
    max_positions = _setting_int(state_settings, settings, "max_positions")
    kill_switch_dd = _setting_float(
        state_settings,
        settings,
        "kill_switch_dd",
        fallback_key="kill_switch_drawdown",
    )
    risk_status = STATUS_PASS
    risk_action: str | None = None
    risk_detail = (
        f"Risk {risk_per_trade:.3%}, max positions {max_positions}, "
        f"kill-switch {kill_switch_dd:.1%}."
    )
    if risk_per_trade > 0.01 or max_positions > 5 or kill_switch_dd > 0.12:
        risk_status = STATUS_FAIL
        risk_action = "Reduce risk_per_trade, max_positions, and kill_switch_dd before operating."
    elif risk_per_trade > 0.005 or max_positions > 3 or kill_switch_dd > 0.08:
        risk_status = STATUS_WARN
        risk_action = "Use the Atlas conservative defaults for production paper validation."
    checks.append(
        _check(
            check_id="risk_limits",
            category="SAFETY",
            label="Risk limits",
            status=risk_status,
            detail=risk_detail,
            action=risk_action,
            metadata={
                "risk_per_trade": risk_per_trade,
                "max_positions": max_positions,
                "kill_switch_dd": kill_switch_dd,
            },
        )
    )

    commission_bps = _setting_float(state_settings, settings, "commission_bps")
    slippage_bps = _setting_float(state_settings, settings, "slippage_base_bps")
    cost_model_enabled = _setting_bool(state_settings, settings, "cost_model_enabled")
    if commission_bps <= 0 or slippage_bps <= 0:
        checks.append(
            _check(
                check_id="costs_slippage",
                category="SAFETY",
                label="Costs and slippage",
                status=STATUS_FAIL,
                detail="Commission or slippage is zero.",
                action="Configure positive commission_bps and slippage_base_bps.",
            )
        )
    elif not cost_model_enabled:
        checks.append(
            _check(
                check_id="costs_slippage",
                category="SAFETY",
                label="Costs and slippage",
                status=STATUS_WARN,
                detail="Simple costs are configured, but the detailed Indian cost model is disabled.",
                action="Enable cost_model_enabled before relying on paper P&L.",
                metadata={"commission_bps": commission_bps, "slippage_base_bps": slippage_bps},
            )
        )
    else:
        checks.append(
            _check(
                check_id="costs_slippage",
                category="SAFETY",
                label="Costs and slippage",
                status=STATUS_PASS,
                detail="Positive slippage, commission, and detailed cost model are configured.",
            )
        )

    auto_run_enabled = bool(health.get("auto_run_enabled"))
    auto_run_shadow_only = bool(health.get("auto_run_shadow_only"))
    if auto_run_enabled and auto_run_shadow_only:
        checks.append(
            _check(
                check_id="shadow_scheduler",
                category="AUTOMATION",
                label="Scheduled operate mode",
                status=STATUS_PASS,
                detail="Scheduled operate runs are enabled and shadow-only.",
            )
        )
    elif auto_run_enabled:
        checks.append(
            _check(
                check_id="shadow_scheduler",
                category="AUTOMATION",
                label="Scheduled operate mode",
                status=STATUS_FAIL,
                detail="Scheduled operate runs can mutate the main paper state.",
                action="Set operate_auto_run_shadow_only=true for production validation.",
            )
        )
    else:
        checks.append(
            _check(
                check_id="shadow_scheduler",
                category="AUTOMATION",
                label="Scheduled operate mode",
                status=STATUS_WARN,
                detail="Scheduled operate is disabled; only manual runs will validate operations.",
                action="Enable scheduled shadow-only operate once setup is stable.",
            )
        )

    auto_eval_auto_switch = bool(
        state_settings.get(
            "operate_auto_eval_auto_switch",
            settings.operate_auto_eval_auto_switch,
        )
    )
    auto_eval_shadow_gate = bool(
        state_settings.get(
            "operate_auto_eval_shadow_only_gate",
            settings.operate_auto_eval_shadow_only_gate,
        )
    )
    if auto_eval_auto_switch:
        checks.append(
            _check(
                check_id="auto_switch",
                category="AUTOMATION",
                label="Policy auto-switch",
                status=STATUS_FAIL if not auto_eval_shadow_gate else STATUS_WARN,
                detail="Automatic policy switching is enabled.",
                action="Keep auto-switch disabled until paper evidence is reviewed by an operator.",
            )
        )
    else:
        checks.append(
            _check(
                check_id="auto_switch",
                category="AUTOMATION",
                label="Policy auto-switch",
                status=STATUS_PASS,
                detail="Automatic policy switching is disabled.",
            )
        )

    latest_quality = health.get("latest_data_quality")
    if not isinstance(latest_quality, dict):
        checks.append(
            _check(
                check_id="data_quality",
                category="DATA",
                label="Latest data quality",
                status=STATUS_WARN,
                detail="No data-quality report exists for the active context.",
                action="Run data quality before production paper operation.",
            )
        )
    else:
        quality_status = str(latest_quality.get("status", "")).upper()
        if quality_status in {"OK", "PASS"}:
            checks.append(
                _check(
                    check_id="data_quality",
                    category="DATA",
                    label="Latest data quality",
                    status=STATUS_PASS,
                    detail="Latest data-quality report passed.",
                    metadata={"report_id": latest_quality.get("id")},
                )
            )
        elif quality_status == "WARN":
            checks.append(
                _check(
                    check_id="data_quality",
                    category="DATA",
                    label="Latest data quality",
                    status=STATUS_WARN,
                    detail="Latest data-quality report has warnings.",
                    action="Review data-quality issues before allowing entries.",
                    metadata={"report_id": latest_quality.get("id")},
                )
            )
        else:
            checks.append(
                _check(
                    check_id="data_quality",
                    category="DATA",
                    label="Latest data quality",
                    status=STATUS_FAIL,
                    detail=f"Latest data-quality report is {quality_status or 'unknown'}.",
                    action="Fix/import data or operate in shadow/exits-only mode.",
                    metadata={"report_id": latest_quality.get("id")},
                )
            )

    provider_enabled = _setting_bool(state_settings, settings, "data_updates_provider_enabled")
    latest_provider_update = health.get("latest_provider_update")
    if provider_enabled and not isinstance(latest_provider_update, dict):
        checks.append(
            _check(
                check_id="provider_updates",
                category="DATA",
                label="Provider update path",
                status=STATUS_WARN,
                detail="Provider updates are enabled but no provider update run exists.",
                action="Run provider updates once and verify coverage.",
            )
        )
    elif provider_enabled:
        provider_status = str(latest_provider_update.get("status", "")).upper()
        checks.append(
            _check(
                check_id="provider_updates",
                category="DATA",
                label="Provider update path",
                status=STATUS_PASS if provider_status == "SUCCEEDED" else STATUS_FAIL,
                detail=f"Latest provider update status is {provider_status or 'unknown'}.",
                action=None
                if provider_status == "SUCCEEDED"
                else "Resolve provider update failures.",
            )
        )
    else:
        checks.append(
            _check(
                check_id="provider_updates",
                category="DATA",
                label="Provider update path",
                status=STATUS_WARN,
                detail="Provider updates are disabled; data freshness depends on manual imports.",
                action="Enable provider updates or maintain a documented manual import runbook.",
            )
        )

    latest_confidence_gate = health.get("latest_confidence_gate")
    if not isinstance(latest_confidence_gate, dict):
        checks.append(
            _check(
                check_id="confidence_gate",
                category="DATA",
                label="Confidence gate",
                status=STATUS_WARN,
                detail="No confidence gate snapshot exists yet.",
                action="Run operate once to generate confidence gate evidence.",
            )
        )
    else:
        decision = str(latest_confidence_gate.get("decision", "PASS")).upper()
        if decision == "PASS":
            status = STATUS_PASS
            action = None
        elif decision == "BLOCK_ENTRIES":
            status = STATUS_FAIL
            action = "Review provider confidence and keep entries blocked."
        else:
            status = STATUS_WARN
            action = "Treat this as shadow-only until confidence recovers."
        checks.append(
            _check(
                check_id="confidence_gate",
                category="DATA",
                label="Confidence gate",
                status=status,
                detail=f"Latest confidence gate decision is {decision}.",
                action=action,
                metadata={
                    "avg_confidence": latest_confidence_gate.get("avg_confidence"),
                    "trading_date": latest_confidence_gate.get("trading_date"),
                },
            )
        )

    error_count = int(dict(health.get("recent_event_counts_24h") or {}).get("ERROR", 0) or 0)
    if error_count >= 5:
        checks.append(
            _check(
                check_id="recent_errors",
                category="SYSTEM",
                label="Recent operate errors",
                status=STATUS_FAIL,
                detail=f"{error_count} operate error events were recorded in the last 24 hours.",
                action="Review Ops events and clear recurring failures before production paper.",
            )
        )
    elif error_count > 0:
        checks.append(
            _check(
                check_id="recent_errors",
                category="SYSTEM",
                label="Recent operate errors",
                status=STATUS_WARN,
                detail=f"{error_count} operate error event(s) were recorded in the last 24 hours.",
                action="Review recent error events.",
            )
        )
    else:
        checks.append(
            _check(
                check_id="recent_errors",
                category="SYSTEM",
                label="Recent operate errors",
                status=STATUS_PASS,
                detail="No operate error events were recorded in the last 24 hours.",
            )
        )

    fast_mode_enabled = bool(health.get("fast_mode_enabled"))
    checks.append(
        _check(
            check_id="fast_mode",
            category="SYSTEM",
            label="Fast mode",
            status=STATUS_WARN if fast_mode_enabled else STATUS_PASS,
            detail=(
                "Fast/e2e mode is enabled; runtime limits are reduced."
                if fast_mode_enabled
                else "Fast/e2e mode is disabled."
            ),
            action="Disable ATLAS_FAST_MODE and ATLAS_E2E_FAST for production paper runs."
            if fast_mode_enabled
            else None,
        )
    )

    last_run_step_at = health.get("last_run_step_at")
    checks.append(
        _check(
            check_id="last_paper_run",
            category="SYSTEM",
            label="Latest paper run",
            status=STATUS_PASS if last_run_step_at else STATUS_WARN,
            detail=(
                f"Latest paper run-step was at {last_run_step_at}."
                if last_run_step_at
                else "No paper run-step has completed yet."
            ),
            action=None
            if last_run_step_at
            else "Run a shadow operate pass before relying on automation.",
        )
    )

    verdict = _worst_status(checks)
    counts = _status_counts(checks)
    blockers = [
        str(item["label"]) for item in checks if str(item.get("status", "")).upper() == STATUS_FAIL
    ]
    warnings = [
        str(item["label"]) for item in checks if str(item.get("status", "")).upper() == STATUS_WARN
    ]

    real_money_required_controls = [
        "Broker execution adapter with exchange/broker-required algo tagging and approvals",
        "Broker reconciliation for positions, cash, holdings, fills, and rejected orders",
        "Duplicate-order protection with broker order idempotency and restart recovery",
        "Hard daily loss and exposure limits enforced outside the strategy layer",
        "Manual confirm switch, emergency flatten, and immutable audit log review",
        "Compliance review for SEBI/broker retail algo requirements before API execution",
    ]
    return {
        "target": "PRODUCTION_PAPER",
        "verdict": verdict,
        "can_run_production_paper": verdict != "BLOCKED",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "active_bundle_id": active_bundle_id,
        "active_timeframe": active_timeframe,
        "operate_mode": health.get("operate_mode"),
        "paper_mode": paper_mode,
        "summary": {
            "pass": counts.get(STATUS_PASS, 0),
            "warn": counts.get(STATUS_WARN, 0),
            "fail": counts.get(STATUS_FAIL, 0),
            "blockers": blockers,
            "warnings": warnings,
        },
        "checks": checks,
        "real_money": {
            "verdict": "BLOCKED",
            "reason": "Atlas has no approved live broker execution adapter or real-money control plane.",
            "required_controls": real_money_required_controls,
        },
        "next_actions": [
            item["action"]
            for item in checks
            if item.get("action") and str(item.get("status", "")).upper() != STATUS_PASS
        ][:6],
    }
