from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import json
from typing import Any
from zoneinfo import ZoneInfo

from fastapi.encoders import jsonable_encoder
import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import (
    AuditLog,
    Dataset,
    DatasetBundle,
    Instrument,
    PolicyEnsemble,
    PolicyHealthSnapshot,
    PaperRun,
    PaperOrder,
    PaperPosition,
    PaperState,
    Policy,
    ShadowPaperState,
    Symbol,
)
from app.engine.costs import (
    estimate_equity_delivery_cost,
    estimate_futures_cost,
    estimate_intraday_cost,
)
from app.engine.signal_engine import SignalGenerationResult, generate_signals_for_policy
from app.services.confidence_gate import (
    DECISION_BLOCK_ENTRIES,
    DECISION_SHADOW_ONLY,
    evaluate_confidence_gate,
    resolve_confidence_risk_scaling,
)
from app.services.confidence_agg import (
    latest_daily_confidence_agg,
    serialize_daily_confidence_agg,
    upsert_daily_confidence_agg,
)
from app.services.data_quality import (
    STATUS_FAIL as DATA_QUALITY_FAIL,
    STATUS_WARN as DATA_QUALITY_WARN,
    run_data_quality_report,
)
from app.services.data_store import DataStore
from app.services.data_updates import inactive_symbols_for_selection
from app.services.ensembles import (
    get_active_policy_ensemble,
    list_policy_ensemble_members,
    list_policy_ensemble_regime_weights,
    serialize_policy_ensemble,
)
from app.services.fast_mode import (
    clamp_job_timeout_seconds,
    clamp_scan_symbols,
    prefer_sample_bundle_id,
    prefer_sample_dataset_id,
    resolve_seed,
)
from app.services.operate_events import emit_operate_event
from app.services.effective_context import build_effective_trading_context
from app.services.no_trade import evaluate_no_trade_gate
from app.services.paper_sim_adapter import execute_paper_step_with_simulator
from app.services.portfolio_risk import create_portfolio_risk_snapshot
from app.services.policy_health import (
    DEGRADED,
    HEALTHY,
    PAUSED,
    RETIRED,
    WARNING,
    apply_policy_health_actions,
    get_policy_health_snapshot,
    latest_policy_health_snapshot_for_policy,
    policy_status,
    select_fallback_policy,
)
from app.services.reports import generate_daily_report
from app.services.regime import regime_policy

IST_ZONE = ZoneInfo("Asia/Kolkata")
FUTURE_KINDS = {"STOCK_FUT", "INDEX_FUT"}


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _paper_effective_context(
    *,
    session: Session,
    settings: Settings,
    state_settings: dict[str, Any],
    bundle_id: int | None,
    timeframe: str,
    asof_dt: datetime,
    provider_stage_status: str | None = None,
    confidence_gate_snapshot: dict[str, Any] | None = None,
    data_digest: str | None = None,
    engine_version: str | None = None,
    seed: int | None = None,
    notes: list[str] | None = None,
    store: DataStore | None = None,
) -> dict[str, Any]:
    agg_row = latest_daily_confidence_agg(
        session,
        bundle_id=(int(bundle_id) if isinstance(bundle_id, int) and bundle_id > 0 else None),
        timeframe=str(timeframe),
    )
    gate_summary = (
        dict(confidence_gate_snapshot.get("summary", {}))
        if isinstance(confidence_gate_snapshot, dict)
        and isinstance(confidence_gate_snapshot.get("summary", {}), dict)
        else {}
    )
    return build_effective_trading_context(
        session,
        settings=settings,
        bundle_id=(int(bundle_id) if isinstance(bundle_id, int) and bundle_id > 0 else None),
        timeframe=str(timeframe),
        asof_ts=asof_dt,
        segment=str(
            state_settings.get("trading_calendar_segment", settings.trading_calendar_segment)
        ),
        provider_stage_status=provider_stage_status,
        confidence_gate_decision=(
            str(confidence_gate_snapshot.get("decision"))
            if isinstance(confidence_gate_snapshot, dict)
            and confidence_gate_snapshot.get("decision") is not None
            else None
        ),
        confidence_risk_scale=(
            float(gate_summary.get("confidence_risk_scale"))
            if gate_summary.get("confidence_risk_scale") is not None
            else None
        ),
        agg_row=agg_row,
        data_digest=data_digest,
        engine_version=engine_version,
        seed=(int(seed) if isinstance(seed, int) else None),
        notes=notes or [],
        store=store,
    )


def get_or_create_paper_state(session: Session, settings: Settings) -> PaperState:
    state = session.get(PaperState, 1)
    if state is not None:
        return state

    state = PaperState(
        id=1,
        equity=1_000_000.0,
        cash=1_000_000.0,
        peak_equity=1_000_000.0,
        drawdown=0.0,
        kill_switch_active=False,
        cooldown_days_left=0,
        settings_json={
            "risk_per_trade": settings.risk_per_trade,
            "max_positions": settings.max_positions,
            "kill_switch_dd": settings.kill_switch_drawdown,
            "max_position_value_pct_adv": settings.max_position_value_pct_adv,
            "diversification_corr_threshold": settings.diversification_corr_threshold,
            "allowed_sides": settings.allowed_sides,
            "paper_short_squareoff_time": settings.paper_short_squareoff_time,
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
            "evaluations_auto_promote_enabled": settings.evaluations_auto_promote_enabled,
            "evaluations_min_window_days": settings.evaluations_min_window_days,
            "evaluations_score_margin": settings.evaluations_score_margin,
            "evaluations_max_dd_multiplier": settings.evaluations_max_dd_multiplier,
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
            "data_quality_stale_severity": (
                "FAIL"
                if str(settings.operate_mode).strip().lower() == "live"
                else settings.data_quality_stale_severity
            ),
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
            "data_updates_provider_mode": settings.data_updates_provider_mode,
            "data_updates_provider_kind": settings.data_updates_provider_kind,
            "data_updates_provider_priority_order": settings.data_updates_provider_priority_order,
            "data_updates_provider_nse_eod_enabled": settings.data_updates_provider_nse_eod_enabled,
            "data_updates_provider_max_symbols_per_run": settings.data_updates_provider_max_symbols_per_run,
            "data_updates_provider_max_calls_per_run": settings.data_updates_provider_max_calls_per_run,
            "data_updates_provider_timeframe_enabled": settings.data_updates_provider_timeframe_enabled,
            "data_updates_provider_timeframes": settings.data_updates_provider_timeframes,
            "data_updates_provider_repair_last_n_trading_days": settings.data_updates_provider_repair_last_n_trading_days,
            "data_updates_provider_backfill_max_days": settings.data_updates_provider_backfill_max_days,
            "data_updates_provider_allow_partial_4h_ish": settings.data_updates_provider_allow_partial_4h_ish,
            "data_provenance_confidence_upstox": settings.data_provenance_confidence_upstox,
            "data_provenance_confidence_nse_eod": settings.data_provenance_confidence_nse_eod,
            "data_provenance_confidence_inbox": settings.data_provenance_confidence_inbox,
            "data_quality_confidence_fail_threshold": settings.data_quality_confidence_fail_threshold,
            "coverage_missing_latest_warn_pct": settings.coverage_missing_latest_warn_pct,
            "coverage_missing_latest_fail_pct": settings.coverage_missing_latest_fail_pct,
            "coverage_inactive_after_missing_days": settings.coverage_inactive_after_missing_days,
            "upstox_auto_renew_enabled": settings.upstox_auto_renew_enabled,
            "upstox_auto_renew_time_ist": settings.upstox_auto_renew_time_ist,
            "upstox_auto_renew_if_expires_within_hours": settings.upstox_auto_renew_if_expires_within_hours,
            "upstox_auto_renew_lead_hours_before_open": settings.upstox_auto_renew_lead_hours_before_open,
            "upstox_auto_renew_only_when_provider_enabled": settings.upstox_auto_renew_only_when_provider_enabled,
            "operate_provider_stage_on_token_invalid": settings.operate_provider_stage_on_token_invalid,
            "operate_last_upstox_auto_renew_date": None,
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
            "confidence_gate_enabled": (
                True if str(settings.operate_mode).strip().lower() == "live" else False
            ),
            "confidence_gate_avg_threshold": settings.confidence_gate_avg_threshold,
            "confidence_gate_low_symbol_threshold": settings.confidence_gate_low_symbol_threshold,
            "confidence_gate_low_pct_threshold": settings.confidence_gate_low_pct_threshold,
            "confidence_gate_fallback_pct_threshold": settings.confidence_gate_fallback_pct_threshold,
            "confidence_gate_hard_floor": settings.confidence_gate_hard_floor,
            "confidence_gate_action_on_trigger": settings.confidence_gate_action_on_trigger,
            "confidence_gate_lookback_days": settings.confidence_gate_lookback_days,
            "confidence_drop_warn_threshold": settings.confidence_drop_warn_threshold,
            "confidence_provider_mix_shift_warn_pct": settings.confidence_provider_mix_shift_warn_pct,
            "confidence_risk_scaling_enabled": (
                True if str(settings.operate_mode).strip().lower() == "live" else False
            ),
            "confidence_risk_scale_exponent": settings.confidence_risk_scale_exponent,
            "confidence_risk_scale_low_threshold": settings.confidence_risk_scale_low_threshold,
            "paper_mode": "strategy",
            "active_policy_id": None,
            "active_ensemble_id": None,
            "active_ensemble_name": None,
        },
    )
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


def get_positions(session: Session) -> list[PaperPosition]:
    return list(session.exec(select(PaperPosition).order_by(PaperPosition.opened_at.desc())).all())


def get_orders(session: Session) -> list[PaperOrder]:
    return list(session.exec(select(PaperOrder).order_by(PaperOrder.created_at.desc())).all())


def _shadow_key(bundle_id: int | None, policy_id: Any) -> tuple[int, int]:
    bundle_key = int(bundle_id) if isinstance(bundle_id, int) and bundle_id > 0 else 0
    policy_key = 0
    try:
        if policy_id is not None:
            policy_key = int(policy_id)
    except (TypeError, ValueError):
        policy_key = 0
    return bundle_key, max(0, policy_key)


def _default_shadow_state_payload(
    *,
    live_state: PaperState,
    state_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "equity": float(live_state.equity),
        "cash": float(live_state.cash),
        "peak_equity": float(live_state.peak_equity),
        "drawdown": float(live_state.drawdown),
        "kill_switch_active": bool(live_state.kill_switch_active),
        "cooldown_days_left": int(live_state.cooldown_days_left),
        "settings_json": dict(state_settings or {}),
        "positions": [],
        "orders": [],
    }


def _get_or_create_shadow_state(
    *,
    session: Session,
    live_state: PaperState,
    state_settings: dict[str, Any],
    bundle_id: int | None,
    policy_id: Any,
) -> ShadowPaperState:
    bundle_key, policy_key = _shadow_key(bundle_id, policy_id)
    row = session.exec(
        select(ShadowPaperState)
        .where(ShadowPaperState.bundle_id == bundle_key)
        .where(ShadowPaperState.policy_id == policy_key)
        .order_by(ShadowPaperState.id.desc())
    ).first()
    if row is not None:
        return row
    row = ShadowPaperState(
        bundle_id=bundle_key,
        policy_id=policy_key,
        state_json=_default_shadow_state_payload(
            live_state=live_state, state_settings=state_settings
        ),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _shadow_state_snapshot(
    *,
    row: ShadowPaperState,
    live_state: PaperState,
    state_settings: dict[str, Any],
) -> dict[str, Any]:
    source = row.state_json if isinstance(row.state_json, dict) else {}
    baseline = _default_shadow_state_payload(live_state=live_state, state_settings=state_settings)
    merged = {**baseline, **source}
    positions = source.get("positions", [])
    if isinstance(positions, list):
        merged["positions"] = [dict(item) for item in positions if isinstance(item, dict)]
    else:
        merged["positions"] = []
    orders = source.get("orders", [])
    if isinstance(orders, list):
        merged["orders"] = [dict(item) for item in orders if isinstance(item, dict)]
    else:
        merged["orders"] = []
    return merged


def _save_shadow_state_snapshot(
    *,
    session: Session,
    row: ShadowPaperState,
    snapshot: dict[str, Any],
    last_run_id: int | None,
) -> None:
    row.state_json = dict(snapshot)
    row.last_run_id = int(last_run_id) if isinstance(last_run_id, int) and last_run_id > 0 else None
    row.updated_at = _utc_now()
    session.add(row)


def _log(session: Session, event_type: str, payload: dict[str, Any]) -> None:
    session.add(AuditLog(type=event_type, payload_json=payload))


def _dump_model(value: Any) -> Any:
    fields = getattr(type(value), "model_fields", None)
    if isinstance(fields, dict):
        return jsonable_encoder({key: getattr(value, key, None) for key in fields})
    return jsonable_encoder(value)


def _stable_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _result_digest(summary: dict[str, Any], cost_summary: dict[str, Any]) -> str:
    payload = {
        "signals_source": summary.get("signals_source"),
        "generated_signals_count": summary.get("generated_signals_count"),
        "selected_signals_count": summary.get("selected_signals_count"),
        "skipped_signals_count": summary.get("skipped_signals_count"),
        "selected_reason_histogram": summary.get("selected_reason_histogram", {}),
        "skipped_reason_histogram": summary.get("skipped_reason_histogram", {}),
        "risk_scale": summary.get("risk_scale"),
        "realized_vol": summary.get("realized_vol"),
        "target_vol": summary.get("target_vol"),
        "caps_applied": summary.get("caps_applied", {}),
        "equity_after": summary.get("equity_after"),
        "cash_after": summary.get("cash_after"),
        "net_pnl": summary.get("net_pnl"),
        "paper_engine": summary.get("paper_engine"),
        "engine_version": summary.get("engine_version"),
        "data_digest": summary.get("data_digest"),
        "seed": summary.get("seed"),
        "cost_summary": {
            "entry_cost_total": cost_summary.get("entry_cost_total"),
            "exit_cost_total": cost_summary.get("exit_cost_total"),
            "total_cost": cost_summary.get("total_cost"),
            "entry_slippage_cost": cost_summary.get("entry_slippage_cost"),
            "exit_slippage_cost": cost_summary.get("exit_slippage_cost"),
        },
    }
    return _stable_hash(payload)


def _cost_ratio_spike_active(
    session: Session,
    *,
    state_settings: dict[str, Any],
    settings: Settings,
) -> tuple[bool, dict[str, Any]]:
    days = max(
        1,
        int(
            state_settings.get(
                "operate_cost_ratio_spike_days",
                settings.operate_cost_ratio_spike_days,
            )
        ),
    )
    threshold = float(
        state_settings.get(
            "operate_cost_ratio_spike_threshold",
            settings.operate_cost_ratio_spike_threshold,
        )
    )
    rows = session.exec(select(PaperRun).order_by(PaperRun.asof_ts.desc()).limit(days)).all()
    if len(rows) < days:
        return False, {"days": days, "threshold": threshold, "ratios": []}
    ratios: list[float] = []
    for row in rows:
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        cost_summary = row.cost_summary_json if isinstance(row.cost_summary_json, dict) else {}
        total_cost = float(summary.get("total_cost", cost_summary.get("total_cost", 0.0)))
        gross_pnl = abs(float(summary.get("gross_pnl", 0.0)))
        if gross_pnl <= 1e-9:
            return False, {"days": days, "threshold": threshold, "ratios": ratios}
        ratios.append(total_cost / gross_pnl)
    return all(value >= threshold for value in ratios), {
        "days": days,
        "threshold": threshold,
        "ratios": ratios,
    }


def _scan_truncation_guard(
    session: Session,
    *,
    state_settings: dict[str, Any],
    settings: Settings,
) -> tuple[bool, dict[str, Any]]:
    days = max(
        1,
        int(
            state_settings.get(
                "operate_scan_truncated_warn_days",
                settings.operate_scan_truncated_warn_days,
            )
        ),
    )
    reduced_to = max(
        5,
        int(
            state_settings.get(
                "operate_scan_truncated_reduce_to",
                settings.operate_scan_truncated_reduce_to,
            )
        ),
    )
    rows = session.exec(select(PaperRun).order_by(PaperRun.asof_ts.desc()).limit(days)).all()
    if len(rows) < days:
        return False, {"days": days, "reduced_to": reduced_to, "truncated_count": 0}
    truncated_count = int(sum(1 for row in rows if bool(row.scan_truncated)))
    return truncated_count >= days, {
        "days": days,
        "reduced_to": reduced_to,
        "truncated_count": truncated_count,
    }


def _data_quality_guard(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    state_settings: dict[str, Any],
    bundle_id: int | None,
    timeframe: str,
    asof_ts: datetime | None = None,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    if bundle_id is None:
        return {
            "status": None,
            "active": False,
            "action": "none",
            "safe_mode_reason": None,
            "warn_summary": [],
            "report": None,
        }
    report = run_data_quality_report(
        session=session,
        settings=settings,
        store=store,
        bundle_id=int(bundle_id),
        timeframe=str(timeframe),
        overrides=state_settings,
        reference_ts=asof_ts,
        correlation_id=correlation_id,
    )
    status = str(report.status).upper()
    safe_mode_on_fail = bool(
        state_settings.get("operate_safe_mode_on_fail", settings.operate_safe_mode_on_fail)
    )
    action = str(
        state_settings.get("operate_safe_mode_action", settings.operate_safe_mode_action)
    ).lower()
    safe_mode_active = bool(status == DATA_QUALITY_FAIL and safe_mode_on_fail)
    reason = "data_quality_fail_safe_mode" if safe_mode_active else None
    warn_summary = [
        {
            "code": str(item.get("code")),
            "message": str(item.get("message")),
            "severity": str(item.get("severity")),
        }
        for item in (report.issues_json or [])[:5]
    ]
    if safe_mode_active:
        emit_operate_event(
            session,
            severity="ERROR",
            category="DATA",
            message="Safe mode activated due to data quality failure.",
            details={
                "bundle_id": bundle_id,
                "timeframe": timeframe,
                "report_id": report.id,
                "safe_mode_action": action,
                "reason": reason,
            },
            correlation_id=correlation_id,
        )
    elif status == DATA_QUALITY_WARN:
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="Data quality warnings detected for active run context.",
            details={
                "bundle_id": bundle_id,
                "timeframe": timeframe,
                "report_id": report.id,
                "warning_count": len(report.issues_json or []),
            },
            correlation_id=correlation_id,
        )
    return {
        "status": status,
        "active": safe_mode_active,
        "action": action,
        "safe_mode_reason": reason,
        "warn_summary": warn_summary,
        "report": report,
    }


def _emit_digest_mismatch_if_any(
    *,
    session: Session,
    run_row: PaperRun,
    run_summary: dict[str, Any],
) -> None:
    data_digest = str(run_summary.get("data_digest", ""))
    seed = run_summary.get("seed")
    result_digest = str(run_summary.get("result_digest", ""))
    if not data_digest or result_digest == "":
        return
    asof = run_row.asof_ts
    day_start = datetime.combine(asof.date(), time.min, tzinfo=timezone.utc)
    day_end = datetime.combine(asof.date(), time.max, tzinfo=timezone.utc)
    rows = session.exec(
        select(PaperRun)
        .where(PaperRun.id != run_row.id)
        .where(PaperRun.asof_ts >= day_start)
        .where(PaperRun.asof_ts <= day_end)
        .order_by(PaperRun.created_at.desc())
    ).all()
    for row in rows:
        prev = row.summary_json if isinstance(row.summary_json, dict) else {}
        if str(prev.get("data_digest", "")) != data_digest:
            continue
        if prev.get("seed") != seed:
            continue
        prev_digest = str(prev.get("result_digest", ""))
        if not prev_digest:
            continue
        if prev_digest != result_digest:
            mismatch_payload = {
                "message": "digest_mismatch",
                "current_run_id": run_row.id,
                "previous_run_id": row.id,
                "asof_date": asof.date().isoformat(),
                "bundle_id": run_row.bundle_id,
                "policy_id": run_row.policy_id,
                "seed": seed,
                "data_digest": data_digest,
                "previous_result_digest": prev_digest,
                "current_result_digest": result_digest,
            }
            _log(session, "digest_mismatch", mismatch_payload)
            emit_operate_event(
                session,
                severity="WARN",
                category="SYSTEM",
                message="Digest mismatch detected for same day/config/data digest.",
                details=mismatch_payload,
                correlation_id=str(run_row.id),
            )
        return


def _position_size(equity: float, risk_per_trade: float, stop_distance: float) -> int:
    if stop_distance <= 0:
        return 0
    risk_amount = equity * risk_per_trade
    return int(np.floor(risk_amount / stop_distance))


def _position_size_lots(
    equity: float,
    risk_per_trade: float,
    stop_distance: float,
    lot_size: int,
) -> int:
    if stop_distance <= 0 or lot_size <= 0:
        return 0
    risk_amount = equity * risk_per_trade
    return int(np.floor(risk_amount / (stop_distance * lot_size)))


def _is_futures_kind(instrument_kind: str) -> bool:
    return str(instrument_kind).upper() in FUTURE_KINDS


def _futures_margin_pct(state_settings: dict[str, Any], settings: Settings) -> float:
    return max(
        0.0,
        float(
            state_settings.get("futures_initial_margin_pct", settings.futures_initial_margin_pct)
        ),
    )


def _correlation_reject(
    candidate: dict[str, Any],
    selected_symbols: set[str],
    threshold: float,
) -> bool:
    matrix = candidate.get("correlations", {})
    if not isinstance(matrix, dict):
        return False
    for symbol in selected_symbols:
        corr = matrix.get(symbol)
        if corr is None:
            continue
        try:
            if abs(float(corr)) >= threshold:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _cost_settings(state_settings: dict[str, Any], settings: Settings) -> dict[str, float]:
    return {
        "brokerage_bps": float(state_settings.get("brokerage_bps", settings.brokerage_bps)),
        "stt_delivery_buy_bps": float(
            state_settings.get("stt_delivery_buy_bps", settings.stt_delivery_buy_bps)
        ),
        "stt_delivery_sell_bps": float(
            state_settings.get("stt_delivery_sell_bps", settings.stt_delivery_sell_bps)
        ),
        "stt_intraday_buy_bps": float(
            state_settings.get("stt_intraday_buy_bps", settings.stt_intraday_buy_bps)
        ),
        "stt_intraday_sell_bps": float(
            state_settings.get("stt_intraday_sell_bps", settings.stt_intraday_sell_bps)
        ),
        "exchange_txn_bps": float(
            state_settings.get("exchange_txn_bps", settings.exchange_txn_bps)
        ),
        "sebi_bps": float(state_settings.get("sebi_bps", settings.sebi_bps)),
        "stamp_delivery_buy_bps": float(
            state_settings.get("stamp_delivery_buy_bps", settings.stamp_delivery_buy_bps)
        ),
        "stamp_intraday_buy_bps": float(
            state_settings.get("stamp_intraday_buy_bps", settings.stamp_intraday_buy_bps)
        ),
        "gst_rate": float(state_settings.get("gst_rate", settings.gst_rate)),
        "futures_brokerage_bps": float(
            state_settings.get("futures_brokerage_bps", settings.futures_brokerage_bps)
        ),
        "futures_stt_sell_bps": float(
            state_settings.get("futures_stt_sell_bps", settings.futures_stt_sell_bps)
        ),
        "futures_exchange_txn_bps": float(
            state_settings.get("futures_exchange_txn_bps", settings.futures_exchange_txn_bps)
        ),
        "futures_stamp_buy_bps": float(
            state_settings.get("futures_stamp_buy_bps", settings.futures_stamp_buy_bps)
        ),
    }


def _transaction_cost(
    *,
    notional: float,
    side: str,
    instrument_kind: str,
    settings: Settings,
    state_settings: dict[str, Any],
    override_cost_model: dict[str, Any] | None,
) -> float:
    if notional <= 0:
        return 0.0

    cost_model = dict(override_cost_model or {})
    enabled = bool(cost_model.get("enabled", state_settings.get("cost_model_enabled", False)))
    mode = str(cost_model.get("mode", state_settings.get("cost_mode", settings.cost_mode))).lower()
    if enabled:
        cfg = _cost_settings(state_settings, settings)
        cfg.update(
            {
                key: float(value)
                for key, value in cost_model.items()
                if key in cfg and isinstance(value, (int, float))
            }
        )
        kind = instrument_kind.upper()
        if kind in {"STOCK_FUT", "INDEX_FUT"}:
            return estimate_futures_cost(notional=notional, side=side, config=cfg)
        if mode == "intraday":
            return estimate_intraday_cost(notional=notional, side=side, config=cfg)
        return estimate_equity_delivery_cost(notional=notional, side=side, config=cfg)

    commission_bps = float(state_settings.get("commission_bps", settings.commission_bps))
    return max(0.0, notional * commission_bps / 10_000)


def _parse_asof(value: Any) -> datetime | date | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _resolve_execution_policy(
    session: Session,
    state: PaperState,
    settings: Settings,
    regime: str,
    policy_override_id: int | None = None,
    asof_date: date | None = None,
) -> dict[str, Any]:
    state_settings = state.settings_json or {}
    base_risk = float(state_settings.get("risk_per_trade", settings.risk_per_trade))
    base_max_positions = int(state_settings.get("max_positions", settings.max_positions))
    base = regime_policy(regime, base_risk, base_max_positions)
    resolved: dict[str, Any] = {
        **base,
        "mode": "strategy",
        "policy_id": None,
        "policy_name": None,
        "policy_definition": {},
        "selection_reason": f"Default regime policy for {regime}.",
        "policy_status": "ACTIVE",
        "health_status": HEALTHY,
        "health_reasons": [],
        "risk_scale_override": 1.0,
        "params": {},
        "cost_model": {},
        "allowed_instruments": {
            "BUY": ["EQUITY_CASH", "STOCK_FUT", "INDEX_FUT"],
            "SELL": ["EQUITY_CASH", "STOCK_FUT", "INDEX_FUT"],
        },
        "ranking_weights": {"signal": 1.0},
    }

    if policy_override_id is None and str(state_settings.get("paper_mode", "strategy")) != "policy":
        return resolved

    if policy_override_id is not None:
        policy_int = int(policy_override_id)
    else:
        policy_id = state_settings.get("active_policy_id")
        try:
            policy_int = int(policy_id)
        except (TypeError, ValueError):
            resolved["selection_reason"] = "Policy mode requested but active policy id is invalid."
            return resolved

    policy = session.get(Policy, policy_int)
    if policy is None:
        resolved["selection_reason"] = "Policy mode requested but policy record was not found."
        return resolved

    selected_policy = policy
    health_status = HEALTHY
    health_reasons: list[str] = []
    risk_scale_override = 1.0
    selection_reasons: list[str] = []
    short_window = max(
        5,
        int(
            state_settings.get(
                "health_window_days_short",
                settings.health_window_days_short,
            )
        ),
    )
    for _ in range(3):
        current_status = policy_status(selected_policy)
        if current_status in {PAUSED, RETIRED}:
            fallback = select_fallback_policy(
                session,
                current_policy_id=int(selected_policy.id),
                regime=regime,
            )
            if fallback is None:
                policy_definition = (
                    selected_policy.definition_json
                    if isinstance(selected_policy.definition_json, dict)
                    else {}
                )
                return {
                    "allowed_templates": [],
                    "risk_per_trade": 0.0,
                    "max_positions": 0,
                    "mode": "policy",
                    "policy_id": selected_policy.id,
                    "policy_name": selected_policy.name,
                    "policy_definition": policy_definition,
                    "selection_reason": (
                        f"Policy {selected_policy.name} is {current_status}; "
                        "no fallback policy available. Running shadow-only mode."
                    ),
                    "policy_status": current_status,
                    "health_status": current_status,
                    "health_reasons": [f"Policy status is {current_status}."],
                    "risk_scale_override": 0.0,
                    "params": {},
                    "cost_model": dict(policy_definition.get("cost_model", {})),
                    "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
                    "ranking_weights": dict(
                        policy_definition.get("ranking", {}).get("weights", {})
                    ),
                }
            selection_reasons.append(
                f"Policy {selected_policy.name} is {current_status}; fallback selected {fallback.name}."
            )
            selected_policy = fallback
            continue

        latest_snapshot = latest_policy_health_snapshot_for_policy(
            session,
            policy_id=int(selected_policy.id),
            window_days=short_window,
            asof_date=asof_date,
        )
        if latest_snapshot is not None:
            health_status = latest_snapshot.status
            health_reasons = list(latest_snapshot.reasons_json or [])
        else:
            health_status = HEALTHY
            health_reasons = []

        if health_status == DEGRADED:
            fallback = select_fallback_policy(
                session,
                current_policy_id=int(selected_policy.id),
                regime=regime,
            )
            if fallback is None:
                policy_definition = (
                    selected_policy.definition_json
                    if isinstance(selected_policy.definition_json, dict)
                    else {}
                )
                return {
                    "allowed_templates": [],
                    "risk_per_trade": 0.0,
                    "max_positions": 0,
                    "mode": "policy",
                    "policy_id": selected_policy.id,
                    "policy_name": selected_policy.name,
                    "policy_definition": policy_definition,
                    "selection_reason": (
                        f"Policy {selected_policy.name} health is DEGRADED and no fallback exists. "
                        "Running shadow-only mode."
                    ),
                    "policy_status": policy_status(selected_policy),
                    "health_status": health_status,
                    "health_reasons": health_reasons,
                    "risk_scale_override": 0.0,
                    "params": {},
                    "cost_model": dict(policy_definition.get("cost_model", {})),
                    "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
                    "ranking_weights": dict(
                        policy_definition.get("ranking", {}).get("weights", {})
                    ),
                }
            selection_reasons.append(
                f"Policy {selected_policy.name} health is DEGRADED; fallback selected {fallback.name}."
            )
            selected_policy = fallback
            continue

        if health_status == WARNING:
            risk_scale_override = max(
                0.0,
                float(
                    state_settings.get(
                        "drift_warning_risk_scale",
                        settings.drift_warning_risk_scale,
                    )
                ),
            )
        break

    policy = selected_policy
    policy_definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    regime_map = policy_definition.get("regime_map", {})
    if not isinstance(regime_map, dict):
        resolved["selection_reason"] = (
            "Policy has no valid regime map; using default regime policy."
        )
        return resolved

    regime_config = regime_map.get(regime)
    if not isinstance(regime_config, dict):
        return {
            "allowed_templates": [],
            "risk_per_trade": 0.0,
            "max_positions": 0,
            "mode": "policy",
            "policy_id": policy.id,
            "policy_name": policy.name,
            "policy_definition": policy_definition,
            "selection_reason": " ".join(
                selection_reasons + [f"Policy {policy.name} blocks new entries in {regime}."]
            ),
            "policy_status": policy_status(policy),
            "health_status": health_status,
            "health_reasons": health_reasons,
            "risk_scale_override": risk_scale_override,
            "params": {},
            "cost_model": dict(policy_definition.get("cost_model", {})),
            "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
            "ranking_weights": dict(policy_definition.get("ranking", {}).get("weights", {})),
        }

    strategy_key = regime_config.get("strategy_key")
    risk_scale = float(regime_config.get("risk_scale", 1.0))
    max_positions_scale = float(regime_config.get("max_positions_scale", 1.0))
    allowed_templates = (
        [str(strategy_key)]
        if isinstance(strategy_key, str) and strategy_key
        else [str(item) for item in regime_config.get("allowed_templates", []) if str(item)]
    )
    selection_reason = str(
        regime_config.get("reason", f"Policy {policy.name} selected for regime {regime}.")
    )
    if selection_reasons:
        selection_reason = " ".join(selection_reasons + [selection_reason])
    return {
        "allowed_templates": allowed_templates,
        "risk_per_trade": max(0.0, base_risk * risk_scale * risk_scale_override),
        "max_positions": max(0, int(round(base_max_positions * max_positions_scale))),
        "mode": "policy",
        "policy_id": policy.id,
        "policy_name": policy.name,
        "policy_definition": policy_definition,
        "selection_reason": selection_reason,
        "policy_status": policy_status(policy),
        "health_status": health_status,
        "health_reasons": health_reasons,
        "risk_scale_override": risk_scale_override,
        "params": dict(regime_config.get("params", {})),
        "cost_model": dict(policy_definition.get("cost_model", {})),
        "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
        "ranking_weights": dict(policy_definition.get("ranking", {}).get("weights", {})),
    }


def _resolve_timeframes(payload: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    payload_timeframes = payload.get("timeframes")
    if isinstance(payload_timeframes, list):
        values = [str(value).strip() for value in payload_timeframes if str(value).strip()]
        if values:
            return values

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        value = definition.get("timeframes", [])
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                return values
    return ["1d"]


def _resolve_symbol_scope(payload: dict[str, Any], policy: dict[str, Any]) -> str:
    explicit = payload.get("symbol_scope")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        universe = definition.get("universe", {})
        if isinstance(universe, dict):
            scope = universe.get("symbol_scope")
            if isinstance(scope, str) and scope.strip():
                return scope.strip()
    return "liquid"


def _resolve_seed(payload: dict[str, Any], policy: dict[str, Any], settings: Settings) -> int:
    explicit = payload.get("seed")
    if isinstance(explicit, int):
        return explicit
    try:
        return int(explicit)
    except (TypeError, ValueError):
        pass

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        ranking = definition.get("ranking", {})
        if isinstance(ranking, dict):
            value = ranking.get("seed")
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return resolve_seed(settings=settings, value=None, default=7)


def _resolve_bundle_id(
    session: Session,
    payload: dict[str, Any],
    policy: dict[str, Any],
    settings: Settings,
) -> int | None:
    explicit_id = payload.get("bundle_id")
    if isinstance(explicit_id, int):
        return explicit_id if session.get(DatasetBundle, explicit_id) is not None else None
    try:
        explicit_id_int = int(explicit_id)
        if session.get(DatasetBundle, explicit_id_int) is not None:
            return explicit_id_int
    except (TypeError, ValueError):
        pass

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        universe = definition.get("universe", {})
        if isinstance(universe, dict):
            bundle_id = universe.get("bundle_id")
            try:
                bundle_id_int = int(bundle_id)
                if session.get(DatasetBundle, bundle_id_int) is not None:
                    return bundle_id_int
            except (TypeError, ValueError):
                pass
    preferred = prefer_sample_bundle_id(session, settings=settings)
    if preferred is not None:
        return preferred
    return None


def _resolve_dataset_id(
    session: Session,
    payload: dict[str, Any],
    policy: dict[str, Any],
    timeframes: list[str],
    settings: Settings,
) -> int | None:
    explicit_id = payload.get("dataset_id")
    if isinstance(explicit_id, int):
        return explicit_id if session.get(Dataset, explicit_id) is not None else None
    try:
        explicit_id_int = int(explicit_id)
        if session.get(Dataset, explicit_id_int) is not None:
            return explicit_id_int
    except (TypeError, ValueError):
        pass

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        universe = definition.get("universe", {})
        if isinstance(universe, dict):
            dataset_id = universe.get("dataset_id")
            try:
                dataset_id_int = int(dataset_id)
                if session.get(Dataset, dataset_id_int) is not None:
                    return dataset_id_int
            except (TypeError, ValueError):
                pass

    preferred_timeframe = timeframes[0] if timeframes else "1d"
    preferred = prefer_sample_dataset_id(
        session,
        settings=settings,
        timeframe=preferred_timeframe,
    )
    if preferred is not None:
        return preferred
    candidate = session.exec(
        select(Dataset)
        .where(Dataset.timeframe == preferred_timeframe)
        .order_by(Dataset.created_at.desc())
    ).first()
    return candidate.id if candidate is not None else None


def _resolve_max_symbols_scan(
    payload: dict[str, Any],
    policy: dict[str, Any],
    state_settings: dict[str, Any],
    settings: Settings,
) -> int:
    explicit = payload.get("max_symbols_scan")
    if isinstance(explicit, int):
        requested = explicit
    else:
        try:
            requested = int(explicit)
        except (TypeError, ValueError):
            requested = None

    if requested is None:
        definition = policy.get("policy_definition", {})
        if isinstance(definition, dict):
            universe = definition.get("universe", {})
            if isinstance(universe, dict):
                value = universe.get("max_symbols_scan")
                try:
                    requested = int(value)
                except (TypeError, ValueError):
                    requested = None
    if requested is None:
        requested = 50

    hard_cap = int(
        state_settings.get("autopilot_max_symbols_scan", settings.autopilot_max_symbols_scan)
    )
    return clamp_scan_symbols(
        settings=settings,
        requested=max(1, int(requested)),
        hard_cap=max(1, hard_cap),
    )


def _resolve_max_runtime_seconds(
    payload: dict[str, Any],
    state_settings: dict[str, Any],
    settings: Settings,
) -> int:
    explicit = payload.get("max_runtime_seconds")
    if isinstance(explicit, int):
        requested = explicit
    else:
        try:
            requested = int(explicit)
        except (TypeError, ValueError):
            requested = None
    if requested is None:
        requested = int(
            state_settings.get(
                "autopilot_max_runtime_seconds", settings.autopilot_max_runtime_seconds
            )
        )
    hard_cap = int(
        state_settings.get("autopilot_max_runtime_seconds", settings.autopilot_max_runtime_seconds)
    )
    effective = max(1, min(max(1, int(requested)), max(1, hard_cap)))
    return clamp_job_timeout_seconds(settings=settings, requested=effective)


def _normalize_allowed_sides(state_settings: dict[str, Any], settings: Settings) -> set[str]:
    value = state_settings.get("allowed_sides", settings.allowed_sides)
    sides: list[str] = []
    if isinstance(value, list):
        sides = [str(item).upper().strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        sides = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not sides:
        sides = ["BUY"]
    return set(sides)


def _parse_cutoff(cutoff: str | None, default_cutoff: str) -> time:
    raw = cutoff if isinstance(cutoff, str) and cutoff.strip() else default_cutoff
    try:
        hour, minute = [int(part) for part in raw.split(":", maxsplit=1)]
        return time(hour=max(0, min(23, hour)), minute=max(0, min(59, minute)))
    except Exception:  # noqa: BLE001
        return time(15, 20)


def _asof_datetime(payload: dict[str, Any]) -> datetime:
    asof = _parse_asof(payload.get("asof"))
    if isinstance(asof, date) and not isinstance(asof, datetime):
        return datetime.combine(asof, time(0, 0), tzinfo=timezone.utc)
    if isinstance(asof, datetime):
        return asof if asof.tzinfo is not None else asof.replace(tzinfo=timezone.utc)
    return _utc_now()


def _is_squareoff_due(asof: datetime, cutoff: time) -> bool:
    return asof.astimezone(IST_ZONE).time() >= cutoff


def _adjust_qty_for_lot(qty: int, lot_size: int) -> int:
    if lot_size <= 1:
        return qty
    return (qty // lot_size) * lot_size


def _reason_histogram(rows: list[dict[str, Any]], key: str = "reason") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get(key, "")).strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _selection_reason_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("instrument_choice_reason", row.get("selection_reason", ""))).strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _positions_notional(positions: list[PaperPosition]) -> float:
    total = 0.0
    for position in positions:
        total += float(position.qty) * float(position.avg_price)
    return total


def _average_holding_days(positions: list[PaperPosition], asof: datetime) -> float:
    if not positions:
        return 0.0
    days: list[float] = []
    for position in positions:
        opened = position.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        delta = asof - opened
        days.append(max(0.0, delta.total_seconds() / 86_400))
    return float(np.mean(days)) if days else 0.0


def _mark_to_market_component(position: PaperPosition, mark: float) -> float:
    if _is_futures_kind(position.instrument_kind):
        margin_reserved = float(getattr(position, "margin_reserved", 0.0) or 0.0)
        if position.side.upper() == "SELL":
            pnl = (position.avg_price - mark) * position.qty
        else:
            pnl = (mark - position.avg_price) * position.qty
        return margin_reserved + pnl
    if position.side.upper() == "SELL":
        return -position.qty * mark
    return position.qty * mark


def _position_exit_condition(position: PaperPosition, mark: float) -> tuple[bool, str]:
    if position.side.upper() == "SELL":
        stop_hit = position.stop_price is not None and mark >= position.stop_price
        target_hit = position.target_price is not None and mark <= position.target_price
    else:
        stop_hit = position.stop_price is not None and mark <= position.stop_price
        target_hit = position.target_price is not None and mark >= position.target_price
    if stop_hit:
        return True, "STOP_HIT"
    if target_hit:
        return True, "EXITED"
    return False, ""


def _close_position(
    *,
    session: Session,
    settings: Settings,
    state_settings: dict[str, Any],
    policy: dict[str, Any],
    state: PaperState,
    position: PaperPosition,
    price: float,
    reason: str,
) -> float:
    side = position.side.upper()
    exit_side = "BUY" if side == "SELL" else "SELL"
    notional = position.qty * price
    qty_lots = max(
        1, int(getattr(position, "qty_lots", max(1, position.qty // max(1, position.lot_size))))
    )
    is_futures = _is_futures_kind(position.instrument_kind)
    cost_override = dict(policy.get("cost_model", {}))
    if side == "SELL" and position.instrument_kind == "EQUITY_CASH":
        cost_override["mode"] = "intraday"
    exit_cost = _transaction_cost(
        notional=notional,
        side=exit_side,
        instrument_kind=position.instrument_kind,
        settings=settings,
        state_settings=state_settings,
        override_cost_model=cost_override,
    )

    if is_futures:
        entry_notional = position.qty * position.avg_price
        margin_reserved = float(getattr(position, "margin_reserved", 0.0) or 0.0)
        pnl = (entry_notional - notional) if side == "SELL" else (notional - entry_notional)
        state.cash += margin_reserved + pnl - exit_cost
    elif side == "SELL":
        state.cash -= notional + exit_cost
    else:
        state.cash += notional - exit_cost

    session.add(
        PaperOrder(
            symbol=position.symbol,
            side=exit_side,
            instrument_kind=position.instrument_kind,
            lot_size=position.lot_size,
            qty_lots=qty_lots,
            qty=position.qty,
            fill_price=price,
            status=reason,
            reason=reason,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
    )
    session.delete(position)
    _log(
        session,
        "paper_exit",
        {
            "symbol": position.symbol,
            "position_side": side,
            "exit_side": exit_side,
            "qty": position.qty,
            "fill_price": price,
            "exit_cost": exit_cost,
            "reason": reason,
            "instrument_kind": position.instrument_kind,
            "qty_lots": qty_lots,
            "margin_released": float(getattr(position, "margin_reserved", 0.0) or 0.0),
        },
    )
    return exit_cost


def _run_paper_step_with_simulator_engine(
    *,
    session: Session,
    settings: Settings,
    state: PaperState,
    state_settings: dict[str, Any],
    payload: dict[str, Any],
    policy: dict[str, Any],
    regime: str,
    asof_dt: datetime,
    base_risk_per_trade: float,
    base_max_positions: int,
    mark_prices: dict[str, Any],
    selected_signals: list[dict[str, Any]],
    skipped_signals: list[dict[str, Any]],
    generated_meta: SignalGenerationResult,
    generated_signals_count: int,
    signals_source: str,
    safe_mode_active: bool,
    safe_mode_action: str,
    safe_mode_reason: str | None,
    no_trade_snapshot: dict[str, Any],
    confidence_gate_snapshot: dict[str, Any],
    quality_status: str | None,
    quality_warn_summary: list[dict[str, Any]],
    risk_overlay: dict[str, Any],
    cost_spike_active: bool,
    cost_spike_meta: dict[str, Any],
    scan_guard_active: bool,
    scan_guard_meta: dict[str, Any],
    resolved_bundle_id: int | None,
    resolved_dataset_id: int | None,
    resolved_timeframes: list[str],
    positions_before: list[PaperPosition],
    positions_before_by_id: dict[int, PaperPosition],
    position_ids_before: set[int],
    order_ids_before: set[int],
    equity_before: float,
    cash_before: float,
    drawdown_before: float,
    mtm_before: float,
) -> dict[str, Any]:
    seed = _resolve_seed(payload, policy, settings)
    sim_execution = execute_paper_step_with_simulator(
        session=session,
        settings=settings,
        state=state,
        state_settings=state_settings,
        policy=policy,
        asof_dt=asof_dt,
        selected_signals=selected_signals,
        mark_prices={str(key): float(value) for key, value in mark_prices.items()},
        open_positions=positions_before,
        seed=seed,
        risk_overlay=risk_overlay,
    )

    executed_signals = list(sim_execution.executed_signals)
    skipped_signals.extend(sim_execution.skipped_signals)
    executed_count = len(executed_signals)
    entry_cost_total = float(sim_execution.entry_cost_total)
    exit_cost_total = float(sim_execution.exit_cost_total)
    entry_slippage_cost_total = float(sim_execution.entry_slippage_cost_total)
    exit_slippage_cost_total = float(sim_execution.exit_slippage_cost_total)
    traded_notional_total = float(sim_execution.traded_notional_total)

    state.peak_equity = max(state.peak_equity, state.equity)
    state.drawdown = (state.equity / state.peak_equity - 1.0) if state.peak_equity > 0 else 0.0

    dd_limit = float(state.settings_json.get("kill_switch_dd", settings.kill_switch_drawdown))
    if state.drawdown <= -dd_limit:
        state.kill_switch_active = True
        state.cooldown_days_left = settings.kill_switch_cooldown_days
        _log(session, "kill_switch_activated", {"drawdown": state.drawdown, "limit": dd_limit})

    for signal in executed_signals:
        _log(
            session,
            "signal_selected",
            {
                "symbol": signal.get("symbol"),
                "underlying_symbol": signal.get("underlying_symbol", signal.get("symbol")),
                "template": signal.get("template"),
                "side": signal.get("side"),
                "instrument_kind": signal.get("instrument_kind"),
                "instrument_choice_reason": signal.get("instrument_choice_reason", "provided"),
                "qty": signal.get("qty"),
                "qty_lots": signal.get("qty_lots"),
                "margin_reserved": signal.get("margin_reserved"),
                "fill_price": signal.get("fill_price"),
                "signal_strength": float(signal.get("signal_strength", 0.0)),
                "selection_reason": policy.get("selection_reason"),
                "policy_mode": policy.get("mode"),
                "policy_id": policy.get("policy_id"),
                "must_exit_by_eod": bool(signal.get("must_exit_by_eod", False)),
                "engine_version": sim_execution.metadata.get("engine_version"),
                "data_digest": sim_execution.metadata.get("data_digest"),
                "seed": sim_execution.metadata.get("seed"),
            },
        )

    for skip in skipped_signals:
        _log(session, "signal_skipped", skip)
        _log(session, "paper_skip", skip)

    session.add(state)
    session.commit()
    session.refresh(state)

    live_positions = get_positions(session)
    orders_after = get_orders(session)
    position_ids_after = {int(row.id) for row in live_positions if row.id is not None}
    order_ids_after = {int(row.id) for row in orders_after if row.id is not None}
    new_position_ids = sorted(position_ids_after - position_ids_before)
    closed_position_ids = sorted(position_ids_before - position_ids_after)
    new_order_ids = sorted(order_ids_after - order_ids_before)
    closed_position_symbols = [
        positions_before_by_id[item].symbol
        for item in closed_position_ids
        if item in positions_before_by_id
    ]

    selected_reason_histogram = _selection_reason_histogram(executed_signals)
    skipped_reason_histogram = _reason_histogram(skipped_signals)
    total_cost = float(entry_cost_total + exit_cost_total)
    realized_pnl = float(state.cash - cash_before)
    mtm = 0.0
    for position in live_positions:
        mark = float(mark_prices.get(position.symbol, position.avg_price))
        mtm += _mark_to_market_component(position, mark)
    unrealized_pnl = float(mtm - mtm_before)
    net_pnl = float(state.equity - equity_before)
    gross_pnl = float(net_pnl + total_cost)
    turnover = float(traded_notional_total / max(1e-9, equity_before))
    exposure = float(_positions_notional(live_positions) / max(1e-9, state.equity))
    avg_holding_days = _average_holding_days(live_positions, asof_dt)

    cost_summary = {
        "entry_cost_total": float(entry_cost_total),
        "exit_cost_total": float(exit_cost_total),
        "total_cost": total_cost,
        "entry_slippage_cost": float(entry_slippage_cost_total),
        "exit_slippage_cost": float(exit_slippage_cost_total),
        "cost_model_enabled": bool(
            policy.get("cost_model", {}).get(
                "enabled",
                state_settings.get("cost_model_enabled", settings.cost_model_enabled),
            )
        ),
        "cost_mode": str(
            policy.get("cost_model", {}).get(
                "mode",
                state_settings.get("cost_mode", settings.cost_mode),
            )
        ),
    }
    ensemble_payload = (
        policy.get("ensemble", {}) if isinstance(policy.get("ensemble"), dict) else {}
    )
    if ensemble_payload:
        actual_counts: dict[str, int] = {}
        for signal in executed_signals:
            try:
                source_policy_id = int(signal.get("source_policy_id") or 0)
            except (TypeError, ValueError):
                source_policy_id = 0
            if source_policy_id <= 0:
                continue
            key = str(source_policy_id)
            actual_counts[key] = int(actual_counts.get(key, 0)) + 1
        ensemble_payload = {**ensemble_payload, "selected_counts_by_policy": actual_counts}

    run_summary = {
        "execution_mode": "LIVE",
        "shadow_only": False,
        "live_state_mutated": True,
        "policy_mode": policy.get("mode"),
        "policy_selection_reason": policy.get("selection_reason"),
        "policy_status": policy.get("policy_status"),
        "health_status": policy.get("health_status"),
        "ensemble_active": bool(ensemble_payload),
        "ensemble_id": ensemble_payload.get("id"),
        "ensemble_name": ensemble_payload.get("name"),
        "ensemble_regime_used": ensemble_payload.get("regime_used"),
        "ensemble_weights_source": ensemble_payload.get("weights_source"),
        "ensemble_risk_budget_by_policy": dict(
            ensemble_payload.get("risk_budget_by_policy", {})
            if isinstance(ensemble_payload.get("risk_budget_by_policy", {}), dict)
            else {}
        ),
        "ensemble_selected_counts_by_policy": dict(
            ensemble_payload.get("selected_counts_by_policy", {})
            if isinstance(ensemble_payload.get("selected_counts_by_policy", {}), dict)
            else {}
        ),
        "safe_mode_active": bool(safe_mode_active),
        "safe_mode_action": safe_mode_action,
        "safe_mode_reason": safe_mode_reason,
        "no_trade": dict(no_trade_snapshot or {}),
        "no_trade_triggered": bool((no_trade_snapshot or {}).get("triggered", False)),
        "no_trade_reasons": list((no_trade_snapshot or {}).get("reasons", [])),
        "confidence_gate": dict(confidence_gate_snapshot or {}),
        "data_quality_status": quality_status,
        "data_quality_warn_summary": quality_warn_summary,
        "cost_ratio_spike_active": bool(cost_spike_active),
        "cost_ratio_spike_meta": cost_spike_meta,
        "scan_guard_active": bool(scan_guard_active),
        "scan_guard_meta": scan_guard_meta,
        "signals_source": signals_source,
        "bundle_id": resolved_bundle_id,
        "dataset_id": resolved_dataset_id,
        "timeframes": resolved_timeframes,
        "scan_truncated": bool(generated_meta.scan_truncated),
        "scanned_symbols": int(generated_meta.scanned_symbols),
        "evaluated_candidates": int(generated_meta.evaluated_candidates),
        "total_symbols": int(generated_meta.total_symbols),
        "generated_signals_count": int(generated_signals_count),
        "selected_signals_count": int(executed_count),
        "skipped_signals_count": int(len(skipped_signals)),
        "positions_before": len(positions_before),
        "positions_after": len(live_positions),
        "positions_opened": len(new_position_ids),
        "positions_closed": len(closed_position_ids),
        "new_position_ids": new_position_ids,
        "closed_position_ids": closed_position_ids,
        "closed_position_symbols": closed_position_symbols,
        "new_order_ids": new_order_ids,
        "selected_signals": [
            {
                "symbol": str(item.get("symbol", "")),
                "side": str(item.get("side", "")),
                "instrument_kind": str(item.get("instrument_kind", "")),
                "selection_reason": str(item.get("instrument_choice_reason", "provided")),
            }
            for item in executed_signals
        ],
        "selected_reason_histogram": selected_reason_histogram,
        "skipped_reason_histogram": skipped_reason_histogram,
        "risk_scale": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "risk_scale", risk_overlay.get("risk_scale", 1.0)
            )
        ),
        "confidence_risk_scale": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "confidence_risk_scale",
                risk_overlay.get("confidence_risk_scale", 1.0),
            )
        ),
        "effective_risk_scale": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "effective_risk_scale",
                float(risk_overlay.get("risk_scale", 1.0))
                * float(risk_overlay.get("confidence_risk_scale", 1.0)),
            )
        ),
        "realized_vol": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "realized_vol", risk_overlay.get("realized_vol", 0.0)
            )
        ),
        "target_vol": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "target_vol", risk_overlay.get("target_vol", 0.0)
            )
        ),
        "caps_applied": dict(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "caps", risk_overlay.get("caps", {})
            )
        ),
        "equity_before": equity_before,
        "equity_after": float(state.equity),
        "cash_before": cash_before,
        "cash_after": float(state.cash),
        "drawdown_before": drawdown_before,
        "drawdown": float(state.drawdown),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "trade_count": int(len(new_position_ids) + len(closed_position_ids)),
        "turnover": turnover,
        "exposure": exposure,
        "avg_holding_days": avg_holding_days,
        "kill_switch_active": bool(state.kill_switch_active),
        "engine_version": sim_execution.metadata.get("engine_version"),
        "data_digest": sim_execution.metadata.get("data_digest"),
        "seed": sim_execution.metadata.get("seed"),
        "paper_engine": "simulator",
    }
    run_summary["result_digest"] = _result_digest(run_summary, cost_summary)
    policy_id_value: int | None = None
    try:
        if policy.get("policy_id") is not None:
            policy_id_value = int(policy["policy_id"])
    except (TypeError, ValueError):
        policy_id_value = None
    run_row = PaperRun(
        bundle_id=resolved_bundle_id,
        policy_id=policy_id_value,
        asof_ts=asof_dt,
        mode="LIVE",
        regime=regime,
        signals_source=signals_source,
        generated_signals_count=generated_signals_count,
        selected_signals_count=executed_count,
        skipped_signals_count=len(skipped_signals),
        scanned_symbols=int(generated_meta.scanned_symbols),
        evaluated_candidates=int(generated_meta.evaluated_candidates),
        scan_truncated=bool(generated_meta.scan_truncated),
        summary_json=run_summary,
        cost_summary_json=cost_summary,
    )
    session.add(run_row)
    session.commit()
    session.refresh(run_row)
    _emit_digest_mismatch_if_any(session=session, run_row=run_row, run_summary=run_summary)
    session.commit()

    health_short_payload: dict[str, Any] | None = None
    health_long_payload: dict[str, Any] | None = None
    health_action_payload: dict[str, Any] | None = None

    active_policy: Policy | None = None
    if policy_id_value is not None:
        active_policy = session.get(Policy, policy_id_value)

    if active_policy is not None and policy.get("mode") == "policy":
        short_window = max(
            5,
            int(
                state_settings.get(
                    "health_window_days_short",
                    settings.health_window_days_short,
                )
            ),
        )
        long_window = max(
            short_window,
            int(
                state_settings.get(
                    "health_window_days_long",
                    settings.health_window_days_long,
                )
            ),
        )
        health_short = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=active_policy,
            window_days=short_window,
            asof_date=asof_dt.date(),
            refresh=True,
            overrides=state_settings,
        )
        health_long = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=active_policy,
            window_days=long_window,
            asof_date=asof_dt.date(),
            refresh=True,
            overrides=state_settings,
        )
        action_source: PolicyHealthSnapshot = (
            health_long if health_long.status == DEGRADED else health_short
        )
        health_action_payload = apply_policy_health_actions(
            session,
            settings=settings,
            policy=active_policy,
            snapshot=action_source,
            overrides=state_settings,
        )
        _log(
            session,
            "policy_health_snapshot",
            {
                "policy_id": active_policy.id,
                "short_window_days": short_window,
                "short_status": health_short.status,
                "short_reasons": health_short.reasons_json,
                "long_window_days": long_window,
                "long_status": health_long.status,
                "long_reasons": health_long.reasons_json,
            },
        )
        _log(
            session,
            "policy_health_action",
            {
                "policy_id": active_policy.id,
                "status": action_source.status,
                "action": health_action_payload.get("action"),
                "risk_scale_override": health_action_payload.get("risk_scale_override"),
                "policy_status": health_action_payload.get("policy_status"),
                "reasons": action_source.reasons_json,
            },
        )

        action_name = str(health_action_payload.get("action", "NONE")).upper()
        action_policy_status = str(
            health_action_payload.get("policy_status", policy_status(active_policy))
        ).upper()
        if action_policy_status in {PAUSED, RETIRED}:
            fallback = select_fallback_policy(
                session,
                current_policy_id=int(active_policy.id),
                regime=regime,
            )
            if fallback is not None:
                merged_state = dict(state.settings_json or {})
                if str(merged_state.get("paper_mode", "strategy")) == "policy" and int(
                    merged_state.get("active_policy_id") or 0
                ) == int(active_policy.id):
                    merged_state["active_policy_id"] = int(fallback.id)
                    merged_state["active_policy_name"] = fallback.name
                    state.settings_json = merged_state
                    session.add(state)
                    _log(
                        session,
                        "policy_fallback_selected",
                        {
                            "from_policy_id": active_policy.id,
                            "to_policy_id": fallback.id,
                            "from_status": action_policy_status,
                            "reason": "policy_degraded_or_paused",
                            "regime": regime,
                        },
                    )
                    session.commit()
                    session.refresh(state)
            elif action_name != "NONE":
                _log(
                    session,
                    "policy_fallback_unavailable",
                    {
                        "policy_id": active_policy.id,
                        "status": action_policy_status,
                        "regime": regime,
                        "reason": "no_fallback_policy",
                    },
                )
        session.commit()
        health_short_payload = health_short.model_dump()
        health_long_payload = health_long.model_dump()

    generated_report_id: int | None = None
    auto_report_enabled = bool(
        state_settings.get("reports_auto_generate_daily", settings.reports_auto_generate_daily)
    )
    if auto_report_enabled:
        report_row = generate_daily_report(
            session=session,
            settings=settings,
            report_date=asof_dt.date(),
            bundle_id=resolved_bundle_id,
            policy_id=run_row.policy_id,
            overwrite=True,
        )
        generated_report_id = int(report_row.id) if report_row.id is not None else None

    effective_context = _paper_effective_context(
        session=session,
        settings=settings,
        state_settings=state_settings,
        bundle_id=resolved_bundle_id,
        timeframe=(resolved_timeframes[0] if resolved_timeframes else "1d"),
        asof_dt=asof_dt,
        provider_stage_status=(
            str(payload.get("provider_stage_status"))
            if payload.get("provider_stage_status") is not None
            else None
        ),
        confidence_gate_snapshot=confidence_gate_snapshot,
        data_digest=(
            str(sim_execution.metadata.get("data_digest"))
            if sim_execution.metadata.get("data_digest") is not None
            else None
        ),
        engine_version=(
            str(sim_execution.metadata.get("engine_version"))
            if sim_execution.metadata.get("engine_version") is not None
            else None
        ),
        seed=(
            int(sim_execution.metadata.get("seed"))
            if sim_execution.metadata.get("seed") is not None
            else None
        ),
        notes=["paper_run_step"],
    )

    return jsonable_encoder(
        {
            "status": "ok",
            "regime": regime,
            "policy": policy,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "policy_status": policy.get("policy_status"),
            "policy_health_status": policy.get("health_status"),
            "policy_health_reasons": policy.get("health_reasons", []),
            "ensemble": ensemble_payload,
            "risk_scaled": bool(
                float(policy["risk_per_trade"]) < base_risk_per_trade
                or int(policy["max_positions"]) < base_max_positions
                or float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "effective_risk_scale",
                        float(risk_overlay.get("risk_scale", 1.0))
                        * float(risk_overlay.get("confidence_risk_scale", 1.0)),
                    )
                )
                < 0.999
            ),
            "paper_run_id": run_row.id,
            "signals_source": signals_source,
            "generated_signals_count": generated_signals_count,
            "selected_signals_count": executed_count,
            "selected_signals": executed_signals,
            "skipped_signals": skipped_signals,
            "safe_mode": {
                "active": bool(safe_mode_active),
                "action": safe_mode_action,
                "reason": safe_mode_reason,
                "status": quality_status,
                "warnings": quality_warn_summary,
            },
            "no_trade": dict(no_trade_snapshot or {}),
            "confidence_gate": dict(confidence_gate_snapshot or {}),
            "guardrails": {
                "cost_ratio_spike_active": bool(cost_spike_active),
                "cost_ratio_spike_meta": cost_spike_meta,
                "scan_guard_active": bool(scan_guard_active),
                "scan_guard_meta": scan_guard_meta,
            },
            "scan_truncated": generated_meta.scan_truncated,
            "scanned_symbols": generated_meta.scanned_symbols,
            "evaluated_candidates": generated_meta.evaluated_candidates,
            "total_symbols": generated_meta.total_symbols,
            "bundle_id": resolved_bundle_id,
            "dataset_id": resolved_dataset_id,
            "timeframes": resolved_timeframes,
            "cost_summary": cost_summary,
            "risk_overlay": {
                "risk_scale": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "risk_scale", risk_overlay.get("risk_scale", 1.0)
                    )
                ),
                "effective_risk_scale": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "effective_risk_scale",
                        float(risk_overlay.get("risk_scale", 1.0))
                        * float(risk_overlay.get("confidence_risk_scale", 1.0)),
                    )
                ),
                "confidence_risk_scaling_enabled": bool(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "confidence_risk_scaling_enabled",
                        risk_overlay.get("confidence_risk_scaling_enabled", False),
                    )
                ),
                "confidence_risk_scale": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "confidence_risk_scale",
                        risk_overlay.get("confidence_risk_scale", 1.0),
                    )
                ),
                "confidence_risk_scale_low_threshold": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "confidence_risk_scale_low_threshold",
                        risk_overlay.get("confidence_risk_scale_low_threshold", 0.35),
                    )
                ),
                "realized_vol": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "realized_vol", risk_overlay.get("realized_vol", 0.0)
                    )
                ),
                "target_vol": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "target_vol", risk_overlay.get("target_vol", 0.0)
                    )
                ),
                "caps_applied": dict(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "caps", risk_overlay.get("caps", {})
                    )
                ),
            },
            "engine_version": sim_execution.metadata.get("engine_version"),
            "data_digest": sim_execution.metadata.get("data_digest"),
            "seed": sim_execution.metadata.get("seed"),
            "paper_engine": "simulator",
            "result_digest": run_summary.get("result_digest"),
            "report_id": generated_report_id,
            "health_short": health_short_payload,
            "health_long": health_long_payload,
            "health_action": health_action_payload,
            "state": _dump_model(state),
            "positions": [_dump_model(p) for p in live_positions],
            "orders": [_dump_model(o) for o in orders_after],
            "effective_context": effective_context,
        }
    )


def _run_paper_step_shadow_only(
    *,
    session: Session,
    settings: Settings,
    live_state: PaperState,
    state_settings: dict[str, Any],
    payload: dict[str, Any],
    policy: dict[str, Any],
    regime: str,
    asof_dt: datetime,
    base_risk_per_trade: float,
    base_max_positions: int,
    mark_prices: dict[str, Any],
    selected_signals: list[dict[str, Any]],
    skipped_signals: list[dict[str, Any]],
    generated_meta: SignalGenerationResult,
    generated_signals_count: int,
    signals_source: str,
    safe_mode_active: bool,
    safe_mode_action: str,
    safe_mode_reason: str | None,
    no_trade_snapshot: dict[str, Any],
    confidence_gate_snapshot: dict[str, Any],
    quality_status: str | None,
    quality_warn_summary: list[dict[str, Any]],
    risk_overlay: dict[str, Any],
    cost_spike_active: bool,
    cost_spike_meta: dict[str, Any],
    scan_guard_active: bool,
    scan_guard_meta: dict[str, Any],
    resolved_bundle_id: int | None,
    resolved_dataset_id: int | None,
    resolved_timeframes: list[str],
    live_positions: list[PaperPosition],
    live_orders: list[PaperOrder],
) -> dict[str, Any]:
    seed = _resolve_seed(payload, policy, settings)
    shadow_row = _get_or_create_shadow_state(
        session=session,
        live_state=live_state,
        state_settings=state_settings,
        bundle_id=resolved_bundle_id,
        policy_id=policy.get("policy_id"),
    )
    shadow_before = _shadow_state_snapshot(
        row=shadow_row,
        live_state=live_state,
        state_settings=state_settings,
    )
    shadow_positions_before = list(shadow_before.get("positions", []))
    shadow_orders_before = list(shadow_before.get("orders", []))
    shadow_cash_before = float(shadow_before.get("cash", float(live_state.cash)))
    shadow_equity_before = float(shadow_before.get("equity", float(live_state.equity)))
    shadow_drawdown_before = float(shadow_before.get("drawdown", float(live_state.drawdown)))

    sim_execution = execute_paper_step_with_simulator(
        session=session,
        settings=settings,
        state=shadow_before,
        state_settings=state_settings,
        policy=policy,
        asof_dt=asof_dt,
        selected_signals=selected_signals,
        mark_prices={str(key): float(value) for key, value in mark_prices.items()},
        open_positions=shadow_positions_before,
        seed=seed,
        risk_overlay=risk_overlay,
        persist_live_state=False,
    )

    executed_signals = list(sim_execution.executed_signals)
    skipped_signals.extend(sim_execution.skipped_signals)
    executed_count = len(executed_signals)

    shadow_cash_after = float(sim_execution.cash)
    shadow_equity_after = float(sim_execution.equity)
    shadow_peak_before = float(shadow_before.get("peak_equity", shadow_equity_before))
    shadow_peak_after = max(shadow_peak_before, shadow_equity_after)
    shadow_drawdown_after = (
        shadow_equity_after / shadow_peak_after - 1.0 if shadow_peak_after > 0 else 0.0
    )

    shadow_positions_after = [dict(item) for item in sim_execution.positions_after]
    shadow_orders_after = (
        shadow_orders_before + [dict(item) for item in sim_execution.orders_generated]
    )[-500:]
    shadow_after = {
        **shadow_before,
        "cash": shadow_cash_after,
        "equity": shadow_equity_after,
        "peak_equity": shadow_peak_after,
        "drawdown": shadow_drawdown_after,
        "positions": shadow_positions_after,
        "orders": shadow_orders_after,
    }

    selected_reason_histogram = _selection_reason_histogram(executed_signals)
    skipped_reason_histogram = _reason_histogram(skipped_signals)
    entry_cost_total = float(sim_execution.entry_cost_total)
    exit_cost_total = float(sim_execution.exit_cost_total)
    total_cost = float(entry_cost_total + exit_cost_total)
    realized_pnl = float(shadow_cash_after - shadow_cash_before)
    net_pnl = float(shadow_equity_after - shadow_equity_before)
    unrealized_pnl = float(net_pnl - realized_pnl)
    gross_pnl = float(net_pnl + total_cost)
    turnover = float(sim_execution.traded_notional_total / max(1e-9, shadow_equity_before))
    positions_notional = float(
        sum(
            float(item.get("qty", 0)) * float(item.get("avg_price", 0.0))
            for item in shadow_positions_after
        )
    )
    exposure = float(positions_notional / max(1e-9, shadow_equity_after))
    live_position_ids = {int(row.id) for row in live_positions if row.id is not None}
    live_order_ids = {int(row.id) for row in live_orders if row.id is not None}

    cost_summary = {
        "entry_cost_total": entry_cost_total,
        "exit_cost_total": exit_cost_total,
        "total_cost": total_cost,
        "entry_slippage_cost": float(sim_execution.entry_slippage_cost_total),
        "exit_slippage_cost": float(sim_execution.exit_slippage_cost_total),
        "cost_model_enabled": bool(
            policy.get("cost_model", {}).get(
                "enabled",
                state_settings.get("cost_model_enabled", settings.cost_model_enabled),
            )
        ),
        "cost_mode": str(
            policy.get("cost_model", {}).get(
                "mode",
                state_settings.get("cost_mode", settings.cost_mode),
            )
        ),
    }
    ensemble_payload = (
        policy.get("ensemble", {}) if isinstance(policy.get("ensemble"), dict) else {}
    )

    run_summary = {
        "execution_mode": "SHADOW",
        "shadow_only": True,
        "shadow_note": "Shadow-only: no live state mutation; simulated trades shown for monitoring.",
        "live_state_mutated": False,
        "policy_mode": policy.get("mode"),
        "policy_selection_reason": policy.get("selection_reason"),
        "policy_status": policy.get("policy_status"),
        "health_status": policy.get("health_status"),
        "ensemble_active": bool(ensemble_payload),
        "ensemble_id": ensemble_payload.get("id"),
        "ensemble_name": ensemble_payload.get("name"),
        "ensemble_regime_used": ensemble_payload.get("regime_used"),
        "ensemble_weights_source": ensemble_payload.get("weights_source"),
        "ensemble_risk_budget_by_policy": dict(
            ensemble_payload.get("risk_budget_by_policy", {})
            if isinstance(ensemble_payload.get("risk_budget_by_policy", {}), dict)
            else {}
        ),
        "ensemble_selected_counts_by_policy": dict(
            ensemble_payload.get("selected_counts_by_policy", {})
            if isinstance(ensemble_payload.get("selected_counts_by_policy", {}), dict)
            else {}
        ),
        "safe_mode_active": bool(safe_mode_active),
        "safe_mode_action": safe_mode_action,
        "safe_mode_reason": safe_mode_reason,
        "no_trade": dict(no_trade_snapshot or {}),
        "no_trade_triggered": bool((no_trade_snapshot or {}).get("triggered", False)),
        "no_trade_reasons": list((no_trade_snapshot or {}).get("reasons", [])),
        "confidence_gate": dict(confidence_gate_snapshot or {}),
        "data_quality_status": quality_status,
        "data_quality_warn_summary": quality_warn_summary,
        "cost_ratio_spike_active": bool(cost_spike_active),
        "cost_ratio_spike_meta": cost_spike_meta,
        "scan_guard_active": bool(scan_guard_active),
        "scan_guard_meta": scan_guard_meta,
        "signals_source": signals_source,
        "bundle_id": resolved_bundle_id,
        "dataset_id": resolved_dataset_id,
        "timeframes": resolved_timeframes,
        "scan_truncated": bool(generated_meta.scan_truncated),
        "scanned_symbols": int(generated_meta.scanned_symbols),
        "evaluated_candidates": int(generated_meta.evaluated_candidates),
        "total_symbols": int(generated_meta.total_symbols),
        "generated_signals_count": int(generated_signals_count),
        "selected_signals_count": int(executed_count),
        "skipped_signals_count": int(len(skipped_signals)),
        "positions_before": len(shadow_positions_before),
        "positions_after": len(shadow_positions_after),
        "positions_opened": max(0, len(shadow_positions_after) - len(shadow_positions_before)),
        "positions_closed": max(0, len(shadow_positions_before) - len(shadow_positions_after)),
        "new_order_ids": [],
        "selected_signals": [
            {
                "symbol": str(item.get("symbol", "")),
                "side": str(item.get("side", "")),
                "instrument_kind": str(item.get("instrument_kind", "")),
                "selection_reason": str(item.get("instrument_choice_reason", "provided")),
            }
            for item in executed_signals
        ],
        "selected_reason_histogram": selected_reason_histogram,
        "skipped_reason_histogram": skipped_reason_histogram,
        "risk_scale": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "risk_scale", risk_overlay.get("risk_scale", 1.0)
            )
        ),
        "confidence_risk_scale": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "confidence_risk_scale",
                risk_overlay.get("confidence_risk_scale", 1.0),
            )
        ),
        "effective_risk_scale": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "effective_risk_scale",
                float(risk_overlay.get("risk_scale", 1.0))
                * float(risk_overlay.get("confidence_risk_scale", 1.0)),
            )
        ),
        "realized_vol": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "realized_vol", risk_overlay.get("realized_vol", 0.0)
            )
        ),
        "target_vol": float(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "target_vol", risk_overlay.get("target_vol", 0.0)
            )
        ),
        "caps_applied": dict(
            (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                "caps", risk_overlay.get("caps", {})
            )
        ),
        "equity_before": shadow_equity_before,
        "equity_after": shadow_equity_after,
        "cash_before": shadow_cash_before,
        "cash_after": shadow_cash_after,
        "drawdown_before": shadow_drawdown_before,
        "drawdown": shadow_drawdown_after,
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "trade_count": int(
            len(sim_execution.orders_generated) + len(sim_execution.trades_generated)
        ),
        "turnover": turnover,
        "exposure": exposure,
        "avg_holding_days": 0.0,
        "kill_switch_active": bool(shadow_after.get("kill_switch_active", False)),
        "engine_version": sim_execution.metadata.get("engine_version"),
        "data_digest": sim_execution.metadata.get("data_digest"),
        "seed": sim_execution.metadata.get("seed"),
        "paper_engine": "simulator_shadow",
    }
    run_summary["result_digest"] = _result_digest(run_summary, cost_summary)

    policy_id_value: int | None = None
    try:
        if policy.get("policy_id") is not None:
            policy_id_value = int(policy["policy_id"])
    except (TypeError, ValueError):
        policy_id_value = None

    run_row = PaperRun(
        bundle_id=resolved_bundle_id,
        policy_id=policy_id_value,
        asof_ts=asof_dt,
        mode="SHADOW",
        regime=regime,
        signals_source=signals_source,
        generated_signals_count=generated_signals_count,
        selected_signals_count=executed_count,
        skipped_signals_count=len(skipped_signals),
        scanned_symbols=int(generated_meta.scanned_symbols),
        evaluated_candidates=int(generated_meta.evaluated_candidates),
        scan_truncated=bool(generated_meta.scan_truncated),
        summary_json=run_summary,
        cost_summary_json=cost_summary,
    )
    session.add(run_row)
    session.commit()
    session.refresh(run_row)

    _save_shadow_state_snapshot(
        session=session,
        row=shadow_row,
        snapshot=shadow_after,
        last_run_id=int(run_row.id) if run_row.id is not None else None,
    )
    _emit_digest_mismatch_if_any(session=session, run_row=run_row, run_summary=run_summary)
    _log(
        session,
        "safe_mode_shadow_run_completed",
        {
            "paper_run_id": run_row.id,
            "bundle_id": resolved_bundle_id,
            "policy_id": policy_id_value,
            "selected_signals_count": executed_count,
            "generated_signals_count": generated_signals_count,
            "safe_mode_reason": safe_mode_reason,
        },
    )
    emit_operate_event(
        session,
        severity="WARN",
        category="EXECUTION",
        message="safe_mode_shadow_run_completed",
        details={
            "paper_run_id": run_row.id,
            "bundle_id": resolved_bundle_id,
            "policy_id": policy_id_value,
            "selected_signals_count": executed_count,
            "generated_signals_count": generated_signals_count,
            "safe_mode_reason": safe_mode_reason,
        },
        correlation_id=str(run_row.id),
    )
    session.commit()
    session.refresh(shadow_row)

    generated_report_id: int | None = None
    auto_report_enabled = bool(
        state_settings.get("reports_auto_generate_daily", settings.reports_auto_generate_daily)
    )
    if auto_report_enabled:
        report_row = generate_daily_report(
            session=session,
            settings=settings,
            report_date=asof_dt.date(),
            bundle_id=resolved_bundle_id,
            policy_id=run_row.policy_id,
            overwrite=True,
        )
        generated_report_id = int(report_row.id) if report_row.id is not None else None

    live_positions_after = get_positions(session)
    live_orders_after = get_orders(session)
    live_state_after = session.get(PaperState, 1) or live_state
    live_state_unchanged = (
        float(live_state_after.cash) == float(live_state.cash)
        and float(live_state_after.equity) == float(live_state.equity)
        and {int(row.id) for row in live_positions_after if row.id is not None} == live_position_ids
        and {int(row.id) for row in live_orders_after if row.id is not None} == live_order_ids
    )

    effective_context = _paper_effective_context(
        session=session,
        settings=settings,
        state_settings=state_settings,
        bundle_id=resolved_bundle_id,
        timeframe=(resolved_timeframes[0] if resolved_timeframes else "1d"),
        asof_dt=asof_dt,
        provider_stage_status=(
            str(payload.get("provider_stage_status"))
            if payload.get("provider_stage_status") is not None
            else None
        ),
        confidence_gate_snapshot=confidence_gate_snapshot,
        data_digest=(
            str(sim_execution.metadata.get("data_digest"))
            if sim_execution.metadata.get("data_digest") is not None
            else None
        ),
        engine_version=(
            str(sim_execution.metadata.get("engine_version"))
            if sim_execution.metadata.get("engine_version") is not None
            else None
        ),
        seed=(
            int(sim_execution.metadata.get("seed"))
            if sim_execution.metadata.get("seed") is not None
            else None
        ),
        notes=["paper_run_step", "shadow_mode"],
    )

    return jsonable_encoder(
        {
            "status": "ok",
            "regime": regime,
            "policy": policy,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "policy_status": policy.get("policy_status"),
            "policy_health_status": policy.get("health_status"),
            "policy_health_reasons": policy.get("health_reasons", []),
            "ensemble": ensemble_payload,
            "risk_scaled": bool(
                float(policy["risk_per_trade"]) < base_risk_per_trade
                or int(policy["max_positions"]) < base_max_positions
                or float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "effective_risk_scale",
                        float(risk_overlay.get("risk_scale", 1.0))
                        * float(risk_overlay.get("confidence_risk_scale", 1.0)),
                    )
                )
                < 0.999
            ),
            "paper_run_id": run_row.id,
            "signals_source": signals_source,
            "generated_signals_count": generated_signals_count,
            "selected_signals_count": executed_count,
            "selected_signals": executed_signals,
            "skipped_signals": skipped_signals,
            "safe_mode": {
                "active": bool(safe_mode_active),
                "action": safe_mode_action,
                "reason": safe_mode_reason,
                "status": quality_status,
                "warnings": quality_warn_summary,
            },
            "no_trade": dict(no_trade_snapshot or {}),
            "confidence_gate": dict(confidence_gate_snapshot or {}),
            "guardrails": {
                "cost_ratio_spike_active": bool(cost_spike_active),
                "cost_ratio_spike_meta": cost_spike_meta,
                "scan_guard_active": bool(scan_guard_active),
                "scan_guard_meta": scan_guard_meta,
            },
            "scan_truncated": generated_meta.scan_truncated,
            "scanned_symbols": generated_meta.scanned_symbols,
            "evaluated_candidates": generated_meta.evaluated_candidates,
            "total_symbols": generated_meta.total_symbols,
            "bundle_id": resolved_bundle_id,
            "dataset_id": resolved_dataset_id,
            "timeframes": resolved_timeframes,
            "cost_summary": cost_summary,
            "risk_overlay": {
                "risk_scale": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "risk_scale", risk_overlay.get("risk_scale", 1.0)
                    )
                ),
                "effective_risk_scale": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "effective_risk_scale",
                        float(risk_overlay.get("risk_scale", 1.0))
                        * float(risk_overlay.get("confidence_risk_scale", 1.0)),
                    )
                ),
                "confidence_risk_scaling_enabled": bool(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "confidence_risk_scaling_enabled",
                        risk_overlay.get("confidence_risk_scaling_enabled", False),
                    )
                ),
                "confidence_risk_scale": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "confidence_risk_scale",
                        risk_overlay.get("confidence_risk_scale", 1.0),
                    )
                ),
                "confidence_risk_scale_low_threshold": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "confidence_risk_scale_low_threshold",
                        risk_overlay.get("confidence_risk_scale_low_threshold", 0.35),
                    )
                ),
                "realized_vol": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "realized_vol", risk_overlay.get("realized_vol", 0.0)
                    )
                ),
                "target_vol": float(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "target_vol", risk_overlay.get("target_vol", 0.0)
                    )
                ),
                "caps_applied": dict(
                    (sim_execution.metadata.get("risk_overlay", {}) or {}).get(
                        "caps", risk_overlay.get("caps", {})
                    )
                ),
            },
            "engine_version": sim_execution.metadata.get("engine_version"),
            "data_digest": sim_execution.metadata.get("data_digest"),
            "seed": sim_execution.metadata.get("seed"),
            "paper_engine": "simulator_shadow",
            "execution_mode": "SHADOW",
            "live_state_mutated": not live_state_unchanged,
            "shadow_note": "Shadow-only: no live state mutation; simulated trades shown for monitoring.",
            "result_digest": run_summary.get("result_digest"),
            "report_id": generated_report_id,
            "state": _dump_model(live_state_after),
            "positions": [_dump_model(p) for p in live_positions_after],
            "orders": [_dump_model(o) for o in live_orders_after],
            "shadow_state": shadow_row.state_json,
            "simulated_positions": shadow_positions_after,
            "simulated_orders": sim_execution.orders_generated,
            "simulated_trades": sim_execution.trades_generated,
            "effective_context": effective_context,
        }
    )


def run_paper_step(
    session: Session,
    settings: Settings,
    payload: dict[str, Any],
    store: DataStore | None = None,
) -> dict[str, Any]:
    store = store or DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    state = get_or_create_paper_state(session, settings)
    regime = str(payload.get("regime", "TREND_UP"))
    state_settings = state.settings_json or {}
    base_risk_per_trade = float(state_settings.get("risk_per_trade", settings.risk_per_trade))
    base_max_positions = int(state_settings.get("max_positions", settings.max_positions))
    asof_dt = _asof_datetime(payload)
    policy = _resolve_execution_policy(
        session,
        state,
        settings,
        regime,
        asof_date=asof_dt.date(),
    )
    mark_prices = payload.get("mark_prices", {})

    positions_before = get_positions(session)
    orders_before = get_orders(session)
    positions_before_by_id = {int(row.id): row for row in positions_before if row.id is not None}
    position_ids_before = {int(row.id) for row in positions_before if row.id is not None}
    order_ids_before = {int(row.id) for row in orders_before if row.id is not None}
    equity_before = float(state.equity)
    cash_before = float(state.cash)
    drawdown_before = float(state.drawdown)
    mtm_before = 0.0
    for position in positions_before:
        mark = float(mark_prices.get(position.symbol, position.avg_price))
        mtm_before += _mark_to_market_component(position, mark)

    if state.kill_switch_active:
        _log(session, "kill_switch", {"active": True})
        kill_timeframes = _resolve_timeframes(payload, policy)
        kill_primary_timeframe = kill_timeframes[0] if kill_timeframes else "1d"
        kill_bundle_id = _resolve_bundle_id(session, payload, policy, settings)
        run_row = PaperRun(
            bundle_id=None,
            policy_id=policy.get("policy_id"),
            asof_ts=asof_dt,
            mode="LIVE",
            regime=regime,
            signals_source="provided",
            generated_signals_count=0,
            selected_signals_count=0,
            skipped_signals_count=0,
            scanned_symbols=0,
            evaluated_candidates=0,
            scan_truncated=False,
            summary_json={
                "status": "kill_switch_active",
                "equity_before": equity_before,
                "equity_after": float(state.equity),
                "cash_before": cash_before,
                "cash_after": float(state.cash),
                "drawdown_before": drawdown_before,
                "drawdown": float(state.drawdown),
                "selected_reason_histogram": {},
                "skipped_reason_histogram": {},
                "positions_before": len(positions_before),
                "positions_after": len(positions_before),
                "positions_opened": 0,
                "positions_closed": 0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "net_pnl": 0.0,
                "gross_pnl": 0.0,
                "trade_count": 0,
                "turnover": 0.0,
                "exposure": float(_positions_notional(positions_before) / max(1e-9, state.equity)),
                "avg_holding_days": _average_holding_days(positions_before, asof_dt),
                "kill_switch_active": True,
            },
            cost_summary_json={
                "entry_cost_total": 0.0,
                "exit_cost_total": 0.0,
                "total_cost": 0.0,
                "entry_slippage_cost": 0.0,
                "exit_slippage_cost": 0.0,
            },
        )
        session.add(run_row)
        session.commit()
        session.refresh(run_row)
        effective_context = _paper_effective_context(
            session=session,
            settings=settings,
            state_settings=state_settings,
            bundle_id=kill_bundle_id,
            timeframe=kill_primary_timeframe,
            asof_dt=asof_dt,
            provider_stage_status=(
                str(payload.get("provider_stage_status"))
                if payload.get("provider_stage_status") is not None
                else None
            ),
            confidence_gate_snapshot=None,
            notes=["paper_run_step", "kill_switch_active"],
            store=store,
        )
        return jsonable_encoder(
            {
                "status": "kill_switch_active",
                "regime": regime,
                "policy": policy,
                "policy_mode": policy.get("mode"),
                "policy_selection_reason": policy.get("selection_reason"),
                "signals_source": "provided",
                "generated_signals_count": 0,
                "selected_signals_count": 0,
                "paper_run_id": run_row.id,
                "positions": [_dump_model(p) for p in get_positions(session)],
                "orders": [_dump_model(o) for o in get_orders(session)],
                "effective_context": effective_context,
            }
        )

    provided_signals = [dict(item) for item in payload.get("signals", []) if isinstance(item, dict)]
    signals_source = "provided"
    generated_signals_count = 0
    generated_meta = SignalGenerationResult(
        signals=[],
        scan_truncated=False,
        scanned_symbols=0,
        evaluated_candidates=0,
        total_symbols=0,
    )
    resolved_timeframes: list[str] = _resolve_timeframes(payload, policy)
    resolved_bundle_id: int | None = _resolve_bundle_id(session, payload, policy, settings)
    resolved_dataset_id: int | None = _resolve_dataset_id(
        session,
        payload,
        policy,
        resolved_timeframes,
        settings,
    )
    primary_timeframe = resolved_timeframes[0] if resolved_timeframes else "1d"
    seed = _resolve_seed(payload, policy, settings)

    cost_spike_active, cost_spike_meta = _cost_ratio_spike_active(
        session,
        state_settings=state_settings,
        settings=settings,
    )
    if cost_spike_active:
        risk_scale = float(
            state_settings.get(
                "operate_cost_spike_risk_scale", settings.operate_cost_spike_risk_scale
            )
        )
        current_risk = float(policy.get("risk_per_trade", base_risk_per_trade))
        policy["risk_per_trade"] = max(0.0001, current_risk * max(0.1, risk_scale))
        policy["max_positions"] = max(
            1, min(int(policy.get("max_positions", 1)), base_max_positions)
        )
        policy["selection_reason"] = (
            f"{policy.get('selection_reason', '')} Risk throttled due to cost-ratio spike."
        ).strip()
        _log(
            session,
            "cost_ratio_spike_throttle",
            {
                "meta": cost_spike_meta,
                "risk_scale": risk_scale,
                "risk_per_trade": policy["risk_per_trade"],
            },
        )
        emit_operate_event(
            session,
            severity="WARN",
            category="EXECUTION",
            message="Cost ratio spike detected. Risk throttling applied.",
            details={
                "meta": cost_spike_meta,
                "risk_scale": risk_scale,
                "risk_per_trade": policy["risk_per_trade"],
            },
            correlation_id=str(seed),
        )

    scan_guard_active, scan_guard_meta = _scan_truncation_guard(
        session,
        state_settings=state_settings,
        settings=settings,
    )
    if scan_guard_active:
        emit_operate_event(
            session,
            severity="WARN",
            category="EXECUTION",
            message="Frequent scan truncation detected. Reducing scan size guardrail.",
            details=scan_guard_meta,
            correlation_id=str(seed),
        )

    policy_id_value: int | None = None
    try:
        if policy.get("policy_id") is not None:
            policy_id_value = int(policy["policy_id"])
    except (TypeError, ValueError):
        policy_id_value = None
    risk_overlay = create_portfolio_risk_snapshot(
        session=session,
        settings=settings,
        bundle_id=resolved_bundle_id,
        policy_id=policy_id_value,
        overrides=state_settings,
        asof=asof_dt,
    )
    if bool(risk_overlay.get("enabled")):
        policy["selection_reason"] = (
            f"{policy.get('selection_reason', '')} Risk overlay scale "
            f"{float(risk_overlay.get('risk_scale', 1.0)):.2f} "
            f"(realized vol {float(risk_overlay.get('realized_vol', 0.0)):.2%})."
        ).strip()

    quality_guard = _data_quality_guard(
        session=session,
        settings=settings,
        store=store,
        state_settings=state_settings,
        bundle_id=resolved_bundle_id,
        timeframe=primary_timeframe,
        asof_ts=asof_dt,
        correlation_id=str(seed),
    )
    safe_mode_active = bool(quality_guard.get("active"))
    safe_mode_action = str(quality_guard.get("action", "none"))
    safe_mode_reason = quality_guard.get("safe_mode_reason")
    quality_status = quality_guard.get("status")
    quality_warn_summary = quality_guard.get("warn_summary", [])
    if safe_mode_active:
        _log(
            session,
            "data_quality_fail_safe_mode",
            {
                "reason": safe_mode_reason,
                "action": safe_mode_action,
                "bundle_id": resolved_bundle_id,
                "dataset_id": resolved_dataset_id,
                "timeframe": primary_timeframe,
                "status": quality_status,
            },
        )

    no_trade_snapshot = evaluate_no_trade_gate(
        session,
        settings=settings,
        store=store,
        bundle_id=resolved_bundle_id,
        timeframe=primary_timeframe,
        asof_ts=asof_dt,
        regime=regime,
        overrides=state_settings,
    )
    no_trade_active = bool(no_trade_snapshot.get("triggered"))
    no_trade_reasons = list(no_trade_snapshot.get("reasons", []))
    if no_trade_active:
        _log(
            session,
            "no_trade_gate_triggered",
            {
                "bundle_id": resolved_bundle_id,
                "timeframe": primary_timeframe,
                "regime": regime,
                "reasons": no_trade_reasons,
                "cooldown_remaining": no_trade_snapshot.get("cooldown_remaining", 0),
                "breadth_pct": no_trade_snapshot.get("breadth_pct"),
                "realized_vol": no_trade_snapshot.get("realized_vol"),
                "trend_strength": no_trade_snapshot.get("trend_strength"),
            },
        )

    operate_mode = str(state_settings.get("operate_mode", settings.operate_mode)).strip().lower()
    confidence_gate_snapshot = evaluate_confidence_gate(
        session,
        settings=settings,
        bundle_id=resolved_bundle_id,
        timeframe=primary_timeframe,
        asof_ts=asof_dt,
        operate_mode=operate_mode,
        overrides=state_settings,
        correlation_id=str(seed),
        persist=True,
    )
    confidence_agg_snapshot: dict[str, Any] | None = None
    try:
        if isinstance(resolved_bundle_id, int) and resolved_bundle_id > 0:
            agg_row, _ = upsert_daily_confidence_agg(
                session,
                settings=settings,
                bundle_id=int(resolved_bundle_id),
                timeframe=primary_timeframe,
                trading_date=asof_dt.date(),
                operate_mode=operate_mode,
                overrides=state_settings,
                force=False,
            )
            confidence_agg_snapshot = serialize_daily_confidence_agg(agg_row)
            confidence_gate_snapshot = {
                **dict(confidence_gate_snapshot or {}),
                "id": confidence_agg_snapshot.get("id"),
                "decision": confidence_agg_snapshot.get("decision", confidence_gate_snapshot.get("decision", "PASS")),
                "reasons": confidence_agg_snapshot.get("reasons", confidence_gate_snapshot.get("reasons", [])),
                "summary": {
                    **dict(confidence_gate_snapshot.get("summary", {})),
                    "trading_date": confidence_agg_snapshot.get("trading_date"),
                    "avg_confidence": float(confidence_agg_snapshot.get("avg_confidence", 0.0)),
                    "pct_low_confidence": float(confidence_agg_snapshot.get("pct_low_confidence", 0.0)),
                    "provider_mix": dict(confidence_agg_snapshot.get("provider_counts", {})),
                    "latest_day_source_counts": dict(confidence_agg_snapshot.get("provider_counts", {})),
                    "confidence_risk_scale": float(confidence_agg_snapshot.get("confidence_risk_scale", 1.0)),
                    "threshold_used": dict(confidence_agg_snapshot.get("threshold_used", {})),
                    "days_lookback_used": max(
                        1,
                        int(
                            confidence_agg_snapshot.get(
                                "threshold_used", {}
                            ).get("confidence_gate_lookback_days", 1)
                        ),
                    ),
                    "eligible_symbols": int(confidence_agg_snapshot.get("eligible_symbols_count", 0)),
                },
            }
    except Exception as exc:  # noqa: BLE001
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="confidence_agg_refresh_failed",
            details={
                "bundle_id": resolved_bundle_id,
                "timeframe": primary_timeframe,
                "stage": "paper_run_step",
                "error": str(exc),
            },
            correlation_id=str(seed),
        )

    confidence_gate_decision = str(confidence_gate_snapshot.get("decision", "PASS")).upper()
    confidence_gate_reasons = list(confidence_gate_snapshot.get("reasons", []))
    confidence_gate_summary = (
        dict(confidence_gate_snapshot.get("summary", {}))
        if isinstance(confidence_gate_snapshot.get("summary", {}), dict)
        else {}
    )
    confidence_scaling = resolve_confidence_risk_scaling(
        settings=settings,
        overrides=state_settings,
        avg_confidence=float(confidence_gate_summary.get("avg_confidence", 0.0)),
        hard_floor=float(
            state_settings.get("confidence_gate_hard_floor", settings.confidence_gate_hard_floor)
        ),
        avg_threshold=float(
            state_settings.get(
                "confidence_gate_avg_threshold", settings.confidence_gate_avg_threshold
            )
        ),
    )
    confidence_risk_scale = float(confidence_scaling.get("confidence_risk_scale", 1.0))
    if confidence_gate_decision == DECISION_BLOCK_ENTRIES:
        confidence_risk_scale = 0.0
    confidence_gate_summary["confidence_risk_scale"] = confidence_risk_scale
    confidence_gate_summary["confidence_risk_scaling_enabled"] = bool(
        confidence_scaling.get("enabled", False)
    )
    confidence_gate_summary["confidence_risk_scale_exponent"] = float(
        confidence_scaling.get("exponent", settings.confidence_risk_scale_exponent)
    )
    confidence_gate_summary["confidence_risk_scale_low_threshold"] = float(
        confidence_scaling.get("low_threshold", settings.confidence_risk_scale_low_threshold)
    )
    confidence_gate_snapshot["summary"] = confidence_gate_summary
    confidence_gate_snapshot["confidence_risk_scale"] = float(confidence_risk_scale)
    confidence_gate_snapshot["confidence_risk_scaling_enabled"] = bool(
        confidence_scaling.get("enabled", False)
    )
    confidence_gate_snapshot["aggregate"] = confidence_agg_snapshot
    risk_overlay["confidence_risk_scaling_enabled"] = bool(
        confidence_scaling.get("enabled", False)
    )
    risk_overlay["confidence_risk_scale"] = float(confidence_risk_scale)
    risk_overlay["confidence_risk_scale_exponent"] = float(
        confidence_scaling.get("exponent", settings.confidence_risk_scale_exponent)
    )
    risk_overlay["confidence_risk_scale_low_threshold"] = float(
        confidence_scaling.get("low_threshold", settings.confidence_risk_scale_low_threshold)
    )
    confidence_gate_force_shadow = confidence_gate_decision == DECISION_SHADOW_ONLY
    confidence_gate_block_entries = confidence_gate_decision == DECISION_BLOCK_ENTRIES
    if confidence_gate_decision != "PASS":
        _log(
            session,
            "confidence_gate_triggered",
            {
                "bundle_id": resolved_bundle_id,
                "timeframe": primary_timeframe,
                "decision": confidence_gate_decision,
                "reasons": confidence_gate_reasons,
                "summary": confidence_gate_snapshot.get("summary", {}),
            },
        )

    preferred_ensemble_id: int | None = None
    active_ensemble: PolicyEnsemble | None = None
    paper_mode_setting = str(state_settings.get("paper_mode", "strategy")).strip().lower()
    explicit_policy_override: int | None = None
    state_active_policy_id: int | None = None
    try:
        if payload.get("policy_id") is not None:
            explicit_policy_override = int(payload.get("policy_id"))
    except (TypeError, ValueError):
        explicit_policy_override = None
    try:
        if state_settings.get("active_policy_id") is not None:
            state_active_policy_id = int(state_settings.get("active_policy_id"))
    except (TypeError, ValueError):
        state_active_policy_id = None
    use_ensemble_mode = (
        paper_mode_setting == "policy"
        and explicit_policy_override is None
        and state_active_policy_id is None
    )
    if use_ensemble_mode:
        try:
            if state_settings.get("active_ensemble_id") is not None:
                preferred_ensemble_id = int(state_settings.get("active_ensemble_id"))
        except (TypeError, ValueError):
            preferred_ensemble_id = None
        active_ensemble = get_active_policy_ensemble(
            session,
            bundle_id=int(resolved_bundle_id) if isinstance(resolved_bundle_id, int) else None,
            preferred_ensemble_id=preferred_ensemble_id,
        )
    ensemble_members = (
        list_policy_ensemble_members(
            session,
            ensemble_id=int(active_ensemble.id or 0),
            enabled_only=False,
        )
        if active_ensemble is not None
        else []
    )
    ensemble_regime_weights = (
        list_policy_ensemble_regime_weights(
            session,
            ensemble_id=int(active_ensemble.id or 0),
        )
        if active_ensemble is not None
        else {}
    )
    ensemble_meta: dict[str, Any] | None = None
    pre_skipped_signals: list[dict[str, Any]] = []

    auto_generate = bool(payload.get("auto_generate_signals", False))
    should_generate = auto_generate or (
        (policy.get("mode") == "policy" or use_ensemble_mode) and len(provided_signals) == 0
    )
    if safe_mode_active and safe_mode_action == "exits_only":
        should_generate = False
    if no_trade_active:
        should_generate = False

    if should_generate:
        symbol_scope = _resolve_symbol_scope(payload, policy)
        max_symbols_scan = _resolve_max_symbols_scan(payload, policy, state_settings, settings)
        if scan_guard_active:
            max_symbols_scan = min(max_symbols_scan, int(scan_guard_meta["reduced_to"]))
        max_runtime_seconds = _resolve_max_runtime_seconds(payload, state_settings, settings)
        ranking_weights = policy.get("ranking_weights", {})

        if resolved_bundle_id is None and resolved_dataset_id is None:
            provided_signals = []
            _log(
                session,
                "signals_generated",
                {
                    "mode": "policy_autopilot",
                    "generated_count": 0,
                    "dataset_id": None,
                    "bundle_id": None,
                    "reason": "dataset_or_bundle_not_found",
                },
            )
        else:
            if active_ensemble is not None and ensemble_members:
                enabled_members = [
                    row for row in ensemble_members if bool(row.get("enabled", True))
                ]
                sorted_members = sorted(
                    enabled_members,
                    key=lambda row: int(row.get("policy_id") or 0),
                )
                base_positive_weights = {
                    int(row.get("policy_id") or 0): max(0.0, float(row.get("weight") or 0.0))
                    for row in sorted_members
                    if int(row.get("policy_id") or 0) > 0
                }
                regime_weight_payload = (
                    ensemble_regime_weights.get(str(regime).strip().upper(), {})
                    if isinstance(ensemble_regime_weights, dict)
                    else {}
                )
                regime_positive_weights: dict[int, float] = {}
                if isinstance(regime_weight_payload, dict):
                    for key, value in regime_weight_payload.items():
                        try:
                            policy_id = int(key)
                        except (TypeError, ValueError):
                            continue
                        if policy_id <= 0:
                            continue
                        regime_positive_weights[policy_id] = max(0.0, float(value))
                use_regime_weights = any(
                    weight > 0 and policy_id in base_positive_weights
                    for policy_id, weight in regime_positive_weights.items()
                )
                weight_source = "regime" if use_regime_weights else "base"
                positive_weights = (
                    {
                        policy_id: regime_positive_weights.get(policy_id, 0.0)
                        for policy_id in sorted(base_positive_weights)
                    }
                    if use_regime_weights
                    else dict(base_positive_weights)
                )
                total_weight = float(sum(value for value in positive_weights.values() if value > 0))
                normalized_weights = {
                    policy_id: (value / total_weight if total_weight > 0 else 0.0)
                    for policy_id, value in positive_weights.items()
                }
                aggregated_signals: list[dict[str, Any]] = []
                scan_truncated_any = False
                scanned_symbols_total = 0
                evaluated_candidates_total = 0
                total_symbols_max = 0
                for member in sorted_members:
                    source_policy_id = int(member.get("policy_id") or 0)
                    if source_policy_id <= 0:
                        continue
                    source_policy_name = str(
                        member.get("policy_name") or f"Policy {source_policy_id}"
                    )
                    member_weight = float(member.get("weight") or 0.0)
                    if member_weight <= 0:
                        pre_skipped_signals.append(
                            {
                                "symbol": "",
                                "underlying_symbol": "",
                                "template": "",
                                "side": "BUY",
                                "instrument_kind": "EQUITY_CASH",
                                "policy_mode": "ensemble",
                                "policy_id": source_policy_id,
                                "policy_name": source_policy_name,
                                "source_policy_id": source_policy_id,
                                "source_policy_name": source_policy_name,
                                "reason": "ensemble_weight_zero",
                            }
                        )
                        continue
                    member_policy = _resolve_execution_policy(
                        session,
                        state,
                        settings,
                        regime,
                        policy_override_id=source_policy_id,
                        asof_date=asof_dt.date(),
                    )
                    allowed_templates = list(member_policy.get("allowed_templates") or [])
                    if not allowed_templates:
                        continue
                    member_result = generate_signals_for_policy(
                        session=session,
                        store=store,
                        dataset_id=resolved_dataset_id,
                        bundle_id=resolved_bundle_id,
                        asof=asof_dt,
                        timeframes=resolved_timeframes,
                        allowed_templates=allowed_templates,
                        params_overrides=member_policy.get("params", {}),
                        max_symbols_scan=max_symbols_scan,
                        seed=int(seed) + (source_policy_id * 9973),
                        mode="paper",
                        symbol_scope=symbol_scope,
                        ranking_weights=(
                            member_policy.get("ranking_weights", {})
                            if isinstance(member_policy.get("ranking_weights", {}), dict)
                            else None
                        ),
                        max_runtime_seconds=max_runtime_seconds,
                    )
                    scan_truncated_any = scan_truncated_any or bool(member_result.scan_truncated)
                    scanned_symbols_total += int(member_result.scanned_symbols)
                    evaluated_candidates_total += int(member_result.evaluated_candidates)
                    total_symbols_max = max(total_symbols_max, int(member_result.total_symbols))
                    for signal in member_result.signals:
                        row = dict(signal)
                        row["source_policy_id"] = source_policy_id
                        row["source_policy_name"] = source_policy_name
                        row["ensemble_id"] = int(active_ensemble.id or 0)
                        row["ensemble_name"] = active_ensemble.name
                        row["ensemble_member_weight"] = float(
                            normalized_weights.get(source_policy_id, 0.0)
                        )
                        row["member_required_risk"] = max(
                            0.0,
                            float(
                                member_policy.get(
                                    "risk_per_trade",
                                    policy.get("risk_per_trade", base_risk_per_trade),
                                )
                            ),
                        )
                        aggregated_signals.append(row)
                aggregated_signals.sort(
                    key=lambda row: (
                        int(row.get("source_policy_id") or 0),
                        -float(row.get("signal_strength", 0.0)),
                        str(row.get("symbol", "")),
                        str(row.get("side", "BUY")),
                        str(row.get("template", "")),
                    )
                )
                provided_signals = aggregated_signals
                generated_signals_count = len(provided_signals)
                generated_meta = SignalGenerationResult(
                    signals=provided_signals,
                    scan_truncated=scan_truncated_any,
                    scanned_symbols=scanned_symbols_total,
                    evaluated_candidates=evaluated_candidates_total,
                    total_symbols=total_symbols_max,
                )
                risk_budget_total = max(
                    0.0, float(policy.get("risk_per_trade", base_risk_per_trade))
                )
                risk_budget_by_policy = {
                    str(policy_id): float(
                        risk_budget_total * normalized_weights.get(policy_id, 0.0)
                    )
                    for policy_id in sorted(normalized_weights)
                }
                ensemble_meta = {
                    "id": int(active_ensemble.id or 0),
                    "name": active_ensemble.name,
                    "bundle_id": int(active_ensemble.bundle_id),
                    "regime_used": str(regime).strip().upper(),
                    "weights_source": weight_source,
                    "member_weights": {
                        str(policy_id): float(normalized_weights.get(policy_id, 0.0))
                        for policy_id in sorted(normalized_weights)
                    },
                    "base_member_weights": {
                        str(policy_id): float(base_positive_weights.get(policy_id, 0.0))
                        for policy_id in sorted(base_positive_weights)
                    },
                    "regime_member_weights": (
                        {
                            str(policy_id): float(regime_positive_weights.get(policy_id, 0.0))
                            for policy_id in sorted(regime_positive_weights)
                            if policy_id in base_positive_weights
                        }
                        if use_regime_weights
                        else {}
                    ),
                    "risk_budget_by_policy": risk_budget_by_policy,
                    "selected_counts_by_policy": {},
                    "weights_sum": float(total_weight),
                }
                policy["ensemble"] = ensemble_meta
                _log(
                    session,
                    "signals_generated",
                    {
                        "mode": "ensemble_autopilot",
                        "dataset_id": resolved_dataset_id,
                        "bundle_id": resolved_bundle_id,
                        "timeframes": resolved_timeframes,
                        "symbol_scope": symbol_scope,
                        "generated_count": generated_signals_count,
                        "ensemble_id": int(active_ensemble.id or 0),
                        "ensemble_name": active_ensemble.name,
                        "regime_used": ensemble_meta.get("regime_used"),
                        "weights_source": ensemble_meta.get("weights_source"),
                        "scan_truncated": generated_meta.scan_truncated,
                        "scanned_symbols": generated_meta.scanned_symbols,
                        "evaluated_candidates": generated_meta.evaluated_candidates,
                        "total_symbols": generated_meta.total_symbols,
                        "risk_budget_by_policy": risk_budget_by_policy,
                    },
                )
            else:
                generated_meta = generate_signals_for_policy(
                    session=session,
                    store=store,
                    dataset_id=resolved_dataset_id,
                    bundle_id=resolved_bundle_id,
                    asof=asof_dt,
                    timeframes=resolved_timeframes,
                    allowed_templates=policy.get("allowed_templates", []),
                    params_overrides=policy.get("params", {}),
                    max_symbols_scan=max_symbols_scan,
                    seed=seed,
                    mode="paper",
                    symbol_scope=symbol_scope,
                    ranking_weights=ranking_weights if isinstance(ranking_weights, dict) else None,
                    max_runtime_seconds=max_runtime_seconds,
                )
                provided_signals = list(generated_meta.signals)
                generated_signals_count = len(provided_signals)
                _log(
                    session,
                    "signals_generated",
                    {
                        "mode": "policy_autopilot",
                        "dataset_id": resolved_dataset_id,
                        "bundle_id": resolved_bundle_id,
                        "timeframes": resolved_timeframes,
                        "symbol_scope": symbol_scope,
                        "generated_count": generated_signals_count,
                        "allowed_templates": policy.get("allowed_templates", []),
                        "policy_id": policy.get("policy_id"),
                        "policy_name": policy.get("policy_name"),
                        "scan_truncated": generated_meta.scan_truncated,
                        "scanned_symbols": generated_meta.scanned_symbols,
                        "evaluated_candidates": generated_meta.evaluated_candidates,
                        "total_symbols": generated_meta.total_symbols,
                    },
                )
        signals_source = "generated"

    current_positions = list(positions_before)
    open_symbols = {p.symbol for p in current_positions}
    open_underlyings = {
        str((p.metadata_json or {}).get("underlying_symbol", p.symbol)).upper()
        for p in current_positions
    }
    max_positions = int(policy["max_positions"])
    sector_limit = max(1, min(2, max_positions))
    correlation_threshold = float(
        state_settings.get(
            "diversification_corr_threshold", settings.diversification_corr_threshold
        )
    )
    max_position_value_pct_adv = float(
        state_settings.get("max_position_value_pct_adv", settings.max_position_value_pct_adv)
    )
    allowed_sides = _normalize_allowed_sides(state_settings, settings)
    policy_allowed_instruments = policy.get("allowed_instruments", {})
    skipped_signals: list[dict[str, Any]] = []
    selected_signals: list[dict[str, Any]] = []
    executed_signals: list[dict[str, Any]] = []
    if pre_skipped_signals:
        skipped_signals.extend(pre_skipped_signals)

    risk_budget_by_policy: dict[int, float] = {}
    if isinstance(ensemble_meta, dict):
        raw_budget = ensemble_meta.get("risk_budget_by_policy", {})
        if isinstance(raw_budget, dict):
            for key, value in raw_budget.items():
                try:
                    risk_budget_by_policy[int(key)] = float(value)
                except (TypeError, ValueError):
                    continue
    member_budget_remaining = dict(risk_budget_by_policy)
    selected_counts_by_policy: dict[int, int] = {}

    if ensemble_meta is not None:
        candidates = sorted(
            [dict(item) for item in provided_signals],
            key=lambda item: (
                int(item.get("source_policy_id") or 0),
                -float(item.get("signal_strength", 0.0)),
                str(item.get("symbol", "")),
                str(item.get("side", "BUY")),
            ),
        )
    else:
        candidates = sorted(
            [dict(item) for item in provided_signals],
            key=lambda item: float(item.get("signal_strength", 0.0)),
            reverse=True,
        )
    if safe_mode_active and safe_mode_action == "exits_only":
        for signal in candidates:
            skipped_signals.append(
                {
                    "symbol": str(signal.get("symbol", "")).upper(),
                    "underlying_symbol": str(
                        signal.get("underlying_symbol", signal.get("symbol", ""))
                    ).upper(),
                    "template": str(signal.get("template", "trend_breakout")),
                    "side": str(signal.get("side", "BUY")).upper(),
                    "instrument_kind": str(signal.get("instrument_kind", "EQUITY_CASH")).upper(),
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                    "policy_name": policy.get("policy_name"),
                    "reason": "data_quality_fail_safe_mode",
                }
            )
        candidates = []
    elif confidence_gate_block_entries:
        for signal in candidates:
            skipped_signals.append(
                {
                    "symbol": str(signal.get("symbol", "")).upper(),
                    "underlying_symbol": str(
                        signal.get("underlying_symbol", signal.get("symbol", ""))
                    ).upper(),
                    "template": str(signal.get("template", "trend_breakout")),
                    "side": str(signal.get("side", "BUY")).upper(),
                    "instrument_kind": str(signal.get("instrument_kind", "EQUITY_CASH")).upper(),
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                    "policy_name": policy.get("policy_name"),
                    "reason": "confidence_gate_block_entries",
                    "details": {
                        "decision": confidence_gate_decision,
                        "reasons": confidence_gate_reasons,
                    },
                }
            )
        candidates = []
    elif no_trade_active:
        for signal in candidates:
            skipped_signals.append(
                {
                    "symbol": str(signal.get("symbol", "")).upper(),
                    "underlying_symbol": str(
                        signal.get("underlying_symbol", signal.get("symbol", ""))
                    ).upper(),
                    "template": str(signal.get("template", "trend_breakout")),
                    "side": str(signal.get("side", "BUY")).upper(),
                    "instrument_kind": str(signal.get("instrument_kind", "EQUITY_CASH")).upper(),
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                    "policy_name": policy.get("policy_name"),
                    "reason": "no_trade_gate_triggered",
                    "details": {
                        "reasons": no_trade_reasons,
                        "cooldown_remaining": no_trade_snapshot.get("cooldown_remaining", 0),
                    },
                }
            )
        candidates = []
    candidate_symbols = {
        str(item.get("symbol", "")).upper()
        for item in candidates
        if str(item.get("symbol", "")).strip()
    }
    candidate_symbols.update(open_underlyings)
    sectors = {
        row.symbol: (row.sector or "UNKNOWN")
        for row in session.exec(
            select(Symbol).where(Symbol.symbol.in_(list(candidate_symbols)))
        ).all()
    }
    sector_counts: dict[str, int] = {}
    for pos in current_positions:
        underlying = str((pos.metadata_json or {}).get("underlying_symbol", pos.symbol)).upper()
        sector = sectors.get(underlying, sectors.get(pos.symbol, "UNKNOWN"))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    selected_symbols = set(open_symbols)
    selected_underlyings = set(open_underlyings)
    inactive_symbols: set[str] = set()
    enforce_inactive_filter = operate_mode == "live" and not (
        safe_mode_active and safe_mode_action == "shadow_only"
    )
    if resolved_bundle_id is not None and enforce_inactive_filter:
        try:
            inactive_symbols = inactive_symbols_for_selection(
                session=session,
                settings=settings,
                store=store,
                bundle_id=int(resolved_bundle_id),
                timeframe=primary_timeframe,
                overrides=state_settings,
                reference_ts=asof_dt,
            )
        except Exception:  # noqa: BLE001
            inactive_symbols = set()

    for signal in candidates:
        symbol = str(signal.get("symbol", "")).upper()
        side = str(signal.get("side", "BUY")).upper()
        template = str(signal.get("template", "trend_breakout"))
        source_policy_id = int(signal.get("source_policy_id") or 0)
        source_policy_name = str(
            signal.get("source_policy_name")
            or (
                f"Policy {source_policy_id}"
                if source_policy_id > 0
                else policy.get("policy_name", "")
            )
        )
        member_required_risk = max(
            0.0,
            float(
                signal.get(
                    "member_required_risk", policy.get("risk_per_trade", base_risk_per_trade)
                )
            ),
        )
        requested_kind = str(signal.get("instrument_kind", "EQUITY_CASH")).upper()
        underlying_symbol = str(signal.get("underlying_symbol", symbol)).upper()
        allowed_set: set[str] | None = None
        if isinstance(policy_allowed_instruments, dict):
            allowed_for_side = policy_allowed_instruments.get(side)
            if isinstance(allowed_for_side, list) and allowed_for_side:
                allowed_set = {str(item).upper() for item in allowed_for_side}
        instrument_kind = requested_kind
        instrument_choice_reason = "provided"
        sector = sectors.get(underlying_symbol, sectors.get(symbol, "UNKNOWN"))
        base_meta = {
            "symbol": symbol,
            "underlying_symbol": underlying_symbol,
            "sector": sector,
            "template": template,
            "side": side,
            "instrument_kind": instrument_kind,
            "policy_mode": policy.get("mode"),
            "policy_id": policy.get("policy_id"),
            "policy_name": policy.get("policy_name"),
            "source_policy_id": source_policy_id if source_policy_id > 0 else None,
            "source_policy_name": source_policy_name if source_policy_id > 0 else None,
        }
        if ensemble_meta is not None and source_policy_id > 0:
            if source_policy_id not in member_budget_remaining:
                skipped_signals.append({**base_meta, "reason": "ensemble_weight_zero"})
                continue
            if member_budget_remaining[source_policy_id] + 1e-12 < member_required_risk:
                skipped_signals.append(
                    {
                        **base_meta,
                        "reason": "ensemble_member_budget_exhausted",
                        "member_required_risk": member_required_risk,
                        "member_budget_remaining": float(member_budget_remaining[source_policy_id]),
                    }
                )
                continue

        if symbol in inactive_symbols or underlying_symbol in inactive_symbols:
            skipped_signals.append({**base_meta, "reason": "inactive_symbol_data_gap"})
            continue

        if sector_counts.get(sector, 0) >= sector_limit:
            skipped_signals.append({**base_meta, "reason": "sector_concentration"})
            continue
        if len(current_positions) + len(selected_signals) >= max_positions:
            skipped_signals.append({**base_meta, "reason": "max_positions_reached"})
            continue
        if underlying_symbol in selected_underlyings:
            skipped_signals.append({**base_meta, "reason": "already_open"})
            continue
        if side not in {"BUY", "SELL"}:
            skipped_signals.append({**base_meta, "reason": "invalid_side"})
            continue
        if side not in allowed_sides:
            skipped_signals.append(
                {
                    **base_meta,
                    "reason": "shorts_disabled" if side == "SELL" else "side_not_allowed",
                }
            )
            continue
        if template not in policy["allowed_templates"]:
            skipped_signals.append(
                {
                    **base_meta,
                    "reason": "template_blocked_by_policy"
                    if policy.get("mode") == "policy"
                    else "template_blocked_by_regime",
                }
            )
            continue

        if side == "SELL":
            explicit_future = None
            if _is_futures_kind(instrument_kind):
                explicit_future = store.find_instrument(
                    session,
                    symbol=symbol,
                    instrument_kind=instrument_kind,
                )
            if explicit_future is not None:
                underlying_symbol = str(explicit_future.underlying or underlying_symbol).upper()
                signal["symbol"] = explicit_future.symbol
                signal["underlying_symbol"] = underlying_symbol
                signal["instrument_kind"] = explicit_future.kind
                signal["lot_size"] = max(1, int(explicit_future.lot_size))
                instrument_kind = explicit_future.kind
                instrument_choice_reason = "provided_futures_signal"
                if explicit_future.id is not None:
                    fut_frame = store.get_ohlcv_for_instrument(
                        session,
                        instrument_id=int(explicit_future.id),
                        timeframe=primary_timeframe,
                        asof=asof_dt,
                        window=20,
                    )
                    if not fut_frame.empty:
                        fut_bar = fut_frame.iloc[-1]
                        fut_open = float(fut_bar["open"])
                        fut_close = float(fut_bar["close"])
                        signal["price"] = fut_open if fut_open > 0 else fut_close
                        signal["adv"] = float(
                            np.nan_to_num(
                                (fut_frame["close"] * fut_frame["volume"]).tail(20).mean(),
                                nan=0.0,
                            )
                        )
            if explicit_future is not None:
                pass
            else:
                futures_candidate: Instrument | None = None
                if allowed_set is None or "STOCK_FUT" in allowed_set:
                    futures_candidate = store.find_futures_instrument_for_underlying(
                        session,
                        underlying=underlying_symbol,
                        bundle_id=resolved_bundle_id,
                        timeframe=primary_timeframe,
                    )
                if futures_candidate is not None:
                    instrument_kind = futures_candidate.kind
                    symbol = futures_candidate.symbol
                    signal["symbol"] = symbol
                    signal["underlying_symbol"] = underlying_symbol
                    signal["instrument_kind"] = instrument_kind
                    signal["lot_size"] = max(1, int(futures_candidate.lot_size))
                    instrument_choice_reason = "swing_short_requires_futures"
                    if futures_candidate.id is not None:
                        fut_frame = store.get_ohlcv_for_instrument(
                            session,
                            instrument_id=int(futures_candidate.id),
                            timeframe=primary_timeframe,
                            asof=asof_dt,
                            window=20,
                        )
                        if not fut_frame.empty:
                            fut_bar = fut_frame.iloc[-1]
                            fut_open = float(fut_bar["open"])
                            fut_close = float(fut_bar["close"])
                            signal["price"] = fut_open if fut_open > 0 else fut_close
                            signal["adv"] = float(
                                np.nan_to_num(
                                    (fut_frame["close"] * fut_frame["volume"]).tail(20).mean(),
                                    nan=0.0,
                                )
                            )
                elif allowed_set is None or "EQUITY_CASH" in allowed_set:
                    instrument_kind = "EQUITY_CASH"
                    signal["instrument_kind"] = instrument_kind
                    signal["symbol"] = underlying_symbol
                    signal["underlying_symbol"] = underlying_symbol
                    instrument_choice_reason = "cash_intraday_short_fallback"
                else:
                    skipped_signals.append(
                        {
                            **base_meta,
                            "reason": "no_short_instrument_available",
                        }
                    )
                    continue
        else:
            if allowed_set is not None and instrument_kind not in allowed_set:
                if "EQUITY_CASH" in allowed_set:
                    instrument_kind = "EQUITY_CASH"
                    signal["instrument_kind"] = instrument_kind
                else:
                    skipped_signals.append(
                        {
                            **base_meta,
                            "reason": "instrument_blocked_by_policy",
                        }
                    )
                    continue

        if allowed_set is not None and instrument_kind not in allowed_set:
            skipped_signals.append(
                {
                    **base_meta,
                    "instrument_kind": instrument_kind,
                    "reason": "instrument_blocked_by_policy",
                }
            )
            continue

        signal["instrument_kind"] = instrument_kind
        signal["instrument_choice_reason"] = instrument_choice_reason
        signal["sector"] = sector

        if _correlation_reject(
            signal, selected_symbols=selected_symbols, threshold=correlation_threshold
        ):
            skipped_signals.append({**base_meta, "reason": "correlation_threshold"})
            continue

        selected_signals.append(signal)
        if (
            ensemble_meta is not None
            and source_policy_id > 0
            and source_policy_id in member_budget_remaining
        ):
            member_budget_remaining[source_policy_id] = max(
                0.0,
                float(member_budget_remaining[source_policy_id]) - member_required_risk,
            )
            selected_counts_by_policy[source_policy_id] = (
                int(selected_counts_by_policy.get(source_policy_id, 0)) + 1
            )
        selected_symbols.add(symbol)
        selected_underlyings.add(underlying_symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if ensemble_meta is not None:
        ensemble_meta["selected_counts_by_policy"] = {
            str(key): int(value) for key, value in sorted(selected_counts_by_policy.items())
        }
        ensemble_meta["remaining_risk_budget_by_policy"] = {
            str(key): float(value) for key, value in sorted(member_budget_remaining.items())
        }
        policy["ensemble"] = ensemble_meta

    use_simulator_engine = bool(
        state_settings.get("paper_use_simulator_engine", settings.paper_use_simulator_engine)
    )
    if (safe_mode_active and safe_mode_action == "shadow_only") or confidence_gate_force_shadow:
        return _run_paper_step_shadow_only(
            session=session,
            settings=settings,
            live_state=state,
            state_settings=state_settings,
            payload=payload,
            policy=policy,
            regime=regime,
            asof_dt=asof_dt,
            base_risk_per_trade=base_risk_per_trade,
            base_max_positions=base_max_positions,
            mark_prices=mark_prices,
            selected_signals=selected_signals,
            skipped_signals=skipped_signals,
            generated_meta=generated_meta,
            generated_signals_count=generated_signals_count,
            signals_source=signals_source,
            safe_mode_active=safe_mode_active,
            safe_mode_action=safe_mode_action,
            safe_mode_reason=safe_mode_reason,
            no_trade_snapshot=no_trade_snapshot,
            confidence_gate_snapshot=confidence_gate_snapshot,
            quality_status=quality_status,
            quality_warn_summary=quality_warn_summary,
            risk_overlay=risk_overlay,
            cost_spike_active=cost_spike_active,
            cost_spike_meta=cost_spike_meta,
            scan_guard_active=scan_guard_active,
            scan_guard_meta=scan_guard_meta,
            resolved_bundle_id=resolved_bundle_id,
            resolved_dataset_id=resolved_dataset_id,
            resolved_timeframes=resolved_timeframes,
            live_positions=positions_before,
            live_orders=orders_before,
        )
    if use_simulator_engine:
        return _run_paper_step_with_simulator_engine(
            session=session,
            settings=settings,
            state=state,
            state_settings=state_settings,
            payload=payload,
            policy=policy,
            regime=regime,
            asof_dt=asof_dt,
            base_risk_per_trade=base_risk_per_trade,
            base_max_positions=base_max_positions,
            mark_prices=mark_prices,
            selected_signals=selected_signals,
            skipped_signals=skipped_signals,
            generated_meta=generated_meta,
            generated_signals_count=generated_signals_count,
            signals_source=signals_source,
            safe_mode_active=safe_mode_active,
            safe_mode_action=safe_mode_action,
            safe_mode_reason=safe_mode_reason,
            no_trade_snapshot=no_trade_snapshot,
            confidence_gate_snapshot=confidence_gate_snapshot,
            quality_status=quality_status,
            quality_warn_summary=quality_warn_summary,
            risk_overlay=risk_overlay,
            cost_spike_active=cost_spike_active,
            cost_spike_meta=cost_spike_meta,
            scan_guard_active=scan_guard_active,
            scan_guard_meta=scan_guard_meta,
            resolved_bundle_id=resolved_bundle_id,
            resolved_dataset_id=resolved_dataset_id,
            resolved_timeframes=resolved_timeframes,
            positions_before=positions_before,
            positions_before_by_id=positions_before_by_id,
            position_ids_before=position_ids_before,
            order_ids_before=order_ids_before,
            equity_before=equity_before,
            cash_before=cash_before,
            drawdown_before=drawdown_before,
            mtm_before=mtm_before,
        )

    entry_cost_total = 0.0
    exit_cost_total = 0.0
    entry_slippage_cost_total = 0.0
    exit_slippage_cost_total = 0.0
    traded_notional_total = 0.0
    executed_count = 0

    for signal in selected_signals:
        symbol = str(signal.get("symbol", "")).upper()
        underlying_symbol = str(signal.get("underlying_symbol", symbol)).upper()
        side = str(signal.get("side", "BUY")).upper()
        template = str(signal.get("template", "trend_breakout"))
        instrument_kind = str(signal.get("instrument_kind", "EQUITY_CASH")).upper()
        lot_size = max(1, int(signal.get("lot_size", 1)))
        is_futures = _is_futures_kind(instrument_kind)
        price = float(signal.get("price", 0.0))
        stop_distance = float(signal.get("stop_distance", 0.0))
        if is_futures:
            qty_lots = _position_size_lots(
                state.equity,
                float(policy["risk_per_trade"]),
                stop_distance,
                lot_size,
            )
            qty = qty_lots * lot_size
        else:
            qty = _position_size(state.equity, float(policy["risk_per_trade"]), stop_distance)
            qty = _adjust_qty_for_lot(qty, lot_size)
            qty_lots = max(1, int(np.floor(qty / max(1, lot_size)))) if qty > 0 else 0
        if price <= 0 or qty <= 0:
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "invalid_price_or_size",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        slippage = (settings.slippage_base_bps / 10_000) * (1 + float(signal.get("vol_scale", 0.0)))
        fill_price = price * (1 + slippage) if side == "BUY" else price * (1 - slippage)
        adv_notional = float(signal.get("adv", 0.0))
        if adv_notional > 0 and max_position_value_pct_adv > 0:
            max_notional = adv_notional * max_position_value_pct_adv
            if is_futures:
                qty_adv_lots = int(np.floor(max_notional / max(fill_price * lot_size, 1e-9)))
                if qty_adv_lots <= 0:
                    skipped_signals.append(
                        {
                            "symbol": symbol,
                            "template": template,
                            "side": side,
                            "instrument_kind": instrument_kind,
                            "reason": "adv_cap_zero_qty",
                            "policy_mode": policy.get("mode"),
                            "policy_id": policy.get("policy_id"),
                        }
                    )
                    continue
                qty_lots = min(qty_lots, qty_adv_lots)
                qty = qty_lots * lot_size
            else:
                qty_adv = int(np.floor(max_notional / max(fill_price, 1e-9)))
                qty_adv = _adjust_qty_for_lot(qty_adv, lot_size)
                if qty_adv <= 0:
                    skipped_signals.append(
                        {
                            "symbol": symbol,
                            "template": template,
                            "side": side,
                            "instrument_kind": instrument_kind,
                            "reason": "adv_cap_zero_qty",
                            "policy_mode": policy.get("mode"),
                            "policy_id": policy.get("policy_id"),
                        }
                    )
                    continue
                qty = min(qty, qty_adv)
                qty_lots = max(1, int(np.floor(qty / max(1, lot_size)))) if qty > 0 else 0

        if qty <= 0 or qty_lots <= 0:
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "qty_zero_after_caps",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        must_exit_by_eod = side == "SELL" and instrument_kind == "EQUITY_CASH"
        notional = qty * fill_price
        traded_notional_total += float(notional)
        entry_slippage_cost = (
            (fill_price - price) * qty if side == "BUY" else (price - fill_price) * qty
        )
        entry_slippage_cost_total += max(0.0, float(entry_slippage_cost))
        margin_required = (
            notional * _futures_margin_pct(state_settings, settings) if is_futures else 0.0
        )
        cost_override = dict(policy.get("cost_model", {}))
        if must_exit_by_eod:
            cost_override["mode"] = "intraday"
        entry_cost = _transaction_cost(
            notional=notional,
            side=side,
            instrument_kind=instrument_kind,
            settings=settings,
            state_settings=state_settings,
            override_cost_model=cost_override,
        )
        if is_futures and (margin_required + entry_cost > state.cash):
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "insufficient_margin",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue
        if (not is_futures) and side == "BUY" and (notional + entry_cost > state.cash):
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "insufficient_cash",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        entry_reason = "SIGNAL_SHORT" if side == "SELL" else "SIGNAL"
        order = PaperOrder(
            symbol=symbol,
            side=side,
            instrument_kind=instrument_kind,
            lot_size=lot_size,
            qty_lots=qty_lots,
            qty=qty,
            fill_price=fill_price,
            status="FILLED",
            reason=entry_reason,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        if side == "BUY":
            stop_price = max(0.0, fill_price - stop_distance)
            if (
                isinstance(signal.get("target_price"), (int, float))
                and float(signal.get("target_price")) > 0
            ):
                target_price = float(signal.get("target_price"))
            else:
                target_price = None
            if is_futures:
                state.cash -= margin_required + entry_cost
            else:
                state.cash -= notional + entry_cost
        else:
            stop_price = fill_price + stop_distance
            if (
                isinstance(signal.get("target_price"), (int, float))
                and float(signal.get("target_price")) > 0
            ):
                target_price = float(signal.get("target_price"))
            else:
                target_price = None
            if is_futures:
                state.cash -= margin_required + entry_cost
            else:
                state.cash += notional - entry_cost

        position = PaperPosition(
            symbol=symbol,
            side=side,
            instrument_kind=instrument_kind,
            lot_size=lot_size,
            qty_lots=qty_lots,
            margin_reserved=margin_required,
            must_exit_by_eod=must_exit_by_eod,
            qty=qty,
            avg_price=fill_price,
            stop_price=stop_price,
            target_price=target_price,
            metadata_json={
                "template": template,
                "underlying_symbol": underlying_symbol,
                "instrument_choice_reason": signal.get("instrument_choice_reason", "provided"),
            },
            opened_at=_utc_now(),
        )
        entry_cost_total += entry_cost
        executed_count += 1

        session.add(order)
        session.add(position)
        _log(
            session,
            "signal_selected",
            {
                "symbol": symbol,
                "underlying_symbol": underlying_symbol,
                "template": template,
                "side": side,
                "instrument_kind": instrument_kind,
                "instrument_choice_reason": signal.get("instrument_choice_reason", "provided"),
                "qty": qty,
                "qty_lots": qty_lots,
                "margin_reserved": margin_required,
                "fill_price": fill_price,
                "signal_strength": float(signal.get("signal_strength", 0.0)),
                "selection_reason": policy.get("selection_reason"),
                "policy_mode": policy.get("mode"),
                "policy_id": policy.get("policy_id"),
                "must_exit_by_eod": must_exit_by_eod,
            },
        )
        executed_signals.append(
            {
                **dict(signal),
                "symbol": symbol,
                "underlying_symbol": underlying_symbol,
                "instrument_kind": instrument_kind,
                "qty": qty,
                "qty_lots": qty_lots,
                "fill_price": fill_price,
                "entry_cost": entry_cost,
            }
        )
        if must_exit_by_eod:
            _log(
                session,
                "short_intraday_opened",
                {
                    "symbol": symbol,
                    "qty": qty,
                    "fill_price": fill_price,
                    "squareoff_cutoff": state_settings.get(
                        "paper_short_squareoff_time", settings.paper_short_squareoff_time
                    ),
                },
            )
        current_positions.append(position)
        open_symbols.add(symbol)
        open_underlyings.add(underlying_symbol)

    for position in list(current_positions):
        if position.symbol not in mark_prices:
            continue
        mark = float(mark_prices[position.symbol])
        should_close, reason = _position_exit_condition(position, mark)
        if not should_close:
            continue
        exit_cost_total += _close_position(
            session=session,
            settings=settings,
            state_settings=state_settings,
            policy=policy,
            state=state,
            position=position,
            price=mark,
            reason=reason,
        )
        if position in current_positions:
            current_positions.remove(position)

    cutoff = _parse_cutoff(
        state_settings.get("paper_short_squareoff_time"),
        settings.paper_short_squareoff_time,
    )
    if _is_squareoff_due(asof_dt, cutoff):
        for position in list(current_positions):
            if not (position.side.upper() == "SELL" and position.must_exit_by_eod):
                continue
            mark = float(mark_prices.get(position.symbol, position.avg_price))
            exit_cost_total += _close_position(
                session=session,
                settings=settings,
                state_settings=state_settings,
                policy=policy,
                state=state,
                position=position,
                price=mark,
                reason="EOD_SQUARE_OFF",
            )
            _log(
                session,
                "forced_squareoff",
                {
                    "symbol": position.symbol,
                    "qty": position.qty,
                    "fill_price": mark,
                    "cutoff": cutoff.isoformat(timespec="minutes"),
                },
            )
            if position in current_positions:
                current_positions.remove(position)

    live_positions = get_positions(session)
    mtm = 0.0
    for position in live_positions:
        mark = float(mark_prices.get(position.symbol, position.avg_price))
        mtm += _mark_to_market_component(position, mark)

    state.equity = state.cash + mtm
    state.peak_equity = max(state.peak_equity, state.equity)
    state.drawdown = (state.equity / state.peak_equity - 1.0) if state.peak_equity > 0 else 0.0

    dd_limit = float(state.settings_json.get("kill_switch_dd", settings.kill_switch_drawdown))
    if state.drawdown <= -dd_limit:
        state.kill_switch_active = True
        state.cooldown_days_left = settings.kill_switch_cooldown_days
        _log(session, "kill_switch_activated", {"drawdown": state.drawdown, "limit": dd_limit})

    session.add(state)
    for skip in skipped_signals:
        _log(session, "signal_skipped", skip)
        _log(session, "paper_skip", skip)
    session.commit()
    session.refresh(state)

    live_positions = get_positions(session)
    orders_after = get_orders(session)
    position_ids_after = {int(row.id) for row in live_positions if row.id is not None}
    order_ids_after = {int(row.id) for row in orders_after if row.id is not None}
    new_position_ids = sorted(position_ids_after - position_ids_before)
    closed_position_ids = sorted(position_ids_before - position_ids_after)
    new_order_ids = sorted(order_ids_after - order_ids_before)
    closed_position_symbols = [
        positions_before_by_id[item].symbol
        for item in closed_position_ids
        if item in positions_before_by_id
    ]

    selected_reason_histogram = _selection_reason_histogram(executed_signals)
    skipped_reason_histogram = _reason_histogram(skipped_signals)
    total_cost = float(entry_cost_total + exit_cost_total)
    realized_pnl = float(state.cash - cash_before)
    unrealized_pnl = float(mtm - mtm_before)
    net_pnl = float(state.equity - equity_before)
    gross_pnl = float(net_pnl + total_cost)
    turnover = float(traded_notional_total / max(1e-9, equity_before))
    exposure = float(_positions_notional(live_positions) / max(1e-9, state.equity))
    avg_holding_days = _average_holding_days(live_positions, asof_dt)

    cost_summary = {
        "entry_cost_total": float(entry_cost_total),
        "exit_cost_total": float(exit_cost_total),
        "total_cost": total_cost,
        "entry_slippage_cost": float(entry_slippage_cost_total),
        "exit_slippage_cost": float(exit_slippage_cost_total),
        "cost_model_enabled": bool(
            policy.get("cost_model", {}).get(
                "enabled",
                state_settings.get("cost_model_enabled", settings.cost_model_enabled),
            )
        ),
        "cost_mode": str(
            policy.get("cost_model", {}).get(
                "mode",
                state_settings.get("cost_mode", settings.cost_mode),
            )
        ),
    }
    ensemble_payload = (
        policy.get("ensemble", {}) if isinstance(policy.get("ensemble"), dict) else {}
    )

    run_summary = {
        "execution_mode": "LIVE",
        "shadow_only": False,
        "live_state_mutated": True,
        "policy_mode": policy.get("mode"),
        "policy_selection_reason": policy.get("selection_reason"),
        "policy_status": policy.get("policy_status"),
        "health_status": policy.get("health_status"),
        "ensemble_active": bool(ensemble_payload),
        "ensemble_id": ensemble_payload.get("id"),
        "ensemble_name": ensemble_payload.get("name"),
        "ensemble_regime_used": ensemble_payload.get("regime_used"),
        "ensemble_weights_source": ensemble_payload.get("weights_source"),
        "ensemble_risk_budget_by_policy": dict(
            ensemble_payload.get("risk_budget_by_policy", {})
            if isinstance(ensemble_payload.get("risk_budget_by_policy", {}), dict)
            else {}
        ),
        "ensemble_selected_counts_by_policy": dict(
            ensemble_payload.get("selected_counts_by_policy", {})
            if isinstance(ensemble_payload.get("selected_counts_by_policy", {}), dict)
            else {}
        ),
        "safe_mode_active": bool(safe_mode_active),
        "safe_mode_action": safe_mode_action,
        "safe_mode_reason": safe_mode_reason,
        "no_trade": dict(no_trade_snapshot or {}),
        "no_trade_triggered": bool((no_trade_snapshot or {}).get("triggered", False)),
        "no_trade_reasons": list((no_trade_snapshot or {}).get("reasons", [])),
        "confidence_gate": dict(confidence_gate_snapshot or {}),
        "data_quality_status": quality_status,
        "data_quality_warn_summary": quality_warn_summary,
        "cost_ratio_spike_active": bool(cost_spike_active),
        "cost_ratio_spike_meta": cost_spike_meta,
        "scan_guard_active": bool(scan_guard_active),
        "scan_guard_meta": scan_guard_meta,
        "signals_source": signals_source,
        "bundle_id": resolved_bundle_id,
        "dataset_id": resolved_dataset_id,
        "timeframes": resolved_timeframes,
        "scan_truncated": bool(generated_meta.scan_truncated),
        "scanned_symbols": int(generated_meta.scanned_symbols),
        "evaluated_candidates": int(generated_meta.evaluated_candidates),
        "total_symbols": int(generated_meta.total_symbols),
        "generated_signals_count": int(generated_signals_count),
        "selected_signals_count": int(executed_count),
        "skipped_signals_count": int(len(skipped_signals)),
        "positions_before": len(positions_before),
        "positions_after": len(live_positions),
        "positions_opened": len(new_position_ids),
        "positions_closed": len(closed_position_ids),
        "new_position_ids": new_position_ids,
        "closed_position_ids": closed_position_ids,
        "closed_position_symbols": closed_position_symbols,
        "new_order_ids": new_order_ids,
        "selected_signals": [
            {
                "symbol": str(item.get("symbol", "")),
                "side": str(item.get("side", "")),
                "instrument_kind": str(item.get("instrument_kind", "")),
                "selection_reason": str(item.get("instrument_choice_reason", "provided")),
            }
            for item in executed_signals
        ],
        "selected_reason_histogram": selected_reason_histogram,
        "skipped_reason_histogram": skipped_reason_histogram,
        "risk_scale": float(risk_overlay.get("risk_scale", 1.0)),
        "confidence_risk_scale": float(
            (confidence_gate_snapshot.get("summary", {}) if isinstance(confidence_gate_snapshot.get("summary", {}), dict) else {}).get(
                "confidence_risk_scale",
                risk_overlay.get("confidence_risk_scale", 1.0),
            )
        ),
        "effective_risk_scale": float(
            float(risk_overlay.get("risk_scale", 1.0))
            * float(
                (confidence_gate_snapshot.get("summary", {}) if isinstance(confidence_gate_snapshot.get("summary", {}), dict) else {}).get(
                    "confidence_risk_scale",
                    risk_overlay.get("confidence_risk_scale", 1.0),
                )
            )
        ),
        "equity_before": equity_before,
        "equity_after": float(state.equity),
        "cash_before": cash_before,
        "cash_after": float(state.cash),
        "drawdown_before": drawdown_before,
        "drawdown": float(state.drawdown),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "trade_count": int(len(new_position_ids) + len(closed_position_ids)),
        "turnover": turnover,
        "exposure": exposure,
        "avg_holding_days": avg_holding_days,
        "kill_switch_active": bool(state.kill_switch_active),
        "paper_engine": "legacy",
    }
    if "data_digest" not in run_summary:
        run_summary["data_digest"] = _stable_hash(
            {
                "bundle_id": resolved_bundle_id,
                "dataset_id": resolved_dataset_id,
                "timeframes": resolved_timeframes,
                "asof_date": asof_dt.date().isoformat(),
                "seed": seed,
            }
        )
    run_summary["seed"] = seed
    run_summary["result_digest"] = _result_digest(run_summary, cost_summary)
    policy_id_value: int | None = None
    try:
        if policy.get("policy_id") is not None:
            policy_id_value = int(policy["policy_id"])
    except (TypeError, ValueError):
        policy_id_value = None
    run_row = PaperRun(
        bundle_id=resolved_bundle_id,
        policy_id=policy_id_value,
        asof_ts=asof_dt,
        mode="LIVE",
        regime=regime,
        signals_source=signals_source,
        generated_signals_count=generated_signals_count,
        selected_signals_count=executed_count,
        skipped_signals_count=len(skipped_signals),
        scanned_symbols=int(generated_meta.scanned_symbols),
        evaluated_candidates=int(generated_meta.evaluated_candidates),
        scan_truncated=bool(generated_meta.scan_truncated),
        summary_json=run_summary,
        cost_summary_json=cost_summary,
    )
    session.add(run_row)
    session.commit()
    session.refresh(run_row)
    _emit_digest_mismatch_if_any(session=session, run_row=run_row, run_summary=run_summary)
    session.commit()

    health_short_payload: dict[str, Any] | None = None
    health_long_payload: dict[str, Any] | None = None
    health_action_payload: dict[str, Any] | None = None

    active_policy: Policy | None = None
    if policy_id_value is not None:
        active_policy = session.get(Policy, policy_id_value)

    if active_policy is not None and policy.get("mode") == "policy":
        short_window = max(
            5,
            int(
                state_settings.get(
                    "health_window_days_short",
                    settings.health_window_days_short,
                )
            ),
        )
        long_window = max(
            short_window,
            int(
                state_settings.get(
                    "health_window_days_long",
                    settings.health_window_days_long,
                )
            ),
        )
        health_short = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=active_policy,
            window_days=short_window,
            asof_date=asof_dt.date(),
            refresh=True,
            overrides=state_settings,
        )
        health_long = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=active_policy,
            window_days=long_window,
            asof_date=asof_dt.date(),
            refresh=True,
            overrides=state_settings,
        )
        action_source: PolicyHealthSnapshot = (
            health_long if health_long.status == DEGRADED else health_short
        )
        health_action_payload = apply_policy_health_actions(
            session,
            settings=settings,
            policy=active_policy,
            snapshot=action_source,
            overrides=state_settings,
        )
        _log(
            session,
            "policy_health_snapshot",
            {
                "policy_id": active_policy.id,
                "short_window_days": short_window,
                "short_status": health_short.status,
                "short_reasons": health_short.reasons_json,
                "long_window_days": long_window,
                "long_status": health_long.status,
                "long_reasons": health_long.reasons_json,
            },
        )
        _log(
            session,
            "policy_health_action",
            {
                "policy_id": active_policy.id,
                "status": action_source.status,
                "action": health_action_payload.get("action"),
                "risk_scale_override": health_action_payload.get("risk_scale_override"),
                "policy_status": health_action_payload.get("policy_status"),
                "reasons": action_source.reasons_json,
            },
        )

        action_name = str(health_action_payload.get("action", "NONE")).upper()
        action_policy_status = str(
            health_action_payload.get("policy_status", policy_status(active_policy))
        ).upper()
        if action_policy_status in {PAUSED, RETIRED}:
            fallback = select_fallback_policy(
                session,
                current_policy_id=int(active_policy.id),
                regime=regime,
            )
            if fallback is not None:
                merged_state = dict(state.settings_json or {})
                if str(merged_state.get("paper_mode", "strategy")) == "policy" and int(
                    merged_state.get("active_policy_id") or 0
                ) == int(active_policy.id):
                    merged_state["active_policy_id"] = int(fallback.id)
                    merged_state["active_policy_name"] = fallback.name
                    state.settings_json = merged_state
                    session.add(state)
                    _log(
                        session,
                        "policy_fallback_selected",
                        {
                            "from_policy_id": active_policy.id,
                            "to_policy_id": fallback.id,
                            "from_status": action_policy_status,
                            "reason": "policy_degraded_or_paused",
                            "regime": regime,
                        },
                    )
                    session.commit()
                    session.refresh(state)
            elif action_name != "NONE":
                _log(
                    session,
                    "policy_fallback_unavailable",
                    {
                        "policy_id": active_policy.id,
                        "status": action_policy_status,
                        "regime": regime,
                        "reason": "no_fallback_policy",
                    },
                )
        session.commit()
        health_short_payload = health_short.model_dump()
        health_long_payload = health_long.model_dump()

    generated_report_id: int | None = None
    auto_report_enabled = bool(
        state_settings.get("reports_auto_generate_daily", settings.reports_auto_generate_daily)
    )
    if auto_report_enabled:
        report_row = generate_daily_report(
            session=session,
            settings=settings,
            report_date=asof_dt.date(),
            bundle_id=resolved_bundle_id,
            policy_id=run_row.policy_id,
            overwrite=True,
        )
        generated_report_id = int(report_row.id) if report_row.id is not None else None

    effective_context = _paper_effective_context(
        session=session,
        settings=settings,
        state_settings=state_settings,
        bundle_id=resolved_bundle_id,
        timeframe=(resolved_timeframes[0] if resolved_timeframes else "1d"),
        asof_dt=asof_dt,
        provider_stage_status=(
            str(payload.get("provider_stage_status"))
            if payload.get("provider_stage_status") is not None
            else None
        ),
        confidence_gate_snapshot=confidence_gate_snapshot,
        data_digest=(
            str(run_summary.get("data_digest"))
            if run_summary.get("data_digest") is not None
            else None
        ),
        engine_version=(
            str(run_summary.get("engine_version"))
            if run_summary.get("engine_version") is not None
            else None
        ),
        seed=(
            int(run_summary.get("seed"))
            if run_summary.get("seed") is not None
            else None
        ),
        notes=["paper_run_step"],
        store=store,
    )

    return jsonable_encoder(
        {
            "status": "ok",
            "regime": regime,
            "policy": policy,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "policy_status": policy.get("policy_status"),
            "policy_health_status": policy.get("health_status"),
            "policy_health_reasons": policy.get("health_reasons", []),
            "ensemble": ensemble_payload,
            "risk_scaled": bool(
                float(policy["risk_per_trade"]) < base_risk_per_trade
                or int(policy["max_positions"]) < base_max_positions
                or (
                    float(risk_overlay.get("risk_scale", 1.0))
                    * float(
                        (confidence_gate_snapshot.get("summary", {}) if isinstance(confidence_gate_snapshot.get("summary", {}), dict) else {}).get(
                            "confidence_risk_scale",
                            risk_overlay.get("confidence_risk_scale", 1.0),
                        )
                    )
                )
                < 0.999
            ),
            "paper_run_id": run_row.id,
            "signals_source": signals_source,
            "generated_signals_count": generated_signals_count,
            "selected_signals_count": executed_count,
            "selected_signals": executed_signals,
            "skipped_signals": skipped_signals,
            "safe_mode": {
                "active": bool(safe_mode_active),
                "action": safe_mode_action,
                "reason": safe_mode_reason,
                "status": quality_status,
                "warnings": quality_warn_summary,
            },
            "no_trade": dict(no_trade_snapshot or {}),
            "confidence_gate": dict(confidence_gate_snapshot or {}),
            "guardrails": {
                "cost_ratio_spike_active": bool(cost_spike_active),
                "cost_ratio_spike_meta": cost_spike_meta,
                "scan_guard_active": bool(scan_guard_active),
                "scan_guard_meta": scan_guard_meta,
            },
            "scan_truncated": generated_meta.scan_truncated,
            "scanned_symbols": generated_meta.scanned_symbols,
            "evaluated_candidates": generated_meta.evaluated_candidates,
            "total_symbols": generated_meta.total_symbols,
            "bundle_id": resolved_bundle_id,
            "dataset_id": resolved_dataset_id,
            "timeframes": resolved_timeframes,
            "cost_summary": cost_summary,
            "risk_overlay": {
                "risk_scale": float(risk_overlay.get("risk_scale", 1.0)),
                "effective_risk_scale": float(
                    float(risk_overlay.get("risk_scale", 1.0))
                    * float(
                        (confidence_gate_snapshot.get("summary", {}) if isinstance(confidence_gate_snapshot.get("summary", {}), dict) else {}).get(
                            "confidence_risk_scale",
                            risk_overlay.get("confidence_risk_scale", 1.0),
                        )
                    )
                ),
                "confidence_risk_scaling_enabled": bool(
                    risk_overlay.get("confidence_risk_scaling_enabled", False)
                ),
                "confidence_risk_scale": float(
                    (confidence_gate_snapshot.get("summary", {}) if isinstance(confidence_gate_snapshot.get("summary", {}), dict) else {}).get(
                        "confidence_risk_scale",
                        risk_overlay.get("confidence_risk_scale", 1.0),
                    )
                ),
                "confidence_risk_scale_low_threshold": float(
                    risk_overlay.get("confidence_risk_scale_low_threshold", 0.35)
                ),
                "realized_vol": float(risk_overlay.get("realized_vol", 0.0)),
                "target_vol": float(risk_overlay.get("target_vol", 0.0)),
                "caps_applied": dict(risk_overlay.get("caps", {})),
            },
            "paper_engine": "legacy",
            "result_digest": run_summary.get("result_digest"),
            "report_id": generated_report_id,
            "health_short": health_short_payload,
            "health_long": health_long_payload,
            "health_action": health_action_payload,
            "state": _dump_model(state),
            "positions": [_dump_model(p) for p in live_positions],
            "orders": [_dump_model(o) for o in orders_after],
            "effective_context": effective_context,
        }
    )


def preview_policy_signals(
    session: Session,
    settings: Settings,
    payload: dict[str, Any],
    store: DataStore | None = None,
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    regime = str(payload.get("regime", "TREND_UP"))
    preview_asof = _asof_datetime(payload)
    policy_override: int | None = None
    try:
        if payload.get("policy_id") is not None:
            policy_override = int(payload.get("policy_id"))
    except (TypeError, ValueError):
        policy_override = None
    policy = _resolve_execution_policy(
        session,
        state,
        settings,
        regime,
        policy_override_id=policy_override,
        asof_date=preview_asof.date(),
    )
    store = store or DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )

    timeframes = _resolve_timeframes(payload, policy)
    state_settings = state.settings_json or {}
    bundle_id = _resolve_bundle_id(session, payload, policy, settings)
    dataset_id = _resolve_dataset_id(session, payload, policy, timeframes, settings)
    if bundle_id is None and dataset_id is None:
        return {
            "regime": regime,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "policy_status": policy.get("policy_status"),
            "health_status": policy.get("health_status"),
            "health_reasons": policy.get("health_reasons", []),
            "signals_source": "generated",
            "signals": [],
            "generated_signals_count": 0,
            "selected_signals_count": 0,
            "scan_truncated": False,
            "scanned_symbols": 0,
            "evaluated_candidates": 0,
            "total_symbols": 0,
            "skipped_signals": [
                {
                    "reason": "dataset_or_bundle_not_found",
                    "message": "No dataset/bundle available for preview.",
                }
            ],
        }

    symbol_scope = _resolve_symbol_scope(payload, policy)
    max_symbols_scan = _resolve_max_symbols_scan(payload, policy, state_settings, settings)
    max_runtime_seconds = _resolve_max_runtime_seconds(payload, state_settings, settings)
    seed = _resolve_seed(payload, policy, settings)
    preferred_ensemble_id: int | None = None
    active_ensemble = None
    paper_mode_setting = str(state_settings.get("paper_mode", "strategy")).strip().lower()
    state_active_policy_id: int | None = None
    try:
        if state_settings.get("active_policy_id") is not None:
            state_active_policy_id = int(state_settings.get("active_policy_id"))
    except (TypeError, ValueError):
        state_active_policy_id = None
    use_ensemble_mode = (
        paper_mode_setting == "policy"
        and policy_override is None
        and state_active_policy_id is None
    )
    if use_ensemble_mode:
        try:
            if state_settings.get("active_ensemble_id") is not None:
                preferred_ensemble_id = int(state_settings.get("active_ensemble_id"))
        except (TypeError, ValueError):
            preferred_ensemble_id = None
        active_ensemble = get_active_policy_ensemble(
            session,
            bundle_id=int(bundle_id) if isinstance(bundle_id, int) else None,
            preferred_ensemble_id=preferred_ensemble_id,
        )
    ensemble_members = (
        list_policy_ensemble_members(
            session,
            ensemble_id=int(active_ensemble.id or 0),
            enabled_only=False,
        )
        if active_ensemble is not None
        else []
    )
    ensemble_regime_weights = (
        list_policy_ensemble_regime_weights(
            session,
            ensemble_id=int(active_ensemble.id or 0),
        )
        if active_ensemble is not None
        else {}
    )

    generated = SignalGenerationResult(
        signals=[],
        scan_truncated=False,
        scanned_symbols=0,
        evaluated_candidates=0,
        total_symbols=0,
    )
    if use_ensemble_mode and active_ensemble is not None and ensemble_members:
        regime_weight_payload = (
            ensemble_regime_weights.get(str(regime).strip().upper(), {})
            if isinstance(ensemble_regime_weights, dict)
            else {}
        )
        regime_positive_weights: dict[int, float] = {}
        if isinstance(regime_weight_payload, dict):
            for key, value in regime_weight_payload.items():
                try:
                    policy_id = int(key)
                except (TypeError, ValueError):
                    continue
                if policy_id <= 0:
                    continue
                regime_positive_weights[policy_id] = max(0.0, float(value))
        merged_signals: list[dict[str, Any]] = []
        scan_truncated_any = False
        scanned_symbols_total = 0
        evaluated_candidates_total = 0
        total_symbols_max = 0
        for member in sorted(
            [row for row in ensemble_members if bool(row.get("enabled", True))],
            key=lambda row: int(row.get("policy_id") or 0),
        ):
            source_policy_id = int(member.get("policy_id") or 0)
            if source_policy_id <= 0:
                continue
            base_member_weight = max(0.0, float(member.get("weight") or 0.0))
            member_weight = (
                max(0.0, float(regime_positive_weights.get(source_policy_id, 0.0)))
                if any(weight > 0 for weight in regime_positive_weights.values())
                else base_member_weight
            )
            if member_weight <= 0:
                continue
            member_policy = _resolve_execution_policy(
                session,
                state,
                settings,
                regime,
                policy_override_id=source_policy_id,
                asof_date=preview_asof.date(),
            )
            allowed_templates = list(member_policy.get("allowed_templates") or [])
            if not allowed_templates:
                continue
            member_generated = generate_signals_for_policy(
                session=session,
                store=store,
                dataset_id=dataset_id,
                bundle_id=bundle_id,
                asof=preview_asof,
                timeframes=timeframes,
                allowed_templates=allowed_templates,
                params_overrides=member_policy.get("params", {}),
                max_symbols_scan=max_symbols_scan,
                seed=int(seed) + (source_policy_id * 9973),
                mode="preview",
                symbol_scope=symbol_scope,
                ranking_weights=member_policy.get("ranking_weights", {}),
                max_runtime_seconds=max_runtime_seconds,
            )
            scan_truncated_any = scan_truncated_any or bool(member_generated.scan_truncated)
            scanned_symbols_total += int(member_generated.scanned_symbols)
            evaluated_candidates_total += int(member_generated.evaluated_candidates)
            total_symbols_max = max(total_symbols_max, int(member_generated.total_symbols))
            for signal in member_generated.signals:
                row = dict(signal)
                row["source_policy_id"] = source_policy_id
                row["source_policy_name"] = str(
                    member.get("policy_name")
                    or member_policy.get("policy_name")
                    or f"Policy {source_policy_id}"
                )
                row["ensemble_id"] = int(active_ensemble.id or 0)
                row["ensemble_name"] = active_ensemble.name
                row["ensemble_member_weight"] = member_weight
                merged_signals.append(row)
        merged_signals.sort(
            key=lambda row: (
                int(row.get("source_policy_id") or 0),
                -float(row.get("signal_strength", 0.0)),
                str(row.get("symbol", "")),
                str(row.get("side", "BUY")),
            )
        )
        generated = SignalGenerationResult(
            signals=merged_signals,
            scan_truncated=scan_truncated_any,
            scanned_symbols=scanned_symbols_total,
            evaluated_candidates=evaluated_candidates_total,
            total_symbols=total_symbols_max,
        )
    else:
        generated = generate_signals_for_policy(
            session=session,
            store=store,
            dataset_id=dataset_id,
            bundle_id=bundle_id,
            asof=preview_asof,
            timeframes=timeframes,
            allowed_templates=policy.get("allowed_templates", []),
            params_overrides=policy.get("params", {}),
            max_symbols_scan=max_symbols_scan,
            seed=seed,
            mode="preview",
            symbol_scope=symbol_scope,
            ranking_weights=policy.get("ranking_weights", {}),
            max_runtime_seconds=max_runtime_seconds,
        )
    ensemble_payload = (
        serialize_policy_ensemble(session, active_ensemble, include_members=True)
        if active_ensemble is not None
        else None
    )

    return {
        "regime": regime,
        "policy_mode": policy.get("mode"),
        "policy_selection_reason": policy.get("selection_reason"),
        "policy_status": policy.get("policy_status"),
        "health_status": policy.get("health_status"),
        "health_reasons": policy.get("health_reasons", []),
        "signals_source": "generated",
        "generated_signals_count": len(generated.signals),
        "selected_signals_count": 0,
        "signals": generated.signals,
        "bundle_id": bundle_id,
        "dataset_id": dataset_id,
        "timeframes": timeframes,
        "symbol_scope": symbol_scope,
        "scan_truncated": generated.scan_truncated,
        "scanned_symbols": generated.scanned_symbols,
        "evaluated_candidates": generated.evaluated_candidates,
        "total_symbols": generated.total_symbols,
        "ensemble": ensemble_payload,
        "ensemble_weights_source": (
            "regime"
            if isinstance(ensemble_regime_weights.get(str(regime).strip().upper(), {}), dict)
            and bool(ensemble_regime_weights.get(str(regime).strip().upper(), {}))
            else "base"
        )
        if active_ensemble is not None
        else None,
    }


def get_paper_state_payload(session: Session, settings: Settings) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    return {
        "state": _dump_model(state),
        "positions": [_dump_model(p) for p in get_positions(session)],
        "orders": [_dump_model(o) for o in get_orders(session)],
    }


def update_runtime_settings(
    session: Session, settings: Settings, payload: dict[str, Any]
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    merged = dict(state.settings_json)
    if "data_quality_stale_severity" in payload:
        merged["data_quality_stale_severity_override"] = True
    if "operate_mode" in payload and "data_quality_stale_severity" not in payload:
        requested_mode = str(payload.get("operate_mode", settings.operate_mode)).strip().lower()
        if requested_mode == "live":
            merged["data_quality_stale_severity"] = "FAIL"
            merged["data_quality_stale_severity_override"] = False
    if "operate_mode" in payload and "data_quality_stale_severity" in payload:
        requested_mode = str(payload.get("operate_mode", settings.operate_mode)).strip().lower()
        if requested_mode == "live":
            merged["data_quality_stale_severity_override"] = True
    merged.update(payload)
    state.settings_json = merged
    session.add(state)
    session.commit()
    session.refresh(state)
    _log(session, "settings_updated", {"keys": sorted(payload.keys())})
    session.commit()
    return {"settings": state.settings_json}


def activate_policy_mode(session: Session, settings: Settings, policy: Policy) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    merged = dict(state.settings_json or {})
    merged["paper_mode"] = "policy"
    merged["active_policy_id"] = policy.id
    merged["active_policy_name"] = policy.name
    merged["active_ensemble_id"] = None
    merged["active_ensemble_name"] = None
    state.settings_json = merged
    session.add(state)
    _log(
        session,
        "policy_promoted_to_paper",
        {"policy_id": policy.id, "policy_name": policy.name, "paper_mode": "policy"},
    )
    session.commit()
    session.refresh(state)
    return {
        "paper_mode": "policy",
        "active_policy_id": policy.id,
        "active_policy_name": policy.name,
    }
