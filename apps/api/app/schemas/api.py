from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DataEnvelope(BaseModel):
    data: Any
    meta: dict[str, Any] | None = None


class ImportConfig(BaseModel):
    symbol: str
    timeframe: str = Field(default="1d")
    provider: str = Field(default="csv")
    mapping: dict[str, str] | None = None


class BacktestRunRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    strategy_template: str
    params: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    strategy_id: int | None = None


class WalkForwardRunRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    strategy_template: str
    config: dict[str, Any] = Field(default_factory=dict)


class ResearchRunRequest(BaseModel):
    bundle_id: int | None = None
    dataset_id: int | None = None
    timeframes: list[str] = Field(default_factory=lambda: ["1d"])
    strategy_templates: list[str] = Field(
        default_factory=lambda: ["trend_breakout", "pullback_trend", "squeeze_breakout"]
    )
    symbol_scope: str = "liquid"
    config: dict[str, Any] = Field(default_factory=dict)


class PromoteStrategyRequest(BaseModel):
    strategy_id: int | None = None
    strategy_name: str
    template: str
    params_json: dict[str, Any] = Field(default_factory=dict)


class CreatePolicyRequest(BaseModel):
    research_run_id: int
    name: str


class PaperRunStepRequest(BaseModel):
    regime: str = "TREND_UP"
    signals: list[dict[str, Any]] = Field(default_factory=list)
    mark_prices: dict[str, float] = Field(default_factory=dict)
    auto_generate_signals: bool = False
    bundle_id: int | None = None
    dataset_id: int | None = None
    timeframes: list[str] = Field(default_factory=list)
    symbol_scope: str | None = None
    max_symbols_scan: int | None = None
    max_runtime_seconds: int | None = None
    seed: int | None = None
    asof: str | None = None


class PaperSignalsPreviewRequest(BaseModel):
    regime: str = "TREND_UP"
    bundle_id: int | None = None
    dataset_id: int | None = None
    policy_id: int | None = None
    timeframes: list[str] = Field(default_factory=list)
    symbol_scope: str | None = None
    max_symbols_scan: int | None = None
    max_runtime_seconds: int | None = None
    seed: int | None = None
    asof: str | None = None


class DailyReportGenerateRequest(BaseModel):
    date: str | None = None
    bundle_id: int | None = None
    policy_id: int | None = None


class DataQualityRunRequest(BaseModel):
    bundle_id: int
    timeframe: str = "1d"


class DataUpdatesRunRequest(BaseModel):
    bundle_id: int
    timeframe: str = "1d"
    max_files_per_run: int | None = None


class ProviderUpdatesRunRequest(BaseModel):
    bundle_id: int
    timeframe: str = "1d"
    provider_kind: str | None = None
    max_symbols_per_run: int | None = None
    max_calls_per_run: int | None = None
    start: str | None = None
    end: str | None = None


class OperateRunRequest(BaseModel):
    bundle_id: int | None = None
    timeframe: str | None = None
    regime: str | None = None
    policy_id: int | None = None
    include_data_updates: bool | None = None
    provider_kind: str | None = None
    provider_max_symbols_per_run: int | None = None
    provider_max_calls_per_run: int | None = None
    provider_start: str | None = None
    provider_end: str | None = None
    date: str | None = None
    asof: str | None = None
    seed: int | None = None


class AutoEvalRunRequest(BaseModel):
    bundle_id: int | None = None
    active_policy_id: int | None = None
    challenger_policy_ids: list[int] | None = None
    timeframe: str | None = None
    lookback_trading_days: int | None = None
    min_trades: int | None = None
    asof_date: str | None = None
    seed: int | None = None
    auto_switch: bool | None = None


class MonthlyReportGenerateRequest(BaseModel):
    month: str | None = None  # YYYY-MM
    bundle_id: int | None = None
    policy_id: int | None = None


class PolicyEvaluationRunRequest(BaseModel):
    bundle_id: int
    champion_policy_id: int
    challenger_policy_ids: list[int] | None = None
    regime: str | None = None
    window_days: int | None = None
    start_date: str | None = None
    end_date: str | None = None
    seed: int | None = None


class ReplayRunRequest(BaseModel):
    bundle_id: int
    policy_id: int
    regime: str | None = None
    start_date: str
    end_date: str
    seed: int | None = None
    window_days: int | None = None


class RuntimeSettingsRequest(BaseModel):
    risk_per_trade: float | None = None
    max_positions: int | None = None
    kill_switch_dd: float | None = None
    cooldown_days: int | None = None
    commission_bps: float | None = None
    slippage_base_bps: float | None = None
    slippage_vol_factor: float | None = None
    max_position_value_pct_adv: float | None = None
    diversification_corr_threshold: float | None = None
    allowed_sides: list[str] | None = None
    paper_short_squareoff_time: str | None = None
    autopilot_max_symbols_scan: int | None = None
    autopilot_max_runtime_seconds: int | None = None
    reports_auto_generate_daily: bool | None = None
    health_window_days_short: int | None = None
    health_window_days_long: int | None = None
    drift_maxdd_multiplier: float | None = None
    drift_negative_return_cost_ratio_threshold: float | None = None
    drift_win_rate_drop_pct: float | None = None
    drift_return_delta_threshold: float | None = None
    drift_warning_risk_scale: float | None = None
    drift_degraded_risk_scale: float | None = None
    drift_degraded_action: str | None = None
    evaluations_auto_promote_enabled: bool | None = None
    evaluations_min_window_days: int | None = None
    evaluations_score_margin: float | None = None
    evaluations_max_dd_multiplier: float | None = None
    four_hour_bars: str | None = None
    paper_mode: str | None = None
    active_policy_id: int | None = None
    cost_model_enabled: bool | None = None
    cost_mode: str | None = None
    brokerage_bps: float | None = None
    stt_delivery_buy_bps: float | None = None
    stt_delivery_sell_bps: float | None = None
    stt_intraday_buy_bps: float | None = None
    stt_intraday_sell_bps: float | None = None
    exchange_txn_bps: float | None = None
    sebi_bps: float | None = None
    stamp_delivery_buy_bps: float | None = None
    stamp_intraday_buy_bps: float | None = None
    gst_rate: float | None = None
    futures_brokerage_bps: float | None = None
    futures_stt_sell_bps: float | None = None
    futures_exchange_txn_bps: float | None = None
    futures_stamp_buy_bps: float | None = None
    futures_initial_margin_pct: float | None = None
    futures_symbol_mapping_strategy: str | None = None
    paper_use_simulator_engine: bool | None = None
    trading_calendar_segment: str | None = None
    operate_safe_mode_on_fail: bool | None = None
    operate_safe_mode_action: str | None = None
    operate_mode: str | None = None
    data_quality_stale_severity: str | None = None
    data_quality_max_stale_minutes_1d: int | None = None
    data_quality_max_stale_minutes_intraday: int | None = None
    operate_auto_run_enabled: bool | None = None
    operate_auto_run_time_ist: str | None = None
    operate_auto_run_include_data_updates: bool | None = None
    operate_auto_eval_enabled: bool | None = None
    operate_auto_eval_frequency: str | None = None
    operate_auto_eval_day_of_week: int | None = None
    operate_auto_eval_time_ist: str | None = None
    operate_auto_eval_lookback_trading_days: int | None = None
    operate_auto_eval_min_trades: int | None = None
    operate_auto_eval_cooldown_trading_days: int | None = None
    operate_auto_eval_max_switches_per_30d: int | None = None
    operate_auto_eval_auto_switch: bool | None = None
    operate_auto_eval_shadow_only_gate: bool | None = None
    operate_max_stale_minutes_1d: int | None = None
    operate_max_stale_minutes_4h_ish: int | None = None
    operate_max_gap_bars: int | None = None
    operate_outlier_zscore: float | None = None
    operate_cost_ratio_spike_threshold: float | None = None
    operate_cost_ratio_spike_days: int | None = None
    operate_cost_spike_risk_scale: float | None = None
    operate_scan_truncated_warn_days: int | None = None
    operate_scan_truncated_reduce_to: int | None = None
    data_updates_inbox_enabled: bool | None = None
    data_updates_max_files_per_run: int | None = None
    data_updates_provider_enabled: bool | None = None
    data_updates_provider_kind: str | None = None
    data_updates_provider_max_symbols_per_run: int | None = None
    data_updates_provider_max_calls_per_run: int | None = None
    data_updates_provider_timeframe_enabled: str | None = None
    coverage_missing_latest_warn_pct: float | None = None
    coverage_missing_latest_fail_pct: float | None = None
    coverage_inactive_after_missing_days: int | None = None
    risk_overlay_enabled: bool | None = None
    risk_overlay_target_vol_annual: float | None = None
    risk_overlay_lookback_days: int | None = None
    risk_overlay_min_scale: float | None = None
    risk_overlay_max_scale: float | None = None
    risk_overlay_max_gross_exposure_pct: float | None = None
    risk_overlay_max_single_name_exposure_pct: float | None = None
    risk_overlay_max_sector_exposure_pct: float | None = None
    risk_overlay_corr_clamp_enabled: bool | None = None
    risk_overlay_corr_threshold: float | None = None
    risk_overlay_corr_reduce_factor: float | None = None
