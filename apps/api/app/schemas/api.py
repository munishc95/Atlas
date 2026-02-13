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
    params: dict[str, float | int] = Field(default_factory=dict)
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
