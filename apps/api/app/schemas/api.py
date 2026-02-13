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


class PromoteStrategyRequest(BaseModel):
    strategy_id: int | None = None
    strategy_name: str
    template: str
    params_json: dict[str, Any] = Field(default_factory=dict)


class PaperRunStepRequest(BaseModel):
    regime: str = "TREND_UP"
    signals: list[dict[str, Any]] = Field(default_factory=list)
    mark_prices: dict[str, float] = Field(default_factory=dict)


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
    four_hour_bars: str | None = None
