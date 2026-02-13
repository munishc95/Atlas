from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.engine.simulator import SimulationConfig, SimulationResult, run_simulation


@dataclass
class BacktestConfig:
    risk_per_trade: float = 0.005
    max_positions: int = 3
    initial_equity: float = 1_000_000.0
    commission_bps: float = 5.0
    slippage_base_bps: float = 2.0
    slippage_vol_factor: float = 15.0
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 2.0
    take_profit_r: float | None = None
    time_stop_bars: int | None = None
    min_notional: float = 2_000_000.0
    max_position_value_pct_adv: float = 0.01
    adv_lookback: int = 20
    allow_long: bool = True
    allow_short: bool = False
    instrument_kind: str = "EQUITY_CASH"
    lot_size: int = 1
    futures_initial_margin_pct: float = 0.18
    equity_short_intraday_only: bool = True
    squareoff_time: str = "15:20"
    cost_model_enabled: bool = False
    cost_mode: str = "delivery"
    cost_params: dict[str, float] = field(default_factory=dict)
    seed: int = 7


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    skipped_signals: list[dict[str, Any]] = field(default_factory=list)


def _to_simulation_config(config: BacktestConfig) -> SimulationConfig:
    return SimulationConfig(
        risk_per_trade=config.risk_per_trade,
        max_positions=config.max_positions,
        initial_equity=config.initial_equity,
        commission_bps=config.commission_bps,
        slippage_base_bps=config.slippage_base_bps,
        slippage_vol_factor=config.slippage_vol_factor,
        atr_period=config.atr_period,
        atr_stop_mult=config.atr_stop_mult,
        atr_trail_mult=config.atr_trail_mult,
        take_profit_r=config.take_profit_r,
        time_stop_bars=config.time_stop_bars,
        min_notional=config.min_notional,
        max_position_value_pct_adv=config.max_position_value_pct_adv,
        adv_lookback=config.adv_lookback,
        allow_long=config.allow_long,
        allow_short=config.allow_short,
        instrument_kind=config.instrument_kind,
        lot_size=config.lot_size,
        futures_initial_margin_pct=config.futures_initial_margin_pct,
        equity_short_intraday_only=config.equity_short_intraday_only,
        squareoff_time=config.squareoff_time,
        cost_model_enabled=config.cost_model_enabled,
        cost_mode=config.cost_mode,
        cost_params=dict(config.cost_params),
        seed=config.seed,
    )


def run_backtest(
    price_df: pd.DataFrame,
    entries: pd.Series | dict[str, pd.Series],
    symbol: str,
    config: BacktestConfig,
) -> BacktestResult:
    """Run deterministic side/instrument-aware simulation with next-bar fills."""
    result: SimulationResult = run_simulation(
        price_df=price_df,
        entries=entries,
        symbol=symbol,
        config=_to_simulation_config(config),
    )
    return BacktestResult(
        trades=result.trades,
        equity_curve=result.equity_curve,
        metrics=result.metrics,
        metadata=result.metadata,
        skipped_signals=result.skipped_signals,
    )
