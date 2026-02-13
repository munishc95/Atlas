from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.engine.backtester import BacktestConfig, run_backtest


def _frame(periods: int = 40, start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=periods, freq="D", tz="UTC")
    close = pd.Series(np.linspace(start_price, start_price + periods - 1, periods))
    frame = pd.DataFrame({"datetime": idx})
    frame["open"] = close
    frame["high"] = close + 2
    frame["low"] = close - 2
    frame["close"] = close
    frame["volume"] = 2_000_000
    return frame


def test_next_bar_fill_timing() -> None:
    frame = _frame(30)
    signals = pd.Series(False, index=frame.index)
    signals.iloc[15] = True

    result = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(min_notional=0.0),
    )

    assert not result.trades.empty
    expected_fill_time = pd.Timestamp(frame.iloc[16]["datetime"])
    actual_fill_time = pd.Timestamp(result.trades.iloc[0]["entry_dt"])
    assert actual_fill_time == expected_fill_time


def test_slippage_and_commission_reduce_pnl() -> None:
    frame = _frame(35)
    signals = pd.Series(False, index=frame.index)
    signals.iloc[15] = True

    low_cost = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(min_notional=0.0, commission_bps=0.0, slippage_base_bps=0.0),
    )

    high_cost = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(min_notional=0.0, commission_bps=20.0, slippage_base_bps=10.0),
    )

    assert low_cost.trades.iloc[0]["pnl"] > high_cost.trades.iloc[0]["pnl"]


def test_atr_sizing_uses_risk_amount_over_stop_distance() -> None:
    frame = _frame(50)
    signals = pd.Series(False, index=frame.index)
    signals.iloc[20] = True

    cfg = BacktestConfig(
        initial_equity=100_000.0,
        risk_per_trade=0.005,
        atr_period=14,
        atr_stop_mult=2.0,
        min_notional=0.0,
        commission_bps=0.0,
        slippage_base_bps=0.0,
    )

    result = run_backtest(price_df=frame, entries=signals, symbol="TEST", config=cfg)

    assert not result.trades.empty
    qty = int(result.trades.iloc[0]["qty"])

    # Range is constant at 4 points, ATR converges close to 4 => stop_distance ~ 8.
    expected = math.floor((cfg.initial_equity * cfg.risk_per_trade) / 8.0)
    assert qty in {expected, expected - 1, expected + 1}


def test_stop_and_trailing_exit_reason() -> None:
    frame = _frame(45)
    frame.loc[30:, "close"] = frame.loc[30:, "close"] - np.linspace(0, 30, len(frame.loc[30:]))
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1) + 1
    frame["low"] = frame[["open", "close"]].min(axis=1) - 3

    signals = pd.Series(False, index=frame.index)
    signals.iloc[15] = True

    result = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(min_notional=0.0, atr_stop_mult=1.5, atr_trail_mult=1.0),
    )

    assert not result.trades.empty
    assert "STOP_HIT" in set(result.trades["reason"].tolist())


def test_max_positions_enforced_to_three() -> None:
    frame = _frame(60)
    signals = pd.Series(True, index=frame.index)

    result = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(max_positions=3, min_notional=0.0, time_stop_bars=None),
    )

    # With perpetual entries and no configured time-stop, the engine should only open up to 3 positions.
    assert len(result.trades) <= 3


def test_metrics_sanity_keys_and_ranges() -> None:
    frame = _frame(55)
    signals = pd.Series(False, index=frame.index)
    signals.iloc[20] = True
    signals.iloc[30] = True

    result = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(min_notional=0.0),
    )

    keys = {
        "cagr",
        "max_drawdown",
        "calmar",
        "sharpe",
        "sortino",
        "win_rate",
        "avg_win",
        "avg_loss",
        "profit_factor",
        "exposure_pct",
        "turnover",
        "avg_holding_period_bars",
        "cvar_95",
        "tail_loss_norm",
    }
    assert keys.issubset(result.metrics.keys())
    assert 0.0 <= result.metrics["tail_loss_norm"] <= 1.0


def test_adv_cap_limits_position_size() -> None:
    frame = _frame(50)
    frame["volume"] = 500  # Notional around 50k-75k in this synthetic series.
    signals = pd.Series(False, index=frame.index)
    signals.iloc[20] = True

    result = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(
            min_notional=0.0,
            commission_bps=0.0,
            slippage_base_bps=0.0,
            max_position_value_pct_adv=0.01,
        ),
    )

    assert not result.trades.empty
    qty = int(result.trades.iloc[0]["qty"])
    assert qty <= 6


def test_gap_through_stop_fills_at_open_worse_than_stop() -> None:
    frame = _frame(45)
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1) + 0.5
    frame["low"] = frame[["open", "close"]].min(axis=1) - 0.05

    # Force a large downside gap after entry to validate gap-through-stop handling.
    gap_idx = 25
    frame.loc[gap_idx, "open"] = frame.loc[gap_idx - 1, "close"] * 0.72
    frame.loc[gap_idx, "high"] = frame.loc[gap_idx, "open"] + 1.0
    frame.loc[gap_idx, "low"] = frame.loc[gap_idx, "open"] - 2.0
    frame.loc[gap_idx, "close"] = frame.loc[gap_idx, "open"] + 0.2

    signals = pd.Series(False, index=frame.index)
    signals.iloc[15] = True

    result = run_backtest(
        price_df=frame,
        entries=signals,
        symbol="TEST",
        config=BacktestConfig(
            min_notional=0.0,
            atr_stop_mult=2.0,
            atr_trail_mult=4.0,
        ),
    )

    assert not result.trades.empty
    stop_trade = result.trades[result.trades["reason"] == "STOP_HIT"].iloc[0]
    gap_open = float(frame.loc[gap_idx, "open"])
    assert float(stop_trade["exit_px"]) <= gap_open
