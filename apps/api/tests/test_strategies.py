from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.templates import generate_signal_sides, generate_signals, list_templates


def _signal_frame(periods: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    trend = np.linspace(100, 300, periods)
    oscillation = np.sin(np.linspace(0, 35, periods)) * 4
    close = trend + oscillation

    frame = pd.DataFrame({"datetime": idx})
    frame["close"] = close
    frame["open"] = frame["close"].shift(1).fillna(frame["close"] * 0.998)
    frame["high"] = frame[["open", "close"]].max(axis=1) + 0.2
    frame["low"] = frame[["open", "close"]].min(axis=1) - 0.2
    frame["volume"] = 3_000_000 + (np.cos(np.linspace(0, 6, periods)) * 300_000)
    return frame


def test_templates_emit_boolean_series() -> None:
    frame = _signal_frame()
    for template in list_templates():
        signals = generate_signals(template.key, frame, params=template.default_params)
        assert len(signals) == len(frame)
        assert signals.dtype == bool


def test_templates_produce_at_least_one_candidate_signal() -> None:
    base = _signal_frame()

    trend_signals = generate_signals("trend_breakout", base)
    assert trend_signals.any()

    pullback = base.copy()
    dip_idx = pullback.index[120:150]
    pullback.loc[dip_idx, "close"] = pullback.loc[dip_idx, "close"] * 0.93
    pullback.loc[dip_idx, "open"] = (
        pullback.loc[dip_idx, "close"].shift(1).fillna(pullback.loc[dip_idx, "close"])
    )
    pullback_signals = generate_signals("pullback_trend", pullback)
    assert pullback_signals.any()

    squeeze = base.copy()
    squeeze_block = squeeze.index[140:180]
    squeeze.loc[squeeze_block, "high"] = squeeze.loc[squeeze_block, "close"] + 0.05
    squeeze.loc[squeeze_block, "low"] = squeeze.loc[squeeze_block, "close"] - 0.05
    squeeze.loc[squeeze.index[180:200], "close"] = squeeze.loc[squeeze.index[180:200], "close"] + 8
    squeeze_signals = generate_signals("squeeze_breakout", squeeze)
    assert squeeze_signals.any()


def test_trend_breakout_generates_sell_when_direction_both() -> None:
    periods = 280
    idx = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    close = np.linspace(320, 110, periods)
    close[-25:] = close[-25:] - np.linspace(0, 28, 25)
    frame = pd.DataFrame({"datetime": idx, "close": close})
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1) + 0.3
    frame["low"] = frame[["open", "close"]].min(axis=1) - 0.3
    frame["volume"] = 3_200_000

    both = generate_signal_sides("trend_breakout", frame, params={"direction": "both"})
    long_only = generate_signal_sides("trend_breakout", frame, params={"direction": "long"})
    assert both["SELL"].any()
    assert not long_only["SELL"].any()


def test_pullback_trend_generates_sell_when_direction_both() -> None:
    periods = 320
    idx = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    baseline = np.linspace(400, 120, periods)
    bounce = np.sin(np.linspace(0, 42, periods)) * 10
    close = baseline + bounce
    close[-60:-40] = close[-60:-40] + 22
    frame = pd.DataFrame({"datetime": idx, "close": close})
    frame["open"] = frame["close"].shift(1).fillna(frame["close"])
    frame["high"] = frame[["open", "close"]].max(axis=1) + 0.5
    frame["low"] = frame[["open", "close"]].min(axis=1) - 0.5
    frame["volume"] = 2_900_000

    both = generate_signal_sides(
        "pullback_trend",
        frame,
        params={"direction": "both", "short_rules": {"rsi_overbought": 65, "pullback_band": 1.01}},
    )
    long_only = generate_signal_sides(
        "pullback_trend",
        frame,
        params={"direction": "long", "short_rules": {"rsi_overbought": 65, "pullback_band": 1.01}},
    )
    assert both["SELL"].any()
    assert not long_only["SELL"].any()


def test_squeeze_breakout_generates_sell_when_direction_both() -> None:
    base = _signal_frame(320)
    squeeze = base.copy()
    squeeze_window = squeeze.index[180:230]
    squeeze.loc[squeeze.index[230:250], "close"] = squeeze.loc[squeeze.index[230:250], "close"] - 30
    squeeze["open"] = squeeze["close"].shift(1).fillna(squeeze["close"])
    squeeze["high"] = squeeze[["open", "close"]].max(axis=1) + 0.2
    squeeze["low"] = squeeze[["open", "close"]].min(axis=1) - 0.2
    squeeze["volume"] = 4_500_000
    squeeze.loc[squeeze_window, "open"] = squeeze.loc[squeeze_window, "close"]
    squeeze.loc[squeeze_window, "high"] = squeeze.loc[squeeze_window, "close"] + 0.05
    squeeze.loc[squeeze_window, "low"] = squeeze.loc[squeeze_window, "close"] - 0.05

    params = {
        "direction": "both",
        "breakout_lookback": 5,
        "bb_std": 1.0,
        "kc_mult": 2.5,
        "short_rules": {"require_squeeze": False},
    }
    both = generate_signal_sides("squeeze_breakout", squeeze, params=params)
    long_only = generate_signal_sides(
        "squeeze_breakout",
        squeeze,
        params={
            "direction": "long",
            "breakout_lookback": 5,
            "bb_std": 1.0,
            "kc_mult": 2.5,
            "short_rules": {"require_squeeze": False},
        },
    )
    assert both["SELL"].any()
    assert not long_only["SELL"].any()
