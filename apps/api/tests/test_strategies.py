from __future__ import annotations

import numpy as np
import pandas as pd

from app.strategies.templates import generate_signals, list_templates


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
