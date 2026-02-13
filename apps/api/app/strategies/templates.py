from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from app.engine.indicators import bollinger_bands, ema, keltner_channels, rsi, sma


@dataclass
class StrategyTemplate:
    key: str
    name: str
    description: str
    default_params: dict[str, float | int]
    param_ranges: dict[str, tuple[float | int, float | int]]
    signal_fn: Callable[[pd.DataFrame, dict[str, float | int]], pd.Series]


def trend_breakout_signals(df: pd.DataFrame, params: dict[str, float | int]) -> pd.Series:
    trend_period = int(params.get("trend_period", 200))
    breakout_lookback = int(params.get("breakout_lookback", 20))

    trend = df["close"] > sma(df["close"], trend_period)
    breakout_level = (
        df["high"].rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
    )
    return (trend & (df["close"] > breakout_level)).fillna(False)


def pullback_trend_signals(df: pd.DataFrame, params: dict[str, float | int]) -> pd.Series:
    ema_fast = ema(df["close"], int(params.get("ema_fast", 20)))
    ema_mid = ema(df["close"], int(params.get("ema_mid", 50)))
    ema_slow = ema(df["close"], int(params.get("ema_slow", 100)))

    rsi_period = int(params.get("rsi_period", 4))
    rsi_oversold = float(params.get("rsi_oversold", 25))
    pullback_band = float(params.get("pullback_band", 0.99))

    trend = (df["close"] > ema_slow) & (ema_fast > ema_mid)
    pullback = rsi(df["close"], rsi_period) < rsi_oversold
    near_band = df["close"] <= (ema_fast * pullback_band)
    return (trend & pullback & near_band).fillna(False)


def squeeze_breakout_signals(df: pd.DataFrame, params: dict[str, float | int]) -> pd.Series:
    bb_period = int(params.get("bb_period", 20))
    bb_std = float(params.get("bb_std", 2.0))
    kc_period = int(params.get("kc_period", 20))
    kc_mult = float(params.get("kc_mult", 1.5))
    breakout_lookback = int(params.get("breakout_lookback", 20))

    bb = bollinger_bands(df["close"], period=bb_period, std_mult=bb_std)
    kc = keltner_channels(df, period=kc_period, atr_mult=kc_mult)

    in_squeeze = (bb["upper"] < kc["upper"]) & (bb["lower"] > kc["lower"])
    breakout_level = (
        df["high"].rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
    )

    vol_confirm = pd.Series(True, index=df.index)
    if "volume" in df.columns:
        vol_ma = df["volume"].rolling(20, min_periods=20).mean()
        vol_confirm = df["volume"] >= vol_ma

    squeeze_prev = (
        pd.Series(in_squeeze.shift(1), index=df.index, dtype="boolean").fillna(False).astype(bool)
    )
    return (
        ((squeeze_prev) & (df["close"] > breakout_level) & vol_confirm).fillna(False).astype(bool)
    )


def list_templates() -> list[StrategyTemplate]:
    return [
        StrategyTemplate(
            key="trend_breakout",
            name="Trend Breakout",
            description="Daily trend filter and 4H-ish breakout with ATR stops.",
            default_params={
                "trend_period": 200,
                "breakout_lookback": 20,
                "atr_stop_mult": 2.0,
                "atr_trail_mult": 2.0,
            },
            param_ranges={
                "trend_period": (120, 260),
                "breakout_lookback": (10, 40),
                "atr_stop_mult": (1.2, 3.5),
                "atr_trail_mult": (1.2, 4.0),
            },
            signal_fn=trend_breakout_signals,
        ),
        StrategyTemplate(
            key="pullback_trend",
            name="Pullback in Trend",
            description="Trend aligned RSI pullback entry with ATR risk controls.",
            default_params={
                "ema_fast": 20,
                "ema_mid": 50,
                "ema_slow": 100,
                "rsi_period": 4,
                "rsi_oversold": 25,
                "pullback_band": 0.99,
                "atr_stop_mult": 2.0,
                "atr_trail_mult": 2.0,
            },
            param_ranges={
                "rsi_period": (2, 7),
                "rsi_oversold": (15, 35),
                "pullback_band": (0.97, 1.0),
                "atr_stop_mult": (1.2, 3.0),
                "atr_trail_mult": (1.0, 3.5),
            },
            signal_fn=pullback_trend_signals,
        ),
        StrategyTemplate(
            key="squeeze_breakout",
            name="Volatility Squeeze Breakout",
            description="BB/KC squeeze followed by breakout and ATR exits.",
            default_params={
                "bb_period": 20,
                "bb_std": 2.0,
                "kc_period": 20,
                "kc_mult": 1.5,
                "breakout_lookback": 20,
                "atr_stop_mult": 2.0,
                "atr_trail_mult": 2.0,
            },
            param_ranges={
                "bb_period": (10, 40),
                "bb_std": (1.5, 2.5),
                "kc_mult": (1.2, 2.0),
                "breakout_lookback": (10, 40),
                "atr_stop_mult": (1.2, 3.0),
                "atr_trail_mult": (1.0, 3.5),
            },
            signal_fn=squeeze_breakout_signals,
        ),
    ]


def get_template(template_key: str) -> StrategyTemplate:
    for template in list_templates():
        if template.key == template_key:
            return template
    raise KeyError(f"Unknown strategy template: {template_key}")


def generate_signals(
    template_key: str,
    df: pd.DataFrame,
    params: dict[str, float | int] | None = None,
) -> pd.Series:
    template = get_template(template_key)
    merged_params = template.default_params.copy()
    if params:
        merged_params.update(params)

    signals = template.signal_fn(df, merged_params)
    if signals.dtype != bool:
        signals = signals.astype(bool)
    return signals.fillna(False)


def signal_strength(df: pd.DataFrame, signal_index: int, lookback: int = 20) -> float:
    if signal_index <= 0 or signal_index >= len(df):
        return 0.0

    price = float(df.iloc[signal_index]["close"])
    recent_high = float(df["high"].iloc[max(0, signal_index - lookback) : signal_index].max())
    trend = float(
        (ema(df["close"], 20).iloc[signal_index] - ema(df["close"], 50).iloc[signal_index])
    )
    breakout_component = (price / recent_high - 1.0) if recent_high > 0 else 0.0
    return float(np.nan_to_num(0.7 * breakout_component + 0.3 * trend / max(price, 1.0)))


def default_template_payload() -> list[dict[str, object]]:
    return [
        {
            "key": t.key,
            "name": t.name,
            "description": t.description,
            "default_params": t.default_params,
            "param_ranges": t.param_ranges,
        }
        for t in list_templates()
    ]
