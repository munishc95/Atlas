from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.engine.indicators import adx, atr, bollinger_bands, sma
from app.services.data_store import DataStore


REGIME_TREND_UP = "TREND_UP"
REGIME_RANGE = "RANGE"
REGIME_HIGH_VOL = "HIGH_VOL"
REGIME_RISK_OFF = "RISK_OFF"


def classify_regime(frame: pd.DataFrame) -> str:
    if frame.empty or len(frame) < 220:
        return REGIME_RANGE

    df = frame.copy().sort_values("datetime")
    close = df["close"]
    sma200 = sma(close, 200)
    adx14 = adx(df, 14)
    atr_pct = atr(df, 14) / close.replace(0, np.nan)
    bb = bollinger_bands(close, period=20, std_mult=2.0)

    last_close = float(close.iloc[-1])
    last_sma200 = float(sma200.iloc[-1]) if not np.isnan(sma200.iloc[-1]) else last_close
    last_adx = float(adx14.iloc[-1]) if not np.isnan(adx14.iloc[-1]) else 15.0
    last_atr_pct = float(atr_pct.iloc[-1]) if not np.isnan(atr_pct.iloc[-1]) else 0.0
    vol_threshold = float(atr_pct.quantile(0.8)) if atr_pct.notna().any() else 0.0
    width_threshold = float(bb["width"].quantile(0.25)) if bb["width"].notna().any() else 0.0
    last_width = float(bb["width"].iloc[-1]) if not np.isnan(bb["width"].iloc[-1]) else 0.0

    vol_high = last_atr_pct >= vol_threshold and vol_threshold > 0

    if last_close < last_sma200 and vol_high:
        return REGIME_RISK_OFF
    if last_close > last_sma200 and last_adx > 20:
        return REGIME_TREND_UP
    if vol_high:
        return REGIME_HIGH_VOL
    if last_adx < 18 and last_width <= width_threshold:
        return REGIME_RANGE
    return REGIME_RANGE


def regime_policy(regime: str, base_risk: float, base_max_positions: int) -> dict[str, Any]:
    if regime == REGIME_TREND_UP:
        return {
            "allowed_templates": ["trend_breakout", "pullback_trend"],
            "risk_per_trade": base_risk,
            "max_positions": base_max_positions,
        }
    if regime == REGIME_RANGE:
        return {
            "allowed_templates": ["pullback_trend", "squeeze_breakout"],
            "risk_per_trade": base_risk * 0.75,
            "max_positions": min(base_max_positions, 3),
        }
    if regime == REGIME_HIGH_VOL:
        return {
            "allowed_templates": ["pullback_trend", "squeeze_breakout"],
            "risk_per_trade": min(base_risk, 0.0025),
            "max_positions": min(base_max_positions, 2),
        }
    return {
        "allowed_templates": [],
        "risk_per_trade": 0.0,
        "max_positions": 0,
    }


def current_regime_payload(store: DataStore, symbol: str, timeframe: str = "1d") -> dict[str, Any]:
    frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
    regime = classify_regime(frame)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "regime": regime,
    }
