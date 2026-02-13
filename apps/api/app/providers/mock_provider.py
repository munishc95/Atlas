from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from app.providers.base import DataProvider


class MockProvider(DataProvider):
    """Synthetic data provider for deterministic tests."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed

    def get_symbols(self) -> list[str]:
        return ["MOCK1", "MOCK2", "MOCK3"]

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed + hash(symbol) % 1_000)
        periods = 300
        dt_index = pd.date_range("2022-01-01", periods=periods, freq="D", tz="UTC")
        returns = rng.normal(0.0005, 0.015, size=periods)
        close = 100 * (1 + pd.Series(returns)).cumprod()

        frame = pd.DataFrame({"datetime": dt_index, "close": close})
        frame["open"] = frame["close"].shift(1).fillna(frame["close"])
        frame["high"] = frame[["open", "close"]].max(axis=1) * (1 + rng.uniform(0.001, 0.01, periods))
        frame["low"] = frame[["open", "close"]].min(axis=1) * (1 - rng.uniform(0.001, 0.01, periods))
        frame["volume"] = rng.integers(200_000, 1_200_000, periods)

        if start is not None:
            frame = frame[frame["datetime"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            frame = frame[frame["datetime"] <= pd.Timestamp(end, tz="UTC")]
        return frame[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def get_corporate_actions(self, symbol: str) -> list[dict[str, object]]:
        return []
