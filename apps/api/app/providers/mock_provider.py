from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from sqlmodel import Session

from app.services.data_store import DataStore
from app.providers.base import BaseProvider, DataProvider


class MockProvider(DataProvider, BaseProvider):
    """Synthetic data provider for deterministic tests."""

    kind = "MOCK"

    def __init__(
        self,
        seed: int = 7,
        *,
        session: Session | None = None,
        store: DataStore | None = None,
    ) -> None:
        BaseProvider.__init__(self)
        self.seed = seed
        self.session = session
        self.store = store

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
        frame["high"] = frame[["open", "close"]].max(axis=1) * (
            1 + rng.uniform(0.001, 0.01, periods)
        )
        frame["low"] = frame[["open", "close"]].min(axis=1) * (
            1 - rng.uniform(0.001, 0.01, periods)
        )
        frame["volume"] = rng.integers(200_000, 1_200_000, periods)

        if start is not None:
            frame = frame[frame["datetime"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            frame = frame[frame["datetime"] <= pd.Timestamp(end, tz="UTC")]
        return frame[["datetime", "open", "high", "low", "close", "volume"]].reset_index(drop=True)

    def get_corporate_actions(self, symbol: str) -> list[dict[str, object]]:
        return []

    def list_symbols(self, bundle_id: int) -> list[str]:
        if self.session is not None and self.store is not None:
            scoped = self.store.get_bundle_symbols(self.session, int(bundle_id), timeframe="1d")
            if scoped:
                return scoped
        return self.get_symbols()

    def fetch_ohlc(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        output: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            self.count_api_call(1)
            output[str(symbol).upper()] = self.get_ohlcv(
                str(symbol).upper(),
                timeframe=timeframe,
                start=start,
                end=end,
            )
        return output

    def supports_timeframes(self) -> set[str]:
        return {"1d", "4h_ish", "4h_ish_resampled"}
