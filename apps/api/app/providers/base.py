from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class DataProvider(ABC):
    """Interface for market data providers."""

    @abstractmethod
    def get_symbols(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_corporate_actions(self, symbol: str) -> list[dict[str, object]]:
        raise NotImplementedError


class BaseProvider(ABC):
    """Batch provider interface for automated update runs."""

    kind: str = "BASE"

    def __init__(self) -> None:
        self._api_calls_made = 0

    def reset_counters(self) -> None:
        self._api_calls_made = 0

    def count_api_call(self, count: int = 1) -> None:
        self._api_calls_made += max(0, int(count))

    @property
    def api_calls_made(self) -> int:
        return int(self._api_calls_made)

    def missing_mapped_symbols(self, symbols: list[str]) -> set[str]:
        return set()

    @abstractmethod
    def list_symbols(self, bundle_id: int) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def fetch_ohlc(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def supports_timeframes(self) -> set[str]:
        raise NotImplementedError
