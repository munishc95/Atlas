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
