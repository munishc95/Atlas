from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from app.providers.base import DataProvider


class CSVProvider(DataProvider):
    """Reads OHLCV data from symbol/timeframe-partitioned parquet files."""

    def __init__(self, parquet_root: str) -> None:
        self.parquet_root = Path(parquet_root)

    def get_symbols(self) -> list[str]:
        if not self.parquet_root.exists():
            return []
        symbols = [p.name.split("=")[-1] for p in self.parquet_root.glob("symbol=*")]
        return sorted(set(symbols))

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        path = self.parquet_root / f"symbol={symbol}" / f"timeframe={timeframe}" / "ohlcv.parquet"
        if not path.exists():
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        df = pd.read_parquet(path)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        if start is not None:
            df = df[df["datetime"] >= pd.Timestamp(start, tz="UTC")]
        if end is not None:
            df = df[df["datetime"] <= pd.Timestamp(end, tz="UTC")]
        return df.sort_values("datetime").reset_index(drop=True)

    def get_corporate_actions(self, symbol: str) -> list[dict[str, object]]:
        return []
