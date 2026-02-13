from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pandas as pd
from sqlmodel import Session, select

from app.db.models import Dataset, Symbol


class DataStore:
    """Persists OHLCV to Parquet and serves analytical reads through DuckDB."""

    def __init__(self, parquet_root: str, duckdb_path: str) -> None:
        self.parquet_root = Path(parquet_root)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path = duckdb_path

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        return self.parquet_root / f"symbol={symbol}" / f"timeframe={timeframe}" / "ohlcv.parquet"

    def save_ohlcv(
        self,
        session: Session,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
        provider: str,
        checksum: str | None = None,
    ) -> Dataset:
        clean = frame.copy()
        clean["datetime"] = pd.to_datetime(clean["datetime"], utc=True)
        clean = clean.sort_values("datetime")

        path = self._parquet_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(path, index=False)

        if session.exec(select(Symbol).where(Symbol.symbol == symbol)).first() is None:
            session.add(Symbol(symbol=symbol, name=symbol))

        dataset = Dataset(
            provider=provider,
            symbol=symbol,
            timeframe=timeframe,
            start_date=clean["datetime"].dt.date.min(),
            end_date=clean["datetime"].dt.date.max(),
            checksum=checksum,
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        return dataset

    def list_symbols(self, session: Session) -> list[str]:
        rows = session.exec(select(Symbol).order_by(Symbol.symbol)).all()
        return [row.symbol for row in rows]

    def data_status(self, session: Session) -> list[dict[str, object]]:
        rows = session.exec(select(Dataset).order_by(Dataset.symbol, Dataset.timeframe)).all()
        return [
            {
                "id": row.id,
                "provider": row.provider,
                "symbol": row.symbol,
                "timeframe": row.timeframe,
                "start_date": row.start_date.isoformat(),
                "end_date": row.end_date.isoformat(),
                "checksum": row.checksum,
                "last_updated": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def load_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        path = self._parquet_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

        with duckdb.connect(self.duckdb_path) as conn:
            query = f"SELECT * FROM read_parquet('{path.as_posix()}')"
            clauses: list[str] = []
            params: list[object] = []

            if start is not None:
                clauses.append("datetime >= ?")
                params.append(start.astimezone(UTC).isoformat())
            if end is not None:
                clauses.append("datetime <= ?")
                params.append(end.astimezone(UTC).isoformat())
            if clauses:
                query += " WHERE " + " AND ".join(clauses)
            query += " ORDER BY datetime"

            frame = conn.execute(query, params).df()

        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        return frame
