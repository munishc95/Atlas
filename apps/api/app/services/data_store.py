from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.db.models import Dataset, Symbol


class DataStore:
    """Persists OHLCV to Parquet and serves analytical reads through DuckDB."""

    def __init__(self, parquet_root: str, duckdb_path: str) -> None:
        self.parquet_root = Path(parquet_root)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path = duckdb_path
        self._adv_cache: dict[tuple[int, str, int], list[tuple[str, float]]] = {}

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
        symbol = symbol.upper()
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
            symbols_json=[symbol],
            start_date=clean["datetime"].dt.date.min(),
            end_date=clean["datetime"].dt.date.max(),
            checksum=checksum,
        )
        session.add(dataset)
        session.commit()
        session.refresh(dataset)
        self._adv_cache.clear()
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

    def get_dataset(self, session: Session, dataset_id: int) -> Dataset | None:
        return session.get(Dataset, dataset_id)

    def get_dataset_symbols(
        self,
        session: Session,
        dataset_id: int,
        *,
        timeframe: str | None = None,
    ) -> list[str]:
        dataset = self.get_dataset(session, dataset_id)
        if dataset is None:
            return []

        target_timeframe = timeframe or dataset.timeframe
        rows = session.exec(
            select(Dataset.symbol)
            .where(Dataset.provider == dataset.provider)
            .where(Dataset.timeframe == target_timeframe)
            .order_by(Dataset.symbol.asc())
        ).all()
        symbols = [str(row).upper() for row in rows if str(row).strip()]

        source_symbols = dataset.symbols_json if isinstance(dataset.symbols_json, list) else None
        if source_symbols:
            symbols.extend([str(item).upper() for item in source_symbols if str(item).strip()])
        if symbols:
            return list(dict.fromkeys(sorted(symbols)))
        return [dataset.symbol.upper()]

    def _symbol_adv(self, frame: pd.DataFrame, lookback: int) -> float:
        if frame.empty:
            return 0.0
        adv = (frame["close"] * frame["volume"]).tail(lookback).mean()
        return float(np.nan_to_num(float(adv), nan=0.0))

    def sample_dataset_symbols(
        self,
        session: Session,
        *,
        dataset_id: int,
        timeframe: str,
        symbol_scope: str,
        max_symbols_scan: int,
        adv_lookback: int = 20,
        seed: int = 7,
    ) -> list[str]:
        symbols = self.get_dataset_symbols(session, dataset_id, timeframe=timeframe)
        if not symbols:
            return []

        scope = str(symbol_scope or "liquid").lower()
        limit = len(symbols) if max_symbols_scan <= 0 else min(len(symbols), max_symbols_scan)
        if scope == "all":
            ordered = sorted(symbols)
            if seed is not None:
                # Deterministic pseudo-shuffle to avoid always taking the same alphabetical prefix.
                ordered = sorted(
                    ordered,
                    key=lambda value: hashlib.sha1(
                        f"{int(seed)}::{value}".encode("utf-8")
                    ).hexdigest(),
                )
            return ordered[:limit]

        cache_key = (dataset_id, timeframe, adv_lookback)
        scored = self._adv_cache.get(cache_key)
        if scored is None:
            rows: list[tuple[str, float]] = []
            for symbol in symbols:
                frame = self.load_ohlcv(symbol=symbol, timeframe=timeframe)
                rows.append((symbol, self._symbol_adv(frame, adv_lookback)))
            scored = sorted(rows, key=lambda item: (-item[1], item[0]))
            self._adv_cache[cache_key] = scored
        return [symbol for symbol, _ in scored[:limit]]

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
