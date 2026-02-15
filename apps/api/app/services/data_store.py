from __future__ import annotations

from datetime import UTC, datetime
import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.exceptions import APIError
from app.db.models import Dataset, DatasetBundle, Instrument, Symbol
from app.engine.indicators import adx, atr, bollinger_bands, ema, keltner_channels, rsi


class DataStore:
    """Persists OHLCV/feature Parquet and serves analytical reads through DuckDB."""

    def __init__(
        self,
        parquet_root: str,
        duckdb_path: str,
        feature_cache_root: str | None = None,
    ) -> None:
        self.parquet_root = Path(parquet_root)
        self.parquet_root.mkdir(parents=True, exist_ok=True)
        self.feature_cache_root = Path(
            feature_cache_root
            if feature_cache_root is not None
            else str(self.parquet_root.parent / "features")
        )
        self.feature_cache_root.mkdir(parents=True, exist_ok=True)
        self.duckdb_path = duckdb_path
        self._adv_cache: dict[tuple[int, str, int], list[tuple[str, float]]] = {}
        self._bundle_adv_cache: dict[tuple[int, str, int], list[tuple[str, float]]] = {}

    def _connect_duckdb(self) -> duckdb.DuckDBPyConnection:
        # Windows can keep file handles locked when another local process has opened the DB;
        # fallback to in-memory because queries read directly from Parquet paths.
        if self.duckdb_path:
            try:
                return duckdb.connect(self.duckdb_path, read_only=True)
            except duckdb.IOException:
                pass
        return duckdb.connect(database=":memory:")

    def _parquet_path(self, symbol: str, timeframe: str) -> Path:
        return self.parquet_root / f"symbol={symbol}" / f"timeframe={timeframe}" / "ohlcv.parquet"

    def _feature_path(self, symbol: str, timeframe: str) -> Path:
        return (
            self.feature_cache_root
            / f"symbol={symbol}"
            / f"timeframe={timeframe}"
            / "features.parquet"
        )

    def _normalize_symbols(self, symbols: list[str]) -> list[str]:
        values = [str(item).upper().strip() for item in symbols if str(item).strip()]
        return list(dict.fromkeys(sorted(values)))

    def _infer_underlying(self, symbol: str, instrument_kind: str) -> str:
        symbol_up = symbol.upper()
        if instrument_kind == "EQUITY_CASH":
            return symbol_up
        if symbol_up.endswith("_FUT"):
            return symbol_up.removesuffix("_FUT")
        return symbol_up

    def _get_or_create_bundle(
        self,
        session: Session,
        *,
        provider: str,
        bundle_id: int | None = None,
        bundle_name: str | None = None,
        bundle_description: str | None = None,
    ) -> DatasetBundle | None:
        if bundle_id is not None:
            bundle = session.get(DatasetBundle, bundle_id)
            if bundle is not None:
                return bundle

        if bundle_name:
            existing = session.exec(
                select(DatasetBundle).where(DatasetBundle.name == bundle_name)
            ).first()
            if existing is not None:
                return existing
            bundle = DatasetBundle(
                name=bundle_name,
                provider=provider,
                description=bundle_description,
                symbols_json=[],
                supported_timeframes_json=[],
            )
            session.add(bundle)
            session.flush()
            return bundle

        default_name = f"{provider}-universe"
        bundle = session.exec(
            select(DatasetBundle).where(DatasetBundle.name == default_name)
        ).first()
        if bundle is None:
            bundle = DatasetBundle(
                name=default_name,
                provider=provider,
                description="Compatibility default bundle for legacy imports.",
                symbols_json=[],
                supported_timeframes_json=[],
            )
            session.add(bundle)
            session.flush()
        return bundle

    def _update_bundle_membership(
        self,
        session: Session,
        *,
        bundle: DatasetBundle | None,
        symbol: str,
        timeframe: str,
    ) -> None:
        if bundle is None:
            return
        symbols = self._normalize_symbols(list(bundle.symbols_json or []))
        if symbol.upper() not in symbols:
            symbols.append(symbol.upper())
        timeframes = [
            str(item).strip()
            for item in (bundle.supported_timeframes_json or [])
            if str(item).strip()
        ]
        if timeframe not in timeframes:
            timeframes.append(timeframe)
        bundle.symbols_json = self._normalize_symbols(symbols)
        bundle.supported_timeframes_json = list(dict.fromkeys(timeframes))
        session.add(bundle)

    def save_ohlcv(
        self,
        session: Session,
        symbol: str,
        timeframe: str,
        frame: pd.DataFrame,
        provider: str,
        checksum: str | None = None,
        instrument_kind: str = "EQUITY_CASH",
        underlying: str | None = None,
        lot_size: int | None = None,
        tick_size: float = 0.05,
        *,
        bundle_id: int | None = None,
        bundle_name: str | None = None,
        bundle_description: str | None = None,
    ) -> Dataset:
        # Ensure newly added tables/columns exist for direct service usage in tests and scripts.
        from app.db.session import init_db

        init_db()
        symbol = symbol.upper()
        kind = str(instrument_kind or "EQUITY_CASH").upper()
        resolved_underlying = (
            str(underlying).upper().strip()
            if isinstance(underlying, str) and underlying.strip()
            else self._infer_underlying(symbol, kind)
        )
        resolved_lot_size = int(lot_size) if lot_size is not None else 1
        if kind in {"STOCK_FUT", "INDEX_FUT"} and resolved_lot_size <= 0:
            raise APIError(
                code="invalid_instrument_metadata",
                message="Futures datasets require a positive lot_size.",
            )
        if resolved_lot_size <= 0:
            resolved_lot_size = 1

        clean = frame.copy()
        clean["datetime"] = pd.to_datetime(clean["datetime"], utc=True)
        clean = clean.sort_values("datetime")

        path = self._parquet_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_parquet(path, index=False)

        if session.exec(select(Symbol).where(Symbol.symbol == symbol)).first() is None:
            session.add(Symbol(symbol=symbol, name=symbol))
        if (
            resolved_underlying != symbol
            and session.exec(select(Symbol).where(Symbol.symbol == resolved_underlying)).first() is None
        ):
            session.add(Symbol(symbol=resolved_underlying, name=resolved_underlying))

        instrument = session.exec(
            select(Instrument)
            .where(Instrument.symbol == symbol)
            .where(Instrument.kind == kind)
            .order_by(Instrument.id.desc())
        ).first()
        if instrument is None:
            instrument = Instrument(
                symbol=symbol,
                kind=kind,
                underlying=resolved_underlying,
                lot_size=resolved_lot_size,
                tick_size=float(tick_size),
            )
        else:
            instrument.underlying = resolved_underlying
            instrument.lot_size = resolved_lot_size
            instrument.tick_size = float(tick_size)
        session.add(instrument)

        bundle = self._get_or_create_bundle(
            session,
            provider=provider,
            bundle_id=bundle_id,
            bundle_name=bundle_name,
            bundle_description=bundle_description,
        )
        self._update_bundle_membership(
            session,
            bundle=bundle,
            symbol=symbol,
            timeframe=timeframe,
        )

        dataset = Dataset(
            bundle_id=bundle.id if bundle is not None else None,
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
        self._bundle_adv_cache.clear()
        return dataset

    def list_symbols(self, session: Session) -> list[str]:
        rows = session.exec(select(Symbol).order_by(Symbol.symbol)).all()
        return [row.symbol for row in rows]

    def list_bundles(self, session: Session) -> list[dict[str, object]]:
        rows = session.exec(select(DatasetBundle).order_by(DatasetBundle.created_at.desc())).all()
        return [
            {
                "id": row.id,
                "name": row.name,
                "provider": row.provider,
                "description": row.description,
                "symbols": list(row.symbols_json or []),
                "supported_timeframes": list(row.supported_timeframes_json or []),
                "created_at": row.created_at.isoformat(),
            }
            for row in rows
        ]

    def get_bundle(self, session: Session, bundle_id: int) -> DatasetBundle | None:
        return session.get(DatasetBundle, bundle_id)

    def data_status(self, session: Session) -> list[dict[str, object]]:
        raw_rows = session.exec(select(Dataset).order_by(Dataset.created_at.desc())).all()
        rows: list[Dataset] = []
        seen: set[tuple[int | None, str, str]] = set()
        for row in raw_rows:
            key = (row.bundle_id, row.symbol, row.timeframe)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        rows.sort(key=lambda item: (str(item.symbol), str(item.timeframe), str(item.provider)))
        bundle_ids = {row.bundle_id for row in rows if row.bundle_id is not None}
        bundle_lookup: dict[int, str] = {}
        if bundle_ids:
            bundle_rows = session.exec(
                select(DatasetBundle).where(DatasetBundle.id.in_(list(bundle_ids)))
            ).all()
            bundle_lookup = {int(row.id): row.name for row in bundle_rows if row.id is not None}
        symbols = {row.symbol for row in rows}
        instrument_lookup: dict[str, Instrument] = {}
        if symbols:
            instrument_rows = session.exec(
                select(Instrument).where(Instrument.symbol.in_(list(symbols))).order_by(Instrument.id.desc())
            ).all()
            for item in instrument_rows:
                if item.symbol not in instrument_lookup:
                    instrument_lookup[item.symbol] = item
        return [
            {
                "id": row.id,
                "bundle_id": row.bundle_id,
                "bundle_name": bundle_lookup.get(int(row.bundle_id))
                if row.bundle_id is not None
                else None,
                "provider": row.provider,
                "symbol": row.symbol,
                "instrument_kind": (
                    instrument_lookup[row.symbol].kind if row.symbol in instrument_lookup else "EQUITY_CASH"
                ),
                "underlying": (
                    instrument_lookup[row.symbol].underlying
                    if row.symbol in instrument_lookup
                    else row.symbol
                ),
                "lot_size": (
                    instrument_lookup[row.symbol].lot_size if row.symbol in instrument_lookup else 1
                ),
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

    def get_bundle_symbols(
        self,
        session: Session,
        bundle_id: int,
        *,
        timeframe: str | None = None,
    ) -> list[str]:
        bundle = self.get_bundle(session, bundle_id)
        if bundle is None:
            return []
        rows_stmt = select(Dataset.symbol).where(Dataset.bundle_id == bundle_id)
        if timeframe:
            rows_stmt = rows_stmt.where(Dataset.timeframe == timeframe)
        rows = session.exec(rows_stmt.order_by(Dataset.symbol.asc())).all()
        dataset_symbols = [str(item).upper() for item in rows if str(item).strip()]
        if dataset_symbols:
            return self._normalize_symbols(dataset_symbols)
        return self._normalize_symbols(list(bundle.symbols_json or []))

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
        if dataset.bundle_id is not None:
            scoped = self.get_bundle_symbols(session, dataset.bundle_id, timeframe=target_timeframe)
            if scoped:
                return scoped

        source_symbols = dataset.symbols_json if isinstance(dataset.symbols_json, list) else None
        if source_symbols:
            return self._normalize_symbols([str(item) for item in source_symbols])
        return [dataset.symbol.upper()]

    def _symbol_adv(self, frame: pd.DataFrame, lookback: int) -> float:
        if frame.empty:
            return 0.0
        adv = (frame["close"] * frame["volume"]).tail(lookback).mean()
        return float(np.nan_to_num(float(adv), nan=0.0))

    def _rank_by_adv(
        self,
        symbols: list[str],
        *,
        timeframe: str,
        lookback: int,
    ) -> list[tuple[str, float]]:
        rows: list[tuple[str, float]] = []
        for symbol in symbols:
            frame = self.load_ohlcv(symbol=symbol, timeframe=timeframe)
            rows.append((symbol, self._symbol_adv(frame, lookback)))
        return sorted(rows, key=lambda item: (-item[1], item[0]))

    def sample_bundle_symbols(
        self,
        session: Session,
        *,
        bundle_id: int,
        timeframe: str,
        symbol_scope: str,
        max_symbols_scan: int,
        adv_lookback: int = 20,
        seed: int = 7,
    ) -> list[str]:
        symbols = self.get_bundle_symbols(session, bundle_id, timeframe=timeframe)
        if not symbols:
            return []
        scope = str(symbol_scope or "liquid").lower()
        limit = len(symbols) if max_symbols_scan <= 0 else min(len(symbols), max_symbols_scan)
        if scope == "all":
            ordered = sorted(symbols)
            ordered = sorted(
                ordered,
                key=lambda value: hashlib.sha1(f"{int(seed)}::{value}".encode("utf-8")).hexdigest(),
            )
            return ordered[:limit]

        cache_key = (bundle_id, timeframe, adv_lookback)
        scored = self._bundle_adv_cache.get(cache_key)
        if scored is None:
            scored = self._rank_by_adv(symbols, timeframe=timeframe, lookback=adv_lookback)
            self._bundle_adv_cache[cache_key] = scored
        return [symbol for symbol, _ in scored[:limit]]

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
        dataset = self.get_dataset(session, dataset_id)
        if dataset is None:
            return []
        if dataset.bundle_id is not None:
            return self.sample_bundle_symbols(
                session,
                bundle_id=int(dataset.bundle_id),
                timeframe=timeframe,
                symbol_scope=symbol_scope,
                max_symbols_scan=max_symbols_scan,
                adv_lookback=adv_lookback,
                seed=seed,
            )

        symbols = self.get_dataset_symbols(session, dataset_id, timeframe=timeframe)
        if not symbols:
            return []

        scope = str(symbol_scope or "liquid").lower()
        limit = len(symbols) if max_symbols_scan <= 0 else min(len(symbols), max_symbols_scan)
        if scope == "all":
            ordered = sorted(symbols)
            ordered = sorted(
                ordered,
                key=lambda value: hashlib.sha1(f"{int(seed)}::{value}".encode("utf-8")).hexdigest(),
            )
            return ordered[:limit]

        cache_key = (dataset_id, timeframe, adv_lookback)
        scored = self._adv_cache.get(cache_key)
        if scored is None:
            scored = self._rank_by_adv(symbols, timeframe=timeframe, lookback=adv_lookback)
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

        with self._connect_duckdb() as conn:
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

    def _compute_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        base = frame.copy().sort_values("datetime")
        base["datetime"] = pd.to_datetime(base["datetime"], utc=True)
        close = base["close"]
        bb = bollinger_bands(close, period=20, std_mult=2.0)
        kc = keltner_channels(base, period=20, atr_mult=1.5)
        atr_14 = atr(base, period=14)
        out = pd.DataFrame(
            {
                "datetime": base["datetime"],
                "close": close,
                "volume": base["volume"],
                "atr_14": atr_14,
                "atr_pct": atr_14 / close.replace(0, np.nan),
                "rsi_4": rsi(close, period=4),
                "rsi_7": rsi(close, period=7),
                "ema_20": ema(close, period=20),
                "ema_50": ema(close, period=50),
                "ema_100": ema(close, period=100),
                "ema_200": ema(close, period=200),
                "bb_mid": bb["mid"],
                "bb_upper": bb["upper"],
                "bb_lower": bb["lower"],
                "bb_width": bb["width"],
                "kc_mid": kc["mid"],
                "kc_upper": kc["upper"],
                "kc_lower": kc["lower"],
                "adx_14": adx(base, period=14),
            }
        )
        return out.sort_values("datetime").reset_index(drop=True)

    def update_feature_cache(self, symbol: str, timeframe: str) -> pd.DataFrame:
        base = self.load_ohlcv(symbol=symbol, timeframe=timeframe)
        if base.empty:
            return pd.DataFrame()
        path = self._feature_path(symbol, timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            computed = self._compute_feature_frame(base)
            computed.to_parquet(path, index=False)
            return computed

        cached = pd.read_parquet(path)
        if cached.empty:
            computed = self._compute_feature_frame(base)
            computed.to_parquet(path, index=False)
            return computed

        cached["datetime"] = pd.to_datetime(cached["datetime"], utc=True)
        cached = cached.sort_values("datetime")
        base = base.sort_values("datetime").reset_index(drop=True)
        latest_cached = cached["datetime"].max()
        latest_base = base["datetime"].max()
        if latest_cached >= latest_base and len(cached) >= len(base):
            return cached.reset_index(drop=True)

        indices = base.index[base["datetime"] >= latest_cached]
        if len(indices) > 0:
            start_idx = max(0, int(indices[0]) - 300)
        else:
            start_idx = max(0, len(base) - 500)
        recomputed_tail = self._compute_feature_frame(base.iloc[start_idx:].reset_index(drop=True))
        overlap_start = recomputed_tail["datetime"].min()
        head = cached[cached["datetime"] < overlap_start]
        merged = (
            pd.concat([head, recomputed_tail], ignore_index=True)
            .drop_duplicates(subset=["datetime"], keep="last")
            .sort_values("datetime")
            .reset_index(drop=True)
        )
        merged.to_parquet(path, index=False)
        return merged

    def load_features(
        self,
        *,
        symbol: str,
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        self.update_feature_cache(symbol=symbol, timeframe=timeframe)
        path = self._feature_path(symbol, timeframe)
        if not path.exists():
            return pd.DataFrame()

        with self._connect_duckdb() as conn:
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
        if frame.empty:
            return frame
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        return frame

    def get_lot_size(
        self,
        session: Session,
        *,
        symbol: str,
        instrument_kind: str = "EQUITY_CASH",
    ) -> int:
        row = session.exec(
            select(Instrument)
            .where(Instrument.symbol == symbol.upper())
            .where(Instrument.kind == instrument_kind.upper())
            .order_by(Instrument.id.desc())
        ).first()
        if row is None:
            return 1
        return max(1, int(row.lot_size))

    def get_instrument(
        self,
        session: Session,
        instrument_id: int,
    ) -> Instrument | None:
        return session.get(Instrument, instrument_id)

    def find_instrument(
        self,
        session: Session,
        *,
        symbol: str,
        instrument_kind: str | None = None,
    ) -> Instrument | None:
        query = select(Instrument).where(Instrument.symbol == symbol.upper())
        if instrument_kind:
            query = query.where(Instrument.kind == instrument_kind.upper())
        return session.exec(query.order_by(Instrument.id.desc())).first()

    def find_futures_instrument_for_underlying(
        self,
        session: Session,
        *,
        underlying: str,
        bundle_id: int | None = None,
        timeframe: str | None = None,
    ) -> Instrument | None:
        query = (
            select(Instrument)
            .where(Instrument.underlying == underlying.upper())
            .where(Instrument.kind == "STOCK_FUT")
            .order_by(Instrument.symbol.asc())
        )
        rows = session.exec(query).all()
        if not rows:
            return None
        if bundle_id is None:
            return rows[0]
        bundle_symbols = set(self.get_bundle_symbols(session, bundle_id, timeframe=timeframe))
        for row in rows:
            if row.symbol in bundle_symbols:
                return row
        return None

    def get_ohlcv_for_instrument(
        self,
        session: Session,
        *,
        instrument_id: int,
        timeframe: str,
        asof: datetime | None = None,
        window: int | None = None,
    ) -> pd.DataFrame:
        instrument = self.get_instrument(session, instrument_id)
        if instrument is None:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        frame = self.load_ohlcv(symbol=instrument.symbol, timeframe=timeframe)
        if frame.empty:
            return frame
        if asof is not None:
            ts = pd.Timestamp(asof)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            frame = frame[frame["datetime"] <= ts]
        if window is not None and int(window) > 0:
            frame = frame.tail(int(window))
        return frame.reset_index(drop=True)

    def get_futures_chain(
        self,
        session: Session,
        *,
        underlying: str,
        instrument_kind: str = "STOCK_FUT",
    ) -> list[dict[str, object]]:
        """Scaffold for derivatives support (phase 2)."""
        rows = session.exec(
            select(Instrument)
            .where(Instrument.underlying == underlying.upper())
            .where(Instrument.kind == instrument_kind.upper())
            .order_by(Instrument.symbol.asc())
        ).all()
        return [
            {
                "symbol": row.symbol,
                "kind": row.kind,
                "underlying": row.underlying,
                "lot_size": row.lot_size,
                "tick_size": row.tick_size,
            }
            for row in rows
        ]
