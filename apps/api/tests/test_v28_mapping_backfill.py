from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import DatasetBundle, InstrumentMap, ProviderUpdateItem
from app.db.session import engine, init_db
from app.providers.base import BaseProvider
from app.providers.upstox_provider import UpstoxProvider
from app.services.data_store import DataStore
from app.services.provider_mapping import get_upstox_mapping_status, import_upstox_mapping_file
from app.services.provider_updates import run_provider_updates


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(days: int = 4, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-02-02", periods=days, freq="D", tz="UTC")
    close = np.linspace(start, start + days - 1, days)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(days, 1_000_000),
        }
    )


class _StaticProvider(BaseProvider):
    kind = "TEST"

    def __init__(self, symbols: list[str], frame: pd.DataFrame, missing: set[str] | None = None) -> None:
        super().__init__()
        self._symbols = [str(item).upper() for item in symbols]
        self._frame = frame.copy()
        self._missing = {str(item).upper() for item in (missing or set())}

    def list_symbols(self, bundle_id: int) -> list[str]:
        return list(self._symbols)

    def missing_mapped_symbols(self, symbols: list[str]) -> set[str]:
        return {str(item).upper() for item in symbols if str(item).upper() in self._missing}

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
            frame = self._frame.copy()
            if start is not None:
                frame = frame[frame["datetime"] >= pd.Timestamp(start).tz_convert("UTC")]
            if end is not None:
                frame = frame[frame["datetime"] <= pd.Timestamp(end).tz_convert("UTC")]
            output[str(symbol).upper()] = frame.reset_index(drop=True)
        return output

    def supports_timeframes(self) -> set[str]:
        return {"1d", "4h_ish"}


def test_mapping_import_and_status(tmp_path: Path) -> None:
    init_db()
    store = _store()
    settings = get_settings()
    unique = uuid4().hex[:8].upper()
    symbol_a = f"MAPA_{unique}"
    symbol_b = f"MAPB_{unique}"

    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"bundle-map-{unique}",
            provider="test",
            symbols_json=[symbol_a, symbol_b],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        mapping_file = tmp_path / "upstox_instruments.csv"
        mapping_file.write_text(
            "symbol,instrument_key\n"
            f"{symbol_a},NSE_EQ|{symbol_a}\n",
            encoding="utf-8",
        )

        row = import_upstox_mapping_file(
            session=session,
            settings=settings,
            store=store,
            path=str(mapping_file),
            mode="UPSERT",
            bundle_id=int(bundle.id),
        )
        assert row.status == "SUCCEEDED"
        assert row.inserted_count >= 1

        records = session.exec(
            select(InstrumentMap)
            .where(InstrumentMap.provider == "UPSTOX")
            .where(InstrumentMap.symbol == symbol_a)
        ).all()
        assert len(records) == 1
        assert records[0].instrument_key == f"NSE_EQ|{symbol_a}"

        status = get_upstox_mapping_status(
            session=session,
            store=store,
            bundle_id=int(bundle.id),
            timeframe="1d",
        )
        assert status["mapped_count"] == 1
        assert status["missing_count"] == 1
        assert symbol_b in status["sample_missing_symbols"]


def test_provider_updates_skip_missing_map_reason() -> None:
    init_db()
    store = _store()
    settings = get_settings()
    unique = uuid4().hex[:8].upper()
    symbols = [f"AAA_{unique}", f"BBB_{unique}"]

    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"bundle-provider-miss-{unique}",
            provider="test",
            symbols_json=symbols,
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        provider = _StaticProvider(symbols=symbols, frame=_frame(days=3, start=100.0), missing={symbols[1]})
        row = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_timeframes": ["1d"],
                "data_updates_provider_max_symbols_per_run": 10,
                "data_updates_provider_max_calls_per_run": 20,
            },
            provider=provider,
        )
        assert row.status == "SUCCEEDED"
        items = session.exec(
            select(ProviderUpdateItem).where(ProviderUpdateItem.run_id == int(row.id or 0))
        ).all()
        missing_item = [item for item in items if item.symbol == symbols[1]][0]
        assert missing_item.status == "SKIPPED"
        assert any(
            str(warning.get("code")) == "missing_instrument_map"
            for warning in (missing_item.warnings_json or [])
        )


def test_provider_backfill_summary_fields() -> None:
    init_db()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"REPAIR_{unique}"
    settings = Settings(
        data_updates_provider_enabled=True,
        data_updates_provider_kind="MOCK",
        data_updates_provider_timeframes=["1d"],
        data_updates_provider_repair_last_n_trading_days=3,
        data_updates_provider_backfill_max_days=5,
    )

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_frame(days=2, start=100.0),
            provider=f"provider-repair-{unique}",
            bundle_name=f"bundle-repair-{unique}",
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)

        corrected = _frame(days=5, start=100.0)
        corrected.loc[1, "close"] = 999.0
        provider = _StaticProvider([symbol], corrected)
        row = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_timeframes": ["1d"],
                "data_updates_provider_repair_last_n_trading_days": 3,
                "data_updates_provider_backfill_max_days": 2,
            },
            start=datetime(2026, 1, 2, tzinfo=UTC),
            end=datetime(2026, 2, 10, tzinfo=UTC),
            provider=provider,
        )
        assert row.repaired_days_used >= 1
        assert row.missing_days_detected >= 1
        assert row.backfill_truncated is True
        assert row.missing_days_detected > 2


def test_provider_repair_updates_existing_bar() -> None:
    init_db()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"REPAIRUP_{unique}"
    settings = Settings(
        data_updates_provider_enabled=True,
        data_updates_provider_kind="MOCK",
        data_updates_provider_timeframes=["1d"],
        data_updates_provider_repair_last_n_trading_days=3,
        data_updates_provider_backfill_max_days=20,
    )

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_frame(days=3, start=100.0),
            provider=f"provider-repairup-{unique}",
            bundle_name=f"bundle-repairup-{unique}",
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)

        corrected = _frame(days=4, start=100.0)
        corrected.loc[2, "close"] = 555.0
        provider = _StaticProvider([symbol], corrected)
        row = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_timeframes": ["1d"],
                "data_updates_provider_repair_last_n_trading_days": 3,
                "data_updates_provider_backfill_max_days": 20,
            },
            start=datetime(2026, 2, 1, tzinfo=UTC),
            end=datetime(2026, 2, 5, tzinfo=UTC),
            provider=provider,
        )
        assert row.status == "SUCCEEDED"
        item = session.exec(
            select(ProviderUpdateItem)
            .where(ProviderUpdateItem.run_id == int(row.id or 0))
            .where(ProviderUpdateItem.symbol == symbol)
            .order_by(ProviderUpdateItem.id.desc())
        ).first()
        assert item is not None
        assert item.bars_updated >= 1

        item = session.exec(
            select(ProviderUpdateItem)
            .where(ProviderUpdateItem.run_id == int(row.id or 0))
            .where(ProviderUpdateItem.symbol == symbol)
            .order_by(ProviderUpdateItem.id.desc())
        ).first()
        assert item is not None
        assert item.bars_updated >= 1


def test_upstox_4h_ish_resample_two_bars_per_day() -> None:
    init_db()
    store = _store()
    settings = get_settings()
    with Session(engine) as session:
        provider = UpstoxProvider(session=session, settings=settings, store=store)
        day = datetime(2026, 2, 13, 9, 15, tzinfo=ZoneInfo("Asia/Kolkata"))
        frame = pd.DataFrame(
            {
                "datetime": [
                    day.astimezone(UTC),
                    day.replace(hour=12, minute=0).astimezone(UTC),
                    day.replace(hour=13, minute=30).astimezone(UTC),
                    day.replace(hour=15, minute=20).astimezone(UTC),
                ],
                "open": [100.0, 101.0, 104.0, 105.0],
                "high": [102.0, 105.0, 106.0, 107.0],
                "low": [99.0, 100.0, 103.0, 104.0],
                "close": [101.0, 104.0, 105.0, 106.0],
                "volume": [10.0, 20.0, 30.0, 40.0],
            }
        )
        out = provider._resample_4h_ish(
            frame,
            start_ts=day.astimezone(UTC),
            end_ts=day.replace(hour=15, minute=30).astimezone(UTC),
            allow_partial=False,
        )
        assert len(out) == 2
        first = out.iloc[0]
        second = out.iloc[1]
        assert float(first["open"]) == 100.0
        assert float(first["high"]) == 105.0
        assert float(first["low"]) == 99.0
        assert float(first["close"]) == 104.0
        assert float(first["volume"]) == 30.0
        assert float(second["open"]) == 104.0
        assert float(second["high"]) == 107.0
        assert float(second["low"]) == 103.0
        assert float(second["close"]) == 106.0
        assert float(second["volume"]) == 70.0
