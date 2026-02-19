from __future__ import annotations

from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import DatasetBundle, PaperState, ProviderCredential
from app.db.session import engine, init_db
from app.providers.base import BaseProvider
from app.services.data_store import DataStore
from app.services.operate_scheduler import run_auto_operate_once
from app.services.paper import get_or_create_paper_state
from app.services.provider_updates import run_provider_updates


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _base_frame(*, rows: int = 3, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1_000_000),
        }
    )


class _StaticProvider(BaseProvider):
    kind = "TEST"

    def __init__(self, symbols: list[str], frame: pd.DataFrame) -> None:
        super().__init__()
        self._symbols = [str(item).upper() for item in symbols]
        self._frame = frame.copy()
        self.seen_symbols: list[str] = []

    def list_symbols(self, bundle_id: int) -> list[str]:
        return list(self._symbols)

    def fetch_ohlc(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        output: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            symbol_up = str(symbol).upper()
            self.seen_symbols.append(symbol_up)
            self.count_api_call(1)
            output[symbol_up] = self._frame.copy()
        return output

    def supports_timeframes(self) -> set[str]:
        return {"1d"}


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def enqueue(self, task_path: str, *args: object, **kwargs: object) -> object:
        self.calls.append((task_path, args, kwargs))
        return {"task_path": task_path}


def test_provider_updates_idempotent_with_mock_provider() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"PRV_{unique}"

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_base_frame(rows=3, start=100.0),
            provider=f"provider-test-{unique}",
            bundle_name=f"bundle-provider-{unique}",
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)

        incoming = _base_frame(rows=4, start=100.0)
        provider = _StaticProvider([symbol], incoming)
        overrides = {
            "data_updates_provider_enabled": True,
            "data_updates_provider_kind": "TEST",
            "data_updates_provider_max_symbols_per_run": 10,
            "data_updates_provider_max_calls_per_run": 20,
            "data_updates_provider_timeframe_enabled": "1d",
        }

        first = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            overrides=overrides,
            provider=provider,
        )
        assert first.status == "SUCCEEDED"
        assert first.bars_added == 1

        second = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            overrides=overrides,
            provider=provider,
        )
        assert second.status == "SUCCEEDED"
        assert second.bars_added == 0

        merged = store.load_ohlcv(symbol=symbol, timeframe="1d")
        merged["datetime"] = pd.to_datetime(merged["datetime"], utc=True)
        assert int(merged["datetime"].duplicated().sum()) == 0


def test_scheduler_queues_provider_updates_before_inbox_updates() -> None:
    init_db()
    settings = get_settings()

    with Session(engine) as session:
        state = get_or_create_paper_state(session, settings)
        bundle = DatasetBundle(
            name=f"bundle-v27-{uuid4().hex[:8]}",
            provider="test",
            symbols_json=["NIFTY500"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_run_enabled": True,
            "operate_auto_run_time_ist": "09:00",
            "operate_last_auto_run_date": None,
            "operate_auto_run_include_data_updates": True,
            "data_updates_provider_enabled": True,
            "data_updates_provider_timeframe_enabled": "1d",
            "active_policy_id": None,
        }
        session.add(state)
        session.commit()

        queue = _FakeQueue()
        now_ist = datetime(2026, 2, 13, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
        triggered = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=now_ist,
        )
        assert triggered is True
        assert [call[0] for call in queue.calls] == [
            "app.jobs.tasks.run_provider_updates_job",
            "app.jobs.tasks.run_data_updates_job",
            "app.jobs.tasks.run_data_quality_job",
            "app.jobs.tasks.run_paper_step_job",
            "app.jobs.tasks.run_daily_report_job",
        ]

        refreshed = session.get(PaperState, 1)
        assert refreshed is not None
        assert (
            str((refreshed.settings_json or {}).get("operate_last_auto_run_date")) == "2026-02-13"
        )


def test_fast_mode_caps_provider_symbol_scan() -> None:
    init_db()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbols = [f"FAST_{unique}_{idx:02d}" for idx in range(30)]
    settings = Settings(
        fast_mode=True,
        fast_mode_max_symbols_scan=10,
        data_updates_provider_enabled=True,
        data_updates_provider_kind="MOCK",
        data_updates_provider_max_symbols_per_run=100,
        data_updates_provider_max_calls_per_run=500,
    )

    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"bundle-fast-provider-{unique}",
            provider="test",
            symbols_json=symbols,
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        frame = _base_frame(rows=2, start=200.0)
        provider = _StaticProvider(symbols, frame)
        row = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_kind": "MOCK",
                "data_updates_provider_max_symbols_per_run": 100,
                "data_updates_provider_max_calls_per_run": 500,
                "data_updates_provider_timeframe_enabled": "1d",
            },
            provider=provider,
        )
        assert row.status == "SUCCEEDED"
        assert row.symbols_attempted == 10
        assert len(provider.seen_symbols) == 10


def test_provider_updates_fail_when_upstox_token_missing() -> None:
    init_db()
    settings = Settings(
        redis_url="redis://127.0.0.1:1/0",
        data_updates_provider_enabled=True,
        data_updates_provider_kind="UPSTOX",
        upstox_access_token="",
    )
    store = _store()
    unique = uuid4().hex[:8].upper()

    with Session(engine) as session:
        for row in session.exec(select(ProviderCredential)).all():
            session.delete(row)
        session.commit()

        bundle = DatasetBundle(
            name=f"bundle-upstox-token-{unique}",
            provider="upstox",
            symbols_json=[f"UPX_{unique}"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        row = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_kind": "UPSTOX",
                "data_updates_provider_timeframe_enabled": "1d",
            },
        )
        assert row.status == "FAILED"
        assert any(
            str(item.get("code", "")).strip() == "provider_token_missing"
            for item in (row.errors_json or [])
            if isinstance(item, dict)
        )
