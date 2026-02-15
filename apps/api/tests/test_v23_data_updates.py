from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import DataUpdateFile, PaperState
from app.db.session import engine, init_db
from app.main import app
from app.services.data_store import DataStore
from app.services.data_updates import run_data_updates


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(*, rows: int, start: float, symbol: str, start_date: str = "2026-01-01") -> pd.DataFrame:
    idx = pd.date_range(start_date, periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "symbol": [symbol] * rows,
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 1_000_000),
        }
    )


def _write_inbox_csv(*, settings, bundle_name: str, timeframe: str, file_name: str, frame: pd.DataFrame) -> Path:
    inbox_dir = Path(settings.data_inbox_root) / bundle_name / timeframe
    inbox_dir.mkdir(parents=True, exist_ok=True)
    path = inbox_dir / file_name
    frame.to_csv(path, index=False)
    return path


def _reset_runtime_settings(session: Session, settings) -> None:
    state = session.get(PaperState, 1)
    if state is None:
        return
    state.settings_json = {
        **(state.settings_json or {}),
        "data_updates_inbox_enabled": True,
        "data_updates_max_files_per_run": 50,
        "operate_mode": "offline",
        "data_quality_stale_severity": "WARN",
        "data_quality_stale_severity_override": True,
    }
    session.add(state)
    session.commit()


def test_data_updates_are_idempotent_for_same_file_hash() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"UPD_{unique}"
    bundle_name = f"bundle-upd-{unique}"

    base = _frame(rows=5, start=100.0, symbol=symbol, start_date="2026-01-01")
    update_rows = _frame(rows=1, start=105.0, symbol=symbol, start_date="2026-01-06")

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=base[["datetime", "open", "high", "low", "close", "volume"]],
            provider=f"test-updates-{unique}",
            bundle_name=bundle_name,
        )
        assert dataset.bundle_id is not None
        _reset_runtime_settings(session, settings)
        _write_inbox_csv(
            settings=settings,
            bundle_name=bundle_name,
            timeframe="1d",
            file_name=f"{unique}_update.csv",
            frame=update_rows,
        )
        state = session.get(PaperState, 1)
        overrides = dict(state.settings_json or {}) if state is not None else {}

        first = run_data_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(dataset.bundle_id),
            timeframe="1d",
            overrides=overrides,
        )
        assert first.status == "SUCCEEDED"
        assert first.rows_ingested >= 1

        second = run_data_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(dataset.bundle_id),
            timeframe="1d",
            overrides=overrides,
        )
        assert second.status == "SUCCEEDED"
        assert second.rows_ingested == 0

        skipped = session.exec(
            select(DataUpdateFile)
            .where(DataUpdateFile.run_id == int(second.id or 0))
            .order_by(DataUpdateFile.id.desc())
        ).first()
        assert skipped is not None
        assert skipped.status == "SKIPPED"
        assert skipped.reason == "duplicate_file_hash"


def test_data_updates_append_new_day_without_duplicates() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"APP_{unique}"
    bundle_name = f"bundle-app-{unique}"

    base = _frame(rows=4, start=120.0, symbol=symbol, start_date="2026-01-01")
    # One overlapping date + one new date.
    overlap = _frame(rows=2, start=123.0, symbol=symbol, start_date="2026-01-04")

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=base[["datetime", "open", "high", "low", "close", "volume"]],
            provider=f"test-append-{unique}",
            bundle_name=bundle_name,
        )
        assert dataset.bundle_id is not None
        _reset_runtime_settings(session, settings)
        _write_inbox_csv(
            settings=settings,
            bundle_name=bundle_name,
            timeframe="1d",
            file_name=f"{unique}_append.csv",
            frame=overlap,
        )
        state = session.get(PaperState, 1)
        overrides = dict(state.settings_json or {}) if state is not None else {}

        run = run_data_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(dataset.bundle_id),
            timeframe="1d",
            overrides=overrides,
        )
        assert run.status == "SUCCEEDED"
        assert run.rows_ingested == 1

        merged = store.load_ohlcv(symbol=symbol, timeframe="1d")
        merged["datetime"] = pd.to_datetime(merged["datetime"], utc=True)
        assert len(merged) == 5
        assert int(merged["datetime"].duplicated().sum()) == 0


def test_data_coverage_endpoint_returns_missing_symbols() -> None:
    import os

    os.environ["ATLAS_JOBS_INLINE"] = "true"
    get_settings.cache_clear()
    init_db()
    settings = get_settings()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"COV_{unique}"
    bundle_name = f"bundle-cov-{unique}"

    stale = _frame(rows=6, start=90.0, symbol=symbol, start_date="2024-01-01")

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=stale[["datetime", "open", "high", "low", "close", "volume"]],
            provider=f"test-coverage-{unique}",
            bundle_name=bundle_name,
        )
        assert dataset.bundle_id is not None
        _reset_runtime_settings(session, settings)
        target_bundle_id = int(dataset.bundle_id)

    with TestClient(app) as client:
        response = client.get(
            f"/api/data/coverage?bundle_id={target_bundle_id}&timeframe=1d&top_n=20"
        )
        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["bundle_id"] == target_bundle_id
        assert symbol in payload["missing_symbols"]
        assert payload["coverage_pct"] < 100.0
        assert isinstance(payload["last_bar_by_symbol"], list)
        assert len(payload["last_bar_by_symbol"]) >= 1
