from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from pathlib import Path
import time
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import DataProvenance, DatasetBundle, ProviderUpdateItem
from app.db.session import engine, init_db
from app.main import app
from app.providers.nse_bhavcopy_provider import NseBhavcopyProvider
from app.services.data_store import DataStore
from app.services.historical_backfill import run_historical_backfill
from app.services.provider_updates import run_provider_updates
from app.services.trading_calendar import list_trading_days


FIXTURE = Path("apps/api/tests/fixtures/bhavcopy_sample_26022026.csv")


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _wait_for_job(client: TestClient, job_id: str, timeout_seconds: float = 20.0) -> dict:
    started = time.monotonic()
    while time.monotonic() - started < timeout_seconds:
        response = client.get(f"/api/jobs/{job_id}")
        assert response.status_code == 200
        payload = response.json().get("data", {})
        status = str(payload.get("status", "")).upper()
        if status in {"SUCCEEDED", "DONE"}:
            return payload
        if status == "FAILED":
            raise AssertionError(f"job {job_id} failed: {payload.get('result_json')}")
        time.sleep(0.2)
    raise AssertionError(f"timed out waiting for job {job_id}")


def test_nse_bhavcopy_parsing_normalization_fixture() -> None:
    init_db()
    store = _store()
    settings = Settings(fast_mode=False, e2e_fast=False)
    with Session(engine) as session:
        provider = NseBhavcopyProvider(session=session, settings=settings, store=store)
        raw = FIXTURE.read_bytes()
        frame = provider._parse_day_csv(raw=raw, day=dt_date(2026, 2, 26))
        assert set(frame.columns) == {"symbol", "datetime", "open", "high", "low", "close", "volume"}
        assert sorted(frame["symbol"].tolist()) == ["HDFCBANK", "RELIANCE", "TCS"]
        assert frame["datetime"].dt.tz is not None
        # 15:30 IST normalized to UTC.
        assert all(value.hour == 10 and value.minute == 0 for value in frame["datetime"].dt.tz_convert("UTC"))
        assert float(frame["open"].min()) > 0.0
        assert float(frame["volume"].min()) > 0.0


def test_historical_backfill_date_planning_uses_trading_calendar() -> None:
    init_db()
    store = _store()
    settings = get_settings()
    unique = uuid4().hex[:8].upper()
    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v40-plan-{unique}",
            provider="test",
            symbols_json=[f"PLAN_{unique}"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        start_day = dt_date(2026, 2, 20)
        end_day = dt_date(2026, 2, 28)
        row = run_historical_backfill(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            timeframe="1d",
            provider_kind="NSE_BHAVCOPY",
            mode="SINGLE",
            start_date=start_day,
            end_date=end_day,
            dry_run=True,
        )
        expected_days = list_trading_days(
            start_date=start_day,
            end_date=end_day,
            segment=str(settings.trading_calendar_segment or "EQUITIES"),
            settings=settings,
        )
        assert row.status == "SUCCEEDED"
        assert row.dry_run is True
        assert row.trading_days_planned == len(expected_days)


def test_provider_updates_nse_bhavcopy_idempotent_and_provenance() -> None:
    init_db()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"BHV_{unique}"
    settings = Settings(
        fast_mode=True,
        data_updates_provider_enabled=True,
        data_updates_provider_kind="NSE_BHAVCOPY",
        data_updates_provider_timeframes=["1d"],
        data_updates_provider_max_symbols_per_run=10,
        data_updates_provider_max_calls_per_run=100,
        data_updates_provider_repair_last_n_trading_days=2,
        data_updates_provider_backfill_max_days=30,
        data_provenance_confidence_nse_bhavcopy=92,
    )

    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v40-bundle-{unique}",
            provider="test",
            symbols_json=[symbol],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        first = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_kind": "NSE_BHAVCOPY",
                "data_updates_provider_timeframes": ["1d"],
                "data_updates_provider_nse_bhavcopy_enabled": True,
            },
            provider_kind="NSE_BHAVCOPY",
            start=datetime(2026, 2, 20, tzinfo=UTC),
            end=datetime(2026, 2, 28, tzinfo=UTC),
        )
        assert first.status == "SUCCEEDED"
        assert first.bars_added > 0

        second = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_kind": "NSE_BHAVCOPY",
                "data_updates_provider_timeframes": ["1d"],
                "data_updates_provider_nse_bhavcopy_enabled": True,
            },
            provider_kind="NSE_BHAVCOPY",
            start=datetime(2026, 2, 20, tzinfo=UTC),
            end=datetime(2026, 2, 28, tzinfo=UTC),
        )
        assert second.status == "SUCCEEDED"
        assert second.bars_added == 0
        second_item = session.exec(
            select(ProviderUpdateItem)
            .where(ProviderUpdateItem.run_id == int(second.id or 0))
            .where(ProviderUpdateItem.symbol == symbol)
            .order_by(ProviderUpdateItem.id.desc())
        ).first()
        assert second_item is not None
        assert int(second_item.bars_updated or 0) == 0

        provenance = session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle.id))
            .where(DataProvenance.timeframe == "1d")
            .where(DataProvenance.symbol == symbol)
            .order_by(DataProvenance.bar_date.desc())
        ).all()
        assert len(provenance) >= 1
        assert all(row.source_provider == "NSE_BHAVCOPY" for row in provenance)
        assert all(abs(float(row.confidence_score) - 92.0) < 0.001 for row in provenance)


def test_historical_backfill_api_enqueues_and_persists() -> None:
    init_db()
    unique = uuid4().hex[:8].upper()
    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v40-api-{unique}",
            provider="test",
            symbols_json=[f"API_{unique}"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None
        bundle_id = int(bundle.id)

    with TestClient(app) as client:
        response = client.post(
            "/api/data/backfill/run",
            json={
                "bundle_id": bundle_id,
                "timeframe": "1d",
                "provider_kind": "MOCK",
                "start_date": "2026-02-20",
                "end_date": "2026-02-28",
                "mode": "SINGLE",
                "dry_run": False,
            },
        )
        assert response.status_code == 200
        payload = response.json().get("data", {})
        job_id = str(payload.get("job_id"))
        assert job_id
        _wait_for_job(client, job_id)

        latest = client.get(f"/api/data/backfill/latest?bundle_id={bundle_id}&timeframe=1d")
        assert latest.status_code == 200
        latest_payload = latest.json().get("data", {})
        assert int(latest_payload.get("bundle_id") or 0) == bundle_id
        assert str(latest_payload.get("provider_kind")) == "MOCK"
        assert int(latest_payload.get("trading_days_planned") or 0) > 0

        history = client.get(f"/api/data/backfill/history?bundle_id={bundle_id}&timeframe=1d&limit=5")
        assert history.status_code == 200
        history_rows = history.json().get("data", [])
        assert isinstance(history_rows, list)
        assert len(history_rows) >= 1
