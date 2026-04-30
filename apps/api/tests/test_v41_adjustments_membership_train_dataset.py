from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from pathlib import Path
import time
from uuid import uuid4

import pandas as pd
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import DataProvenance, DatasetBundle, TrainDataset, TrainDatasetRun
from app.db.session import engine, init_db
from app.main import app
from app.services.corporate_actions import import_corporate_actions
from app.services.data_store import DataStore
from app.services.train_dataset import build_train_dataset, create_train_dataset
from app.services.universe_history import import_bundle_membership_history


CORP_FIXTURE = Path("apps/api/tests/fixtures/corporate_actions.csv")
MEMBERSHIP_FIXTURE = Path("apps/api/tests/fixtures/bundle_membership_history.csv")
ALPHA_FIXTURE = Path("apps/api/tests/fixtures/adjustment_sample_alpha.csv")


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        parquet_root=str(tmp_path / "parquet"),
        duckdb_path=str(tmp_path / "ohlcv.duckdb"),
        feature_cache_root=str(tmp_path / "features"),
        train_datasets_root=str(tmp_path / "train_datasets"),
        jobs_inline=True,
        fast_mode=True,
        e2e_fast=True,
    )


def _store(settings: Settings) -> DataStore:
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )


def _alpha_frame() -> pd.DataFrame:
    frame = pd.read_csv(ALPHA_FIXTURE)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    return frame


def _beta_frame() -> pd.DataFrame:
    rows = [
        ("2026-01-01T10:00:00+00:00", 200.0, 204.0, 198.0, 202.0, 900),
        ("2026-01-02T10:00:00+00:00", 202.0, 206.0, 200.0, 204.0, 950),
        ("2026-01-03T10:00:00+00:00", 204.0, 208.0, 202.0, 206.0, 980),
        ("2026-01-04T10:00:00+00:00", 206.0, 210.0, 204.0, 208.0, 990),
        ("2026-01-05T10:00:00+00:00", 208.0, 212.0, 206.0, 210.0, 1005),
        ("2026-01-06T10:00:00+00:00", 210.0, 214.0, 208.0, 212.0, 1015),
    ]
    return pd.DataFrame(
        rows,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    ).assign(datetime=lambda df: pd.to_datetime(df["datetime"], utc=True))


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


def _seed_bundle_data(session: Session, store: DataStore, bundle_id: int) -> None:
    store.save_ohlcv(
        session=session,
        symbol="ALPHA",
        timeframe="1d",
        frame=_alpha_frame(),
        provider="fixture",
        bundle_id=bundle_id,
        instrument_kind="EQUITY_CASH",
    )
    store.save_ohlcv(
        session=session,
        symbol="BETA",
        timeframe="1d",
        frame=_beta_frame(),
        provider="fixture",
        bundle_id=bundle_id,
        instrument_kind="EQUITY_CASH",
    )
    for symbol in ("ALPHA", "BETA"):
        frame = store.load_ohlcv(symbol=symbol, timeframe="1d")
        days = pd.to_datetime(frame["datetime"], utc=True).dt.tz_convert("Asia/Kolkata").dt.date
        for day in days.tolist():
            session.add(
                DataProvenance(
                    bundle_id=bundle_id,
                    timeframe="1d",
                    symbol=symbol,
                    bar_date=day,
                    source_provider="INBOX",
                    source_run_kind="test",
                    source_run_id=f"fixture-{symbol.lower()}",
                    confidence_score=72.0 if symbol == "ALPHA" else 74.0,
                )
            )
    session.commit()


def test_corporate_action_import_and_adjusted_read(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    store = _store(settings)
    unique = uuid4().hex[:8].upper()
    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v41-adjust-{unique}",
            provider="test",
            symbols_json=["ALPHA"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None
        store.save_ohlcv(
            session=session,
            symbol="ALPHA",
            timeframe="1d",
            frame=_alpha_frame(),
            provider="fixture",
            bundle_id=int(bundle.id),
        )
        summary = import_corporate_actions(session, path=str(CORP_FIXTURE), mode="REPLACE")
        assert summary["inserted_count"] >= 1

        adjusted = store.load_ohlcv(
            symbol="ALPHA",
            timeframe="1d",
            session=session,
            adjustment_mode="ADJUSTED",
        )
        raw = store.load_ohlcv(
            symbol="ALPHA",
            timeframe="1d",
            session=session,
            adjustment_mode="RAW",
        )
        assert float(raw.iloc[0]["close"]) == 100.0
        assert round(float(adjusted.iloc[0]["close"]), 2) == 50.0
        assert round(float(adjusted.iloc[1]["close"]), 2) == 52.0


def test_bundle_history_active_symbol_selection_by_date(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    store = _store(settings)
    unique = uuid4().hex[:8].upper()
    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v41-membership-{unique}",
            provider="test",
            symbols_json=["ALPHA", "BETA"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None
        import_bundle_membership_history(
            session,
            bundle_id=int(bundle.id),
            path=str(MEMBERSHIP_FIXTURE),
            mode="REPLACE",
        )
        early = store.get_bundle_symbols(
            session,
            int(bundle.id),
            timeframe="1d",
            asof_date=dt_date(2026, 1, 2),
            membership_mode="HISTORICAL",
        )
        later = store.get_bundle_symbols(
            session,
            int(bundle.id),
            timeframe="1d",
            asof_date=dt_date(2026, 1, 4),
            membership_mode="HISTORICAL",
        )
        assert early == ["ALPHA"]
        assert later == ["ALPHA", "BETA"]


def test_train_dataset_build_uses_adjusted_prices_and_historical_membership(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    store = _store(settings)
    unique = uuid4().hex[:8].upper()
    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v41-train-{unique}",
            provider="test",
            symbols_json=["ALPHA", "BETA"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        _seed_bundle_data(session, store, int(bundle.id))
        import_corporate_actions(session, path=str(CORP_FIXTURE), mode="REPLACE")
        import_bundle_membership_history(
            session,
            bundle_id=int(bundle.id),
            path=str(MEMBERSHIP_FIXTURE),
            mode="REPLACE",
        )

        dataset = create_train_dataset(
            session,
            payload={
                "name": f"dataset-{unique}",
                "bundle_id": int(bundle.id),
                "timeframe": "1d",
                "start_date": "2026-01-01",
                "end_date": "2026-01-06",
                "adjustment_mode": "ADJUSTED",
                "membership_mode": "HISTORICAL",
                "feature_config_json": {
                    "return_1d": True,
                    "return_5d": True,
                    "atr_14": True,
                    "rsi_14": True,
                    "ema_20": True,
                },
                "label_config_json": {"future_return_5d": True},
            },
        )
        run = build_train_dataset(
            session,
            settings=settings,
            store=store,
            dataset_id=int(dataset.id or 0),
            force=True,
        )
        assert run.status == "SUCCEEDED"
        assert run.output_path is not None
        assert Path(run.output_path).exists()

        built = pd.read_parquet(run.output_path)
        assert {
            "symbol",
            "trading_date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_provider_dominant",
            "confidence_score_day",
            "return_1d",
            "return_5d",
            "atr_14",
            "rsi_14",
            "ema_20",
            "future_return_5d",
        }.issubset(set(built.columns))
        alpha_first = built[built["symbol"] == "ALPHA"].sort_values("trading_date").iloc[0]
        assert round(float(alpha_first["close"]), 2) == 50.0
        beta_days = sorted(pd.to_datetime(built[built["symbol"] == "BETA"]["trading_date"]).dt.date.unique())
        assert beta_days[0] == dt_date(2026, 1, 3)
        assert built["source_provider_dominant"].notna().any()
        warning_codes = {str(item.get("code")) for item in run.warnings_json}
        assert "corporate_actions_missing_for_symbol" in warning_codes
        assert "confidence_source_summary" in warning_codes


def test_train_dataset_build_warns_when_membership_history_missing(tmp_path: Path) -> None:
    init_db()
    settings = get_settings()
    store = _store(settings)
    unique = uuid4().hex[:8].upper()
    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v41-nohist-{unique}",
            provider="test",
            symbols_json=["ALPHA"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None
        store.save_ohlcv(
            session=session,
            symbol="ALPHA",
            timeframe="1d",
            frame=_alpha_frame(),
            provider="fixture",
            bundle_id=int(bundle.id),
        )
        dataset = create_train_dataset(
            session,
            payload={
                "name": f"dataset-nohist-{unique}",
                "bundle_id": int(bundle.id),
                "timeframe": "1d",
                "start_date": "2026-01-01",
                "end_date": "2026-01-05",
                "adjustment_mode": "RAW",
                "membership_mode": "HISTORICAL",
                "feature_config_json": {},
                "label_config_json": {},
            },
        )
        run = build_train_dataset(
            session,
            settings=settings,
            store=store,
            dataset_id=int(dataset.id or 0),
            force=True,
        )
        warning_codes = {str(item.get("code")) for item in run.warnings_json}
        assert "membership_history_missing" in warning_codes


def test_train_dataset_api_build_and_download_info(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    store = _store(settings)
    unique = uuid4().hex[:8].upper()
    with Session(engine) as session:
        bundle = DatasetBundle(
            name=f"v41-api-{unique}",
            provider="test",
            symbols_json=["ALPHA", "BETA"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None
        _seed_bundle_data(session, store, int(bundle.id))
        import_corporate_actions(session, path=str(CORP_FIXTURE), mode="REPLACE")
        import_bundle_membership_history(
            session,
            bundle_id=int(bundle.id),
            path=str(MEMBERSHIP_FIXTURE),
            mode="REPLACE",
        )
        bundle_id = int(bundle.id)

    with TestClient(app) as client:
        create_response = client.post(
            "/api/train-datasets",
            json={
                "name": f"api-dataset-{unique}",
                "bundle_id": bundle_id,
                "timeframe": "1d",
                "start_date": "2026-01-01",
                "end_date": "2026-01-06",
                "adjustment_mode": "ADJUSTED",
                "membership_mode": "HISTORICAL",
                "feature_config_json": {"return_1d": True, "ema_20": True},
                "label_config_json": {"future_return_5d": True},
            },
        )
        assert create_response.status_code == 200
        dataset_payload = create_response.json().get("data", {})
        dataset_id = int(dataset_payload.get("id") or 0)
        assert dataset_id > 0

        build_response = client.post(
            f"/api/train-datasets/{dataset_id}/build",
            json={"force": True},
        )
        assert build_response.status_code == 200
        job_id = str(build_response.json().get("data", {}).get("job_id"))
        assert job_id
        _wait_for_job(client, job_id)

        latest_run = client.get(f"/api/train-datasets/{dataset_id}/latest-run")
        assert latest_run.status_code == 200
        latest_run_payload = latest_run.json().get("data", {})
        assert str(latest_run_payload.get("status")) == "SUCCEEDED"
        assert int(latest_run_payload.get("row_count") or 0) > 0

        download_info = client.get(f"/api/train-datasets/{dataset_id}/download-info")
        assert download_info.status_code == 200
        info_payload = download_info.json().get("data", {})
        assert bool(info_payload.get("file_exists")) is True
        assert int(info_payload.get("file_size_bytes") or 0) > 0

        with Session(engine) as session:
            dataset = session.get(TrainDataset, dataset_id)
            assert dataset is not None
            assert dataset.status == "READY"
            run = session.exec(
                select(TrainDatasetRun)
                .where(TrainDatasetRun.dataset_id == dataset_id)
                .order_by(TrainDatasetRun.id.desc())
            ).first()
            assert run is not None
            assert run.status == "SUCCEEDED"
