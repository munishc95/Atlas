from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import DataProvenance, DatasetBundle, ProviderCredential
from app.db.session import engine, init_db
from app.main import app
from app.providers.base import BaseProvider
from app.services.data_provenance import upsert_provenance_rows
from app.services.data_quality import run_data_quality_report
from app.services.data_store import DataStore
from app.services.provider_updates import run_provider_updates


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(start_day: str, days: int, start_price: float) -> pd.DataFrame:
    idx = pd.date_range(start_day, periods=days, freq="D", tz="UTC")
    close = np.linspace(start_price, start_price + days - 1, days)
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
    kind = "STATIC"

    def __init__(self, *, kind: str, symbols: list[str], frame: pd.DataFrame) -> None:
        super().__init__()
        self.kind = str(kind).upper()
        self._symbols = [str(item).upper() for item in symbols]
        self._frame = frame.copy()

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
            self.count_api_call(1)
            frame = self._frame.copy()
            if start is not None:
                frame = frame[frame["datetime"] >= pd.Timestamp(start).tz_convert("UTC")]
            if end is not None:
                frame = frame[frame["datetime"] <= pd.Timestamp(end).tz_convert("UTC")]
            output[symbol_up] = frame.reset_index(drop=True)
        return output

    def supports_timeframes(self) -> set[str]:
        return {"1d"}


def test_fallback_uses_nse_eod_and_persists_provenance() -> None:
    init_db()
    store = _store()
    settings = get_settings()
    unique = uuid4().hex[:8].upper()
    symbol = f"FB_{unique}"

    with Session(engine) as session:
        for row in session.exec(select(ProviderCredential)).all():
            session.delete(row)
        session.commit()

        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_frame("2026-02-02", 1, 100.0),
            provider=f"provider-v36-{unique}",
            bundle_name=f"bundle-v36-{unique}",
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)

        upstox_provider = _StaticProvider(
            kind="UPSTOX",
            symbols=[symbol],
            frame=_frame("2026-02-02", 5, 100.0),
        )
        nse_provider = _StaticProvider(
            kind="NSE_EOD",
            symbols=[symbol],
            frame=_frame("2026-02-02", 5, 100.0),
        )

        run = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            overrides={
                "data_updates_provider_enabled": True,
                "data_updates_provider_kind": "UPSTOX",
                "data_updates_provider_mode": "FALLBACK",
                "data_updates_provider_priority_order": ["UPSTOX", "NSE_EOD"],
                "data_updates_provider_timeframes": ["1d"],
                "data_updates_provider_nse_eod_enabled": True,
                "data_updates_provider_max_symbols_per_run": 10,
                "data_updates_provider_max_calls_per_run": 50,
            },
            start=datetime(2026, 2, 2, tzinfo=UTC),
            end=datetime(2026, 2, 7, tzinfo=UTC),
            provider_registry={
                "UPSTOX": upstox_provider,
                "NSE_EOD": nse_provider,
            },
        )
        assert run.status == "SUCCEEDED"
        assert int((run.by_provider_count_json or {}).get("NSE_EOD", 0)) > 0
        assert any(
            str(item.get("code", "")) == "primary_token_invalid"
            for item in (run.warnings_json or [])
            if isinstance(item, dict)
        )

        provenance_rows = session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == bundle_id)
            .where(DataProvenance.timeframe == "1d")
            .where(DataProvenance.symbol == symbol)
            .order_by(DataProvenance.bar_date.asc())
        ).all()
        assert len(provenance_rows) >= 1
        assert all(row.source_provider == "NSE_EOD" for row in provenance_rows)
        assert any((row.reason or "") == "primary_token_invalid" for row in provenance_rows)


def test_data_quality_low_confidence_live_mode_fail() -> None:
    init_db()
    store = _store()
    settings = get_settings()
    unique = uuid4().hex[:8].upper()
    symbol = f"LQC_{unique}"

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_frame("2026-02-10", 4, 120.0),
            provider=f"provider-v36-quality-{unique}",
            bundle_name=f"bundle-v36-quality-{unique}",
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)

        upsert_provenance_rows(
            session,
            bundle_id=bundle_id,
            timeframe="1d",
            symbol=symbol,
            bar_dates=[
                pd.Timestamp(value).tz_convert("Asia/Kolkata").date()
                for value in _frame("2026-02-10", 4, 120.0)["datetime"].tolist()
            ],
            source_provider="INBOX",
            source_run_kind="data_updates",
            source_run_id="v36-test",
            confidence_score=50.0,
            reason="manual_inbox_import",
            metadata={"test": True},
        )
        session.commit()

        report = run_data_quality_report(
            session=session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            overrides={
                "operate_mode": "live",
                "data_quality_stale_severity": "WARN",
                "data_quality_stale_severity_override": True,
                "data_quality_confidence_fail_threshold": 65,
            },
            reference_ts=datetime(2026, 2, 14, 12, 0, tzinfo=UTC),
        )
        assert report.status == "FAIL"
        assert int(report.low_confidence_symbols_count or 0) >= 1
        assert float((report.coverage_by_source_json or {}).get("INBOX", 0.0)) > 0.0
        issue_codes = {
            str(item.get("code", ""))
            for item in (report.issues_json or [])
            if isinstance(item, dict)
        }
        assert "low_confidence_latest_day" in issue_codes


def test_providers_status_endpoint_returns_breakdown() -> None:
    init_db()
    with TestClient(app) as client:
        response = client.get("/api/providers/status")
        assert response.status_code == 200
        payload = response.json()["data"]
        providers = payload.get("providers", [])
        provider_names = {str(item.get("provider", "")) for item in providers if isinstance(item, dict)}
        assert {"UPSTOX", "NSE_EOD", "INBOX"}.issubset(provider_names)
        assert isinstance(payload.get("upstox_token_status"), dict)


def test_data_provenance_endpoint_returns_rows() -> None:
    init_db()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"PRVAPI_{unique}"
    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_frame("2026-02-10", 3, 150.0),
            provider=f"provider-v36-api-{unique}",
            bundle_name=f"bundle-v36-api-{unique}",
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)
        upsert_provenance_rows(
            session,
            bundle_id=bundle_id,
            timeframe="1d",
            symbol=symbol,
            bar_dates=[datetime(2026, 2, 11, tzinfo=UTC).date()],
            source_provider="NSE_EOD",
            source_run_kind="provider_updates",
            source_run_id="api-test-run",
            confidence_score=80.0,
            reason="primary_fetch_failed",
            metadata={"note": "test"},
        )
        session.commit()

    with TestClient(app) as client:
        response = client.get(
            f"/api/data/provenance?bundle_id={bundle_id}&timeframe=1d&symbol={symbol}&limit=20"
        )
        assert response.status_code == 200
        payload = response.json()["data"]
        entries = payload.get("entries", [])
        assert isinstance(entries, list)
        assert len(entries) >= 1
        first = entries[0]
        assert first["symbol"] == symbol
        assert first["source_provider"] == "NSE_EOD"
