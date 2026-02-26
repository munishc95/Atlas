from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from uuid import uuid4
import os

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import get_settings
from app.db.models import PaperRun
from app.db.session import engine, init_db
from app.jobs.tasks import _operate_run_result
from app.main import app
from app.services.confidence_agg import upsert_daily_confidence_agg
from app.services.data_provenance import upsert_provenance_rows
from app.services.data_store import DataStore
from app.services.paper import get_or_create_paper_state, run_paper_step
from app.services.reports import generate_daily_report


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(start_day: str, days: int, start_price: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start_day, periods=days, freq="D", tz="UTC")
    close = np.linspace(start_price, start_price + days - 1, days)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(days, 1_200_000),
        }
    )


def _seed_bundle_symbol(
    *,
    session: Session,
    store: DataStore,
    symbol: str,
    bundle_name: str,
    start_day: str = "2026-02-10",
    days: int = 4,
) -> tuple[int, list[dt_date], list[datetime]]:
    frame = _frame(start_day=start_day, days=days, start_price=100.0)
    dataset = store.save_ohlcv(
        session=session,
        symbol=symbol,
        timeframe="1d",
        frame=frame,
        provider="v39-tests",
        bundle_name=bundle_name,
    )
    assert dataset.bundle_id is not None
    bar_dates = [
        pd.Timestamp(item).tz_convert("Asia/Kolkata").date() for item in frame["datetime"].tolist()
    ]
    asofs = [pd.Timestamp(item).to_pydatetime() for item in frame["datetime"].tolist()]
    return int(dataset.bundle_id), bar_dates, asofs


def _upsert_day_provenance(
    session: Session,
    *,
    bundle_id: int,
    symbol: str,
    bar_day: dt_date,
    source_provider: str,
    confidence_score: float,
    source_run_id: str,
) -> None:
    upsert_provenance_rows(
        session,
        bundle_id=bundle_id,
        timeframe="1d",
        symbol=symbol,
        bar_dates=[bar_day],
        source_provider=source_provider,
        source_run_kind="provider_updates",
        source_run_id=source_run_id,
        confidence_score=confidence_score,
        reason="v39_test",
        metadata={"v": "3.9"},
    )


def _confidence_overrides(settings) -> dict[str, object]:
    return {
        "operate_mode": "live",
        "confidence_gate_enabled": True,
        "confidence_gate_avg_threshold": 70.0,
        "confidence_gate_low_symbol_threshold": 65.0,
        "confidence_gate_low_pct_threshold": 0.50,
        "confidence_gate_fallback_pct_threshold": 0.80,
        "confidence_gate_hard_floor": 55.0,
        "confidence_gate_action_on_trigger": "SHADOW_ONLY",
        "confidence_gate_lookback_days": 1,
        "confidence_drop_warn_threshold": 5.0,
        "confidence_provider_mix_shift_warn_pct": 0.20,
        "confidence_risk_scaling_enabled": True,
        "confidence_risk_scale_exponent": 1.0,
        "confidence_risk_scale_low_threshold": 0.35,
        "trading_calendar_segment": settings.trading_calendar_segment,
    }


def test_effective_context_in_status_paper_report_and_operate_result() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    token = uuid4().hex[:8].upper()
    bundle_name = f"v39-bundle-{token}"
    symbol = f"V39CTX{token[:3]}"

    report_id: int | None = None
    with Session(engine) as session:
        bundle_id, bar_dates, asofs = _seed_bundle_symbol(
            session=session,
            store=store,
            symbol=symbol,
            bundle_name=bundle_name,
        )
        _upsert_day_provenance(
            session,
            bundle_id=bundle_id,
            symbol=symbol,
            bar_day=bar_dates[-1],
            source_provider="UPSTOX",
            confidence_score=82.0,
            source_run_id=f"ctx-{token}",
        )
        session.commit()

        state = get_or_create_paper_state(session, settings)
        merged = dict(state.settings_json or {})
        merged.update(
            {
                "paper_use_simulator_engine": True,
                "paper_mode": "strategy",
                "active_policy_id": None,
                "data_quality_stale_severity": "WARN",
                "data_quality_stale_severity_override": True,
                "operate_mode": "live",
                "confidence_gate_enabled": True,
            }
        )
        state.settings_json = merged
        session.add(state)
        session.commit()

        paper_result = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "asof": asofs[-1].isoformat(),
                "signals": [
                    {
                        "symbol": symbol,
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 103.0,
                        "stop_distance": 2.0,
                        "signal_strength": 0.9,
                        "adv": 1_000_000_000.0,
                        "vol_scale": 0.01,
                    }
                ],
                "auto_generate_signals": False,
            },
            store=store,
        )
        effective_context = paper_result.get("effective_context")
        assert isinstance(effective_context, dict)
        assert str(effective_context.get("timeframe")) == "1d"
        assert str(effective_context.get("trading_date", "")).strip() != ""

        paper_run_id = int(paper_result.get("paper_run_id", 0))
        assert paper_run_id > 0
        run_row = session.get(PaperRun, paper_run_id)
        assert run_row is not None
        report = generate_daily_report(
            session=session,
            settings=settings,
            report_date=run_row.asof_ts.date(),
            bundle_id=bundle_id,
            policy_id=None,
            overwrite=True,
        )
        assert report.id is not None
        report_id = int(report.id)

        operate_result = _operate_run_result(
            session=session,
            settings=settings,
            store=store,
            payload={"bundle_id": bundle_id, "timeframe": "1d", "include_data_updates": False},
            job_id=f"v39-operate-{token}",
        )
        assert isinstance(operate_result.get("effective_context"), dict)
        assert str((operate_result.get("effective_context") or {}).get("timeframe")) == "1d"

    with TestClient(app) as client:
        status_res = client.get("/api/operate/status")
        assert status_res.status_code == 200
        status_data = status_res.json().get("data", {})
        assert isinstance(status_data.get("effective_context"), dict)

        assert report_id is not None
        report_res = client.get(f"/api/reports/daily/{report_id}")
        assert report_res.status_code == 200
        report_data = report_res.json().get("data", {})
        assert isinstance(report_data.get("effective_context"), dict)
        assert str((report_data.get("effective_context") or {}).get("trading_date", "")).strip() != ""


def test_timeline_uses_aggregates_with_drop_and_flags() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    token = uuid4().hex[:8].upper()
    bundle_name = f"v39-timeline-{token}"

    with Session(engine) as session:
        bundle_id, bar_dates, _ = _seed_bundle_symbol(
            session=session,
            store=store,
            symbol=f"V39TL{token[:3]}",
            bundle_name=bundle_name,
        )
        # Previous day: high confidence via UPSTOX.
        _upsert_day_provenance(
            session,
            bundle_id=bundle_id,
            symbol=f"V39TL{token[:3]}",
            bar_day=bar_dates[-2],
            source_provider="UPSTOX",
            confidence_score=88.0,
            source_run_id=f"tl-prev-{token}",
        )
        # Latest day: low confidence via INBOX to trigger drop + mix shift.
        _upsert_day_provenance(
            session,
            bundle_id=bundle_id,
            symbol=f"V39TL{token[:3]}",
            bar_day=bar_dates[-1],
            source_provider="INBOX",
            confidence_score=50.0,
            source_run_id=f"tl-latest-{token}",
        )
        session.commit()

        overrides = _confidence_overrides(settings)
        upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=bar_dates[-2],
            operate_mode="live",
            overrides=overrides,
            force=True,
        )
        upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=bar_dates[-1],
            operate_mode="live",
            overrides=overrides,
            force=True,
        )
        session.commit()

    with TestClient(app) as client:
        res = client.get(f"/api/confidence/timeline?bundle_id={bundle_id}&timeframe=1d&limit=10")
        assert res.status_code == 200
        rows = (res.json().get("data", {}) or {}).get("rows", [])
        assert isinstance(rows, list)
        assert len(rows) >= 2
        latest = rows[-1]
        assert float(latest.get("confidence_drop_vs_prev", 0.0)) < 0.0
        flags = [str(item) for item in (latest.get("flags") or [])]
        assert "CONF_DROP" in flags
        assert "MIX_SHIFT" in flags


def test_drilldown_structure_and_symbol_limit_cap() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    token = uuid4().hex[:8].upper()
    bundle_name = f"v39-drill-{token}"
    sym_a = f"V39DA{token[:2]}"
    sym_b = f"V39DB{token[:2]}"
    sym_c = f"V39DC{token[:2]}"

    with Session(engine) as session:
        bundle_id, bar_dates, _ = _seed_bundle_symbol(
            session=session,
            store=store,
            symbol=sym_a,
            bundle_name=bundle_name,
        )
        _seed_bundle_symbol(
            session=session,
            store=store,
            symbol=sym_b,
            bundle_name=bundle_name,
        )
        _seed_bundle_symbol(
            session=session,
            store=store,
            symbol=sym_c,
            bundle_name=bundle_name,
        )
        _upsert_day_provenance(
            session,
            bundle_id=bundle_id,
            symbol=sym_a,
            bar_day=bar_dates[-1],
            source_provider="NSE_EOD",
            confidence_score=54.0,
            source_run_id=f"drill-a-{token}",
        )
        _upsert_day_provenance(
            session,
            bundle_id=bundle_id,
            symbol=sym_b,
            bar_day=bar_dates[-1],
            source_provider="UPSTOX",
            confidence_score=83.0,
            source_run_id=f"drill-b-{token}",
        )
        session.commit()
        overrides = _confidence_overrides(settings)
        upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=bar_dates[-1],
            operate_mode="live",
            overrides=overrides,
            force=True,
        )
        session.commit()

    with TestClient(app) as client:
        day = bar_dates[-1].isoformat()
        res = client.get(
            f"/api/confidence/drilldown?bundle_id={bundle_id}&timeframe=1d&trading_date={day}"
        )
        assert res.status_code == 200
        payload = res.json().get("data", {})
        assert isinstance(payload.get("summary"), dict)
        assert isinstance(payload.get("worst_symbols_by_confidence"), list)
        assert isinstance(payload.get("provider_mix_delta"), dict)
        missing = payload.get("missing_symbols", [])
        assert isinstance(missing, list)
        assert any(str(item).upper() == sym_c for item in missing)

        rows_res = client.get(
            f"/api/confidence/drilldown/symbols?bundle_id={bundle_id}&timeframe=1d&trading_date={day}&only=all&limit=1"
        )
        assert rows_res.status_code == 200
        rows_payload = rows_res.json().get("data", {})
        rows = rows_payload.get("rows", [])
        assert int(rows_payload.get("limit", 0)) == 1
        assert isinstance(rows, list)
        assert len(rows) == 1


def test_fast_mode_timeline_and_drilldown_are_deterministic_without_data() -> None:
    previous = os.environ.get("ATLAS_FAST_MODE")
    os.environ["ATLAS_FAST_MODE"] = "1"
    get_settings.cache_clear()
    try:
        with TestClient(app) as client:
            timeline = client.get("/api/confidence/timeline?bundle_id=999999&timeframe=1d&limit=5")
            assert timeline.status_code == 200
            timeline_rows = (timeline.json().get("data", {}) or {}).get("rows", [])
            assert isinstance(timeline_rows, list)
            assert len(timeline_rows) == 5

            day = datetime.now(UTC).date().isoformat()
            drill = client.get(
                f"/api/confidence/drilldown?bundle_id=999999&timeframe=1d&trading_date={day}"
            )
            assert drill.status_code == 200
            data = drill.json().get("data", {})
            assert isinstance(data.get("summary"), dict)
            notes = [str(item) for item in (data.get("notes") or [])]
            assert "fast_mode_stub" in notes
    finally:
        if previous is None:
            os.environ.pop("ATLAS_FAST_MODE", None)
        else:
            os.environ["ATLAS_FAST_MODE"] = previous
        get_settings.cache_clear()
