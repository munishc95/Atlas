from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import ConfidenceGateSnapshot, PaperRun, ShadowPaperState
from app.db.session import engine, init_db
from app.main import app
from app.services.confidence_gate import evaluate_confidence_gate
from app.services.data_provenance import upsert_provenance_rows
from app.services.data_store import DataStore
from app.services.paper import get_or_create_paper_state, get_paper_state_payload, run_paper_step
from app.services.reports import generate_daily_report


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


def _seed_bundle_with_low_confidence(
    *,
    session: Session,
    store: DataStore,
    symbol: str,
    start_day: str = "2026-02-10",
    days: int = 4,
    confidence: float = 50.0,
) -> tuple[int, list[datetime]]:
    frame = _frame(start_day, days, 100.0)
    dataset = store.save_ohlcv(
        session=session,
        symbol=symbol,
        timeframe="1d",
        frame=frame,
        provider=f"provider-v37-{symbol}",
        bundle_name=f"bundle-v37-{symbol}",
    )
    assert dataset.bundle_id is not None
    bundle_id = int(dataset.bundle_id)
    bar_dates = [pd.Timestamp(value).tz_convert("Asia/Kolkata").date() for value in frame["datetime"].tolist()]
    upsert_provenance_rows(
        session,
        bundle_id=bundle_id,
        timeframe="1d",
        symbol=symbol,
        bar_dates=bar_dates,
        source_provider="INBOX",
        source_run_kind="data_updates",
        source_run_id=f"v37-{symbol}",
        confidence_score=confidence,
        reason="test_low_confidence",
        metadata={"test": True},
    )
    session.commit()
    return bundle_id, list(frame["datetime"].tolist())


def _base_settings_payload(settings: Settings) -> dict[str, object]:
    return {
        "paper_use_simulator_engine": True,
        "operate_mode": "live",
        "operate_safe_mode_on_fail": False,
        "data_quality_stale_severity": "WARN",
        "data_quality_stale_severity_override": True,
        "coverage_missing_latest_warn_pct": 100.0,
        "coverage_missing_latest_fail_pct": 100.0,
        "no_trade_enabled": False,
        "confidence_gate_enabled": True,
        "confidence_gate_avg_threshold": 70.0,
        "confidence_gate_low_symbol_threshold": 65.0,
        "confidence_gate_low_pct_threshold": 0.50,
        "confidence_gate_fallback_pct_threshold": 0.80,
        "confidence_gate_hard_floor": 55.0,
        "confidence_gate_action_on_trigger": "SHADOW_ONLY",
        "confidence_gate_lookback_days": 1,
        "trading_calendar_segment": settings.trading_calendar_segment,
    }


def test_confidence_gate_deterministic_for_same_inputs() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"CGDET_{uuid4().hex[:8].upper()}"
    with Session(engine) as session:
        bundle_id, datetimes = _seed_bundle_with_low_confidence(
            session=session,
            store=store,
            symbol=symbol,
            confidence=52.0,
        )
        asof_ts = pd.Timestamp(datetimes[-1]).to_pydatetime()
        overrides = _base_settings_payload(settings)
        first = evaluate_confidence_gate(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            asof_ts=asof_ts,
            operate_mode="live",
            overrides=overrides,
            persist=False,
        )
        second = evaluate_confidence_gate(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            asof_ts=asof_ts,
            operate_mode="live",
            overrides=overrides,
            persist=False,
        )
        assert first["decision"] == second["decision"]
        assert first["reasons"] == second["reasons"]
        assert first["summary"] == second["summary"]


def test_confidence_gate_shadow_only_keeps_live_state_intact() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"CGSHDW_{uuid4().hex[:8].upper()}"
    with Session(engine) as session:
        bundle_id, datetimes = _seed_bundle_with_low_confidence(
            session=session,
            store=store,
            symbol=symbol,
            confidence=50.0,
        )
        state = get_or_create_paper_state(session, settings)
        merged = dict(state.settings_json or {})
        merged.update(_base_settings_payload(settings))
        state.settings_json = merged
        session.add(state)
        session.commit()
        before = get_paper_state_payload(session, settings)
        result = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "asof": pd.Timestamp(datetimes[-1]).to_pydatetime().isoformat(),
                "signals": [
                    {
                        "symbol": symbol,
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 104.0,
                        "stop_distance": 2.0,
                        "signal_strength": 1.0,
                        "adv": 1_000_000.0,
                        "vol_scale": 1.0,
                    }
                ],
                "auto_generate_signals": False,
            },
            store=store,
        )
        after = get_paper_state_payload(session, settings)
        assert str(result.get("execution_mode")) == "SHADOW"
        assert bool(result.get("live_state_mutated")) is False
        confidence_gate = result.get("confidence_gate", {})
        assert str(confidence_gate.get("decision", "")).upper() == "SHADOW_ONLY"
        assert before["state"]["cash"] == after["state"]["cash"]
        assert before["state"]["equity"] == after["state"]["equity"]
        assert len(after["positions"]) == len(before["positions"])
        shadow_row = session.exec(
            select(ShadowPaperState)
            .where(ShadowPaperState.bundle_id == bundle_id)
            .order_by(ShadowPaperState.updated_at.desc(), ShadowPaperState.id.desc())
        ).first()
        assert shadow_row is not None


def test_confidence_gate_block_entries_allows_exit_and_records_skip_reason() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"CGBLOCK_{uuid4().hex[:8].upper()}"
    with Session(engine) as session:
        bundle_id, datetimes = _seed_bundle_with_low_confidence(
            session=session,
            store=store,
            symbol=symbol,
            confidence=45.0,
        )
        state = get_or_create_paper_state(session, settings)
        merged = dict(state.settings_json or {})
        merged.update(_base_settings_payload(settings))
        merged["confidence_gate_enabled"] = False
        state.settings_json = merged
        session.add(state)
        session.commit()

        open_result = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "asof": pd.Timestamp(datetimes[-1]).to_pydatetime().isoformat(),
                "signals": [
                    {
                        "symbol": symbol,
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 104.0,
                        "stop_distance": 2.0,
                        "signal_strength": 1.0,
                        "adv": 1_000_000.0,
                        "vol_scale": 1.0,
                    }
                ],
                "auto_generate_signals": False,
            },
            store=store,
        )
        assert int(open_result.get("selected_signals_count", 0)) >= 1

        merged = dict(state.settings_json or {})
        merged["confidence_gate_enabled"] = True
        merged["confidence_gate_action_on_trigger"] = "BLOCK_ENTRIES"
        merged["confidence_gate_avg_threshold"] = 90.0
        state.settings_json = merged
        session.add(state)
        session.commit()

        close_result = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "asof": pd.Timestamp(datetimes[-1]).to_pydatetime().isoformat(),
                "mark_prices": {symbol: 90.0},
                "signals": [
                    {
                        "symbol": symbol,
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 105.0,
                        "stop_distance": 2.0,
                        "signal_strength": 1.2,
                        "adv": 1_000_000.0,
                        "vol_scale": 1.0,
                    }
                ],
                "auto_generate_signals": False,
            },
            store=store,
        )
        confidence_gate = close_result.get("confidence_gate", {})
        assert str(confidence_gate.get("decision", "")).upper() == "BLOCK_ENTRIES"
        skipped = close_result.get("skipped_signals", [])
        assert any(
            str(item.get("reason", "")) == "confidence_gate_block_entries"
            for item in skipped
            if isinstance(item, dict)
        )
        assert len(close_result.get("positions", [])) == 0


def test_confidence_gate_history_endpoints_and_report_metadata() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"CGAPI_{uuid4().hex[:8].upper()}"
    with Session(engine) as session:
        bundle_id, datetimes = _seed_bundle_with_low_confidence(
            session=session,
            store=store,
            symbol=symbol,
            confidence=50.0,
        )
        state = get_or_create_paper_state(session, settings)
        merged = dict(state.settings_json or {})
        merged.update(_base_settings_payload(settings))
        state.settings_json = merged
        session.add(state)
        session.commit()
        run_result = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "asof": pd.Timestamp(datetimes[-1]).to_pydatetime().isoformat(),
                "signals": [],
                "auto_generate_signals": True,
            },
            store=store,
        )
        paper_run_id = int(run_result.get("paper_run_id", 0))
        assert paper_run_id > 0
        run_row = session.get(PaperRun, paper_run_id)
        assert run_row is not None
        assert isinstance(run_row.summary_json, dict)
        assert isinstance(run_row.summary_json.get("confidence_gate"), dict)

        report = generate_daily_report(
            session=session,
            settings=settings,
            report_date=pd.Timestamp(datetimes[-1]).date(),
            bundle_id=bundle_id,
            policy_id=None,
            overwrite=True,
        )
        content = report.content_json if isinstance(report.content_json, dict) else {}
        assert isinstance(content.get("confidence_gate"), dict)
        assert str((content.get("confidence_gate") or {}).get("decision", "")).upper() in {
            "PASS",
            "SHADOW_ONLY",
            "BLOCK_ENTRIES",
        }

        snapshots = session.exec(
            select(ConfidenceGateSnapshot).where(ConfidenceGateSnapshot.bundle_id == bundle_id)
        ).all()
        assert len(snapshots) >= 1

    with TestClient(app) as client:
        latest = client.get(f"/api/confidence-gate/latest?bundle_id={bundle_id}&timeframe=1d")
        assert latest.status_code == 200
        latest_payload = latest.json().get("data")
        assert isinstance(latest_payload, dict)
        assert str(latest_payload.get("decision", "")).upper() in {
            "PASS",
            "SHADOW_ONLY",
            "BLOCK_ENTRIES",
        }

        history = client.get(
            f"/api/confidence-gate/history?bundle_id={bundle_id}&timeframe=1d&limit=30"
        )
        assert history.status_code == 200
        entries = history.json().get("data", [])
        assert isinstance(entries, list)
        assert len(entries) >= 1
