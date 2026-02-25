from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import DailyConfidenceAggregate, PaperRun
from app.db.session import engine, init_db
from app.engine.simulator import (
    SimulationConfig,
    simulate_portfolio_step,
)
from app.main import app
from app.services.confidence_agg import (
    upsert_daily_confidence_agg,
)
from app.services.confidence_gate import (
    compute_confidence_risk_scale,
    evaluate_confidence_gate,
)
from app.services.data_provenance import upsert_provenance_rows
from app.services.data_store import DataStore
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
            "volume": np.full(days, 1_500_000),
        }
    )


def _buy_signal(index: pd.DatetimeIndex, at: int) -> dict[str, pd.Series]:
    buy = pd.Series(False, index=index)
    sell = pd.Series(False, index=index)
    buy.iloc[at] = True
    return {"BUY": buy, "SELL": sell}


def _seed_bundle_with_provenance(
    *,
    session: Session,
    store: DataStore,
    symbol: str,
    start_day: str,
    days: int,
    confidence_score: float,
    source_provider: str,
) -> tuple[int, list[datetime], dt_date]:
    frame = _frame(start_day=start_day, days=days, start_price=100.0)
    dataset = store.save_ohlcv(
        session=session,
        symbol=symbol,
        timeframe="1d",
        frame=frame,
        provider=f"v38-{source_provider.lower()}",
        bundle_name=f"bundle-v38-{symbol}",
    )
    assert dataset.bundle_id is not None
    bar_dates = [pd.Timestamp(value).tz_convert("Asia/Kolkata").date() for value in frame["datetime"].tolist()]
    upsert_provenance_rows(
        session,
        bundle_id=int(dataset.bundle_id),
        timeframe="1d",
        symbol=symbol,
        bar_dates=bar_dates,
        source_provider=source_provider,
        source_run_kind="provider_updates",
        source_run_id=f"v38-{symbol}",
        confidence_score=confidence_score,
        reason="v38_test",
        metadata={"test": True},
    )
    session.commit()
    asof_ts = [pd.Timestamp(value).to_pydatetime() for value in frame["datetime"].tolist()]
    return int(dataset.bundle_id), asof_ts, bar_dates[-1]


def _confidence_overrides(settings: Settings) -> dict[str, object]:
    return {
        "operate_mode": "live",
        "confidence_gate_enabled": True,
        "confidence_gate_avg_threshold": 70.0,
        "confidence_gate_low_symbol_threshold": 65.0,
        "confidence_gate_low_pct_threshold": 0.5,
        "confidence_gate_fallback_pct_threshold": 0.8,
        "confidence_gate_hard_floor": 55.0,
        "confidence_gate_action_on_trigger": "SHADOW_ONLY",
        "confidence_gate_lookback_days": 1,
        "confidence_risk_scaling_enabled": True,
        "confidence_risk_scale_exponent": 1.0,
        "confidence_risk_scale_low_threshold": 0.35,
        "trading_calendar_segment": settings.trading_calendar_segment,
    }


def test_confidence_risk_scale_mapping_is_correct() -> None:
    assert compute_confidence_risk_scale(55.0, hard_floor=55.0, avg_threshold=70.0) == 0.0
    assert compute_confidence_risk_scale(70.0, hard_floor=55.0, avg_threshold=70.0) == 1.0
    mid = compute_confidence_risk_scale(62.0, hard_floor=55.0, avg_threshold=70.0)
    assert abs(mid - (7.0 / 15.0)) < 1e-9
    shaped = compute_confidence_risk_scale(
        62.0,
        hard_floor=55.0,
        avg_threshold=70.0,
        exponent=1.5,
    )
    assert 0.0 < shaped < mid


def test_simulator_sizing_uses_confidence_risk_scale() -> None:
    signal = {
        "symbol": "V38SIZE",
        "side": "BUY",
        "price": 150.0,
        "stop_distance": 2.0,
        "signal_strength": 1.0,
        "adv": 0.0,
        "vol_scale": 0.0,
    }
    asof = pd.Timestamp("2026-01-20T00:00:00Z")
    base = simulate_portfolio_step(
        signals=[signal],
        open_positions=[],
        mark_prices={"V38SIZE": 150.0},
        asof=asof,
        cash=200_000.0,
        equity_reference=200_000.0,
        config=SimulationConfig(
            initial_equity=200_000.0,
            risk_per_trade=0.01,
            allow_long=True,
            allow_short=False,
            max_position_value_pct_adv=1.0,
            min_notional=0.0,
            slippage_base_bps=0.0,
            slippage_vol_factor=0.0,
            commission_bps=0.0,
            confidence_risk_scaling_enabled=True,
            confidence_risk_scale=1.0,
        ),
    )
    reduced = simulate_portfolio_step(
        signals=[signal],
        open_positions=[],
        mark_prices={"V38SIZE": 150.0},
        asof=asof,
        cash=200_000.0,
        equity_reference=200_000.0,
        config=SimulationConfig(
            initial_equity=200_000.0,
            risk_per_trade=0.01,
            allow_long=True,
            allow_short=False,
            max_position_value_pct_adv=1.0,
            min_notional=0.0,
            slippage_base_bps=0.0,
            slippage_vol_factor=0.0,
            commission_bps=0.0,
            confidence_risk_scaling_enabled=True,
            confidence_risk_scale=0.4,
        ),
    )
    assert len(base.positions) == 1
    assert len(reduced.positions) == 1
    base_qty = int(base.positions[0]["qty"])
    reduced_qty = int(reduced.positions[0]["qty"])
    assert reduced_qty < base_qty


def test_daily_confidence_agg_upsert_is_idempotent_and_recomputes_on_threshold_change() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"V38IDEM_{uuid4().hex[:6].upper()}"
    with Session(engine) as session:
        bundle_id, _, latest_day = _seed_bundle_with_provenance(
            session=session,
            store=store,
            symbol=symbol,
            start_day="2026-02-10",
            days=3,
            confidence_score=62.0,
            source_provider="NSE_EOD",
        )
        overrides = _confidence_overrides(settings)
        first_row, first_changed = upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=latest_day,
            operate_mode="live",
            overrides=overrides,
            force=False,
        )
        session.commit()
        second_row, second_changed = upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=latest_day,
            operate_mode="live",
            overrides=overrides,
            force=False,
        )
        assert first_changed is True
        assert second_changed is False
        assert first_row.id == second_row.id
        second_signature = str(second_row.thresholds_signature)

        changed_overrides = dict(overrides)
        changed_overrides["confidence_gate_avg_threshold"] = 75.0
        third_row, third_changed = upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=latest_day,
            operate_mode="live",
            overrides=changed_overrides,
            force=False,
        )
        session.commit()
        assert third_changed is True
        assert str(third_row.thresholds_signature) != second_signature


def test_daily_agg_gate_decision_matches_gate_engine() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"V38PAR_{uuid4().hex[:6].upper()}"
    with Session(engine) as session:
        bundle_id, asof_list, latest_day = _seed_bundle_with_provenance(
            session=session,
            store=store,
            symbol=symbol,
            start_day="2026-02-12",
            days=4,
            confidence_score=58.0,
            source_provider="NSE_EOD",
        )
        overrides = _confidence_overrides(settings)
        row, _ = upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=latest_day,
            operate_mode="live",
            overrides=overrides,
            force=True,
        )
        session.commit()
        gate = evaluate_confidence_gate(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            asof_ts=asof_list[-1],
            operate_mode="live",
            overrides=overrides,
            persist=False,
        )
        assert str(row.gate_decision).upper() == str(gate.get("decision", "PASS")).upper()
        assert list(row.gate_reasons_json or []) == list(gate.get("reasons", []))


def test_provider_trend_endpoint_reads_materialized_aggregates() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"V38TREND_{uuid4().hex[:6].upper()}"
    with Session(engine) as session:
        bundle_id, _, latest_day = _seed_bundle_with_provenance(
            session=session,
            store=store,
            symbol=symbol,
            start_day="2026-02-16",
            days=4,
            confidence_score=64.0,
            source_provider="NSE_EOD",
        )
        overrides = _confidence_overrides(settings)
        upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=latest_day,
            operate_mode="live",
            overrides=overrides,
            force=True,
        )
        upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=latest_day - pd.Timedelta(days=1),
            operate_mode="live",
            overrides=overrides,
            force=True,
        )
        session.commit()

    with TestClient(app) as client:
        response = client.get(
            f"/api/providers/status/trend?bundle_id={bundle_id}&timeframe=1d&days=10"
        )
        assert response.status_code == 200
        rows = response.json().get("data", {}).get("trend", [])
        assert isinstance(rows, list)
        assert len(rows) >= 1
        assert "confidence_risk_scale" in rows[-1]
        assert "decision" in rows[-1]


def test_daily_report_uses_materialized_aggregate_values() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"V38RPT_{uuid4().hex[:6].upper()}"
    with Session(engine) as session:
        bundle_id, _, latest_day = _seed_bundle_with_provenance(
            session=session,
            store=store,
            symbol=symbol,
            start_day="2026-02-18",
            days=3,
            confidence_score=60.0,
            source_provider="NSE_EOD",
        )
        overrides = _confidence_overrides(settings)
        agg_row, _ = upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=bundle_id,
            timeframe="1d",
            trading_date=latest_day,
            operate_mode="live",
            overrides=overrides,
            force=True,
        )
        session.flush()
        run = PaperRun(
            bundle_id=bundle_id,
            policy_id=None,
            asof_ts=datetime.combine(latest_day, datetime.min.time(), tzinfo=UTC),
            mode="LIVE",
            regime="TREND_UP",
            signals_source="provided",
            summary_json={
                "execution_mode": "LIVE",
                "positions_after": 0,
                "drawdown": 0.0,
                "net_pnl": 0.0,
                "timeframes": ["1d"],
                "confidence_gate": {
                    "decision": "PASS",
                    "reasons": [],
                    "summary": {
                        "avg_confidence": 99.0,
                        "pct_low_confidence": 0.0,
                        "provider_mix": {"UPSTOX": 1.0},
                        "confidence_risk_scale": 1.0,
                    },
                },
            },
        )
        session.add(run)
        session.commit()

        report = generate_daily_report(
            session=session,
            settings=settings,
            report_date=latest_day,
            bundle_id=bundle_id,
            policy_id=None,
            overwrite=True,
        )
        content = report.content_json if isinstance(report.content_json, dict) else {}
        confidence_gate = content.get("confidence_gate", {})
        confidence_scaling = content.get("confidence_risk_scaling", {})
        assert isinstance(confidence_gate, dict)
        assert isinstance(confidence_scaling, dict)
        assert str(confidence_gate.get("decision", "")).upper() == str(agg_row.gate_decision).upper()
        assert abs(float(confidence_gate.get("avg_confidence", 0.0)) - float(agg_row.avg_confidence)) < 1e-9
        assert abs(
            float(confidence_scaling.get("scale", 0.0)) - float(agg_row.confidence_risk_scale)
        ) < 1e-9
        stored = session.exec(
            select(DailyConfidenceAggregate)
            .where(DailyConfidenceAggregate.bundle_id == bundle_id)
            .where(DailyConfidenceAggregate.timeframe == "1d")
            .where(DailyConfidenceAggregate.trading_date == latest_day)
        ).first()
        assert stored is not None
