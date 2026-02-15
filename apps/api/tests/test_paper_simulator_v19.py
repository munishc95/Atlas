from __future__ import annotations

import os
import time
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import PaperOrder, PaperPosition, PaperState
from app.db.session import engine
from app.engine.simulator import SimulationConfig, simulate_portfolio_step
from app.main import app
from app.services.data_store import DataStore


def _client_inline_jobs() -> TestClient:
    os.environ["ATLAS_JOBS_INLINE"] = "true"
    get_settings.cache_clear()
    return TestClient(app)


def _wait_job(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        payload = client.get(f"/api/jobs/{job_id}").json()["data"]
        if payload["status"] in {"SUCCEEDED", "FAILED", "DONE"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} timed out")


def _frame(rows: int = 220, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 2_500_000),
        }
    )


def _seed_symbol(symbol: str, *, instrument_kind: str = "EQUITY_CASH", lot_size: int = 1) -> None:
    settings = get_settings()
    store = DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    with Session(engine) as session:
        store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_frame(),
            provider=f"v19-{uuid4().hex[:8]}",
            instrument_kind=instrument_kind,
            underlying=(symbol.replace("_FUT", "") if instrument_kind != "EQUITY_CASH" else None),
            lot_size=lot_size,
            bundle_name=f"v19-bundle-{uuid4().hex[:8]}",
        )


def _reset_paper_state(*, use_simulator_engine: bool) -> None:
    with Session(engine) as session:
        state = session.get(PaperState, 1)
        if state is None:
            return
        for row in session.exec(select(PaperPosition)).all():
            session.delete(row)
        for row in session.exec(select(PaperOrder)).all():
            session.delete(row)
        state.equity = 1_000_000.0
        state.cash = 1_000_000.0
        state.peak_equity = 1_000_000.0
        state.drawdown = 0.0
        state.kill_switch_active = False
        state.cooldown_days_left = 0
        state.settings_json = {
            **(state.settings_json or {}),
            "paper_mode": "strategy",
            "active_policy_id": None,
            "active_ensemble_id": None,
            "allowed_sides": ["BUY", "SELL"],
            "operate_mode": "offline",
            "data_quality_stale_severity": "WARN",
            "data_quality_stale_severity_override": True,
            "paper_use_simulator_engine": use_simulator_engine,
            "cost_model_enabled": False,
            "slippage_base_bps": 2.0,
            "slippage_vol_factor": 15.0,
            "commission_bps": 5.0,
            "no_trade_enabled": False,
            "no_trade_regimes": ["HIGH_VOL"],
            "no_trade_cooldown_trading_days": 0,
            "no_trade_max_realized_vol_annual": 10.0,
            "no_trade_min_breadth_pct": 0.0,
            "no_trade_min_trend_strength": 0.0,
        }
        session.add(state)
        session.commit()


def _manual_signal(symbol: str, *, side: str, instrument_kind: str = "EQUITY_CASH") -> dict:
    return {
        "symbol": symbol,
        "side": side,
        "template": "trend_breakout",
        "instrument_kind": instrument_kind,
        "price": 100.0,
        "stop_distance": 5.0,
        "signal_strength": 0.9,
        "adv": 10_000_000_000.0,
        "vol_scale": 0.01,
    }


@pytest.fixture(autouse=True)
def _reset_state_after_test() -> None:
    try:
        yield
    finally:
        _reset_paper_state(use_simulator_engine=False)


def test_paper_flag_false_uses_legacy_path() -> None:
    symbol = f"LEGACY_{uuid4().hex[:6].upper()}"
    _seed_symbol(symbol)
    with _client_inline_jobs() as client:
        _reset_paper_state(use_simulator_engine=False)
        response = client.post(
            "/api/paper/run-step",
            json={"regime": "TREND_UP", "signals": [_manual_signal(symbol, side="BUY")], "mark_prices": {}},
        )
        assert response.status_code == 200
        job = _wait_job(client, response.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"
        result = job["result_json"] or {}
        assert result.get("paper_engine") == "legacy"
        assert int(result.get("selected_signals_count", 0)) >= 1


def test_paper_flag_true_returns_simulator_metadata() -> None:
    symbol = f"SIM_{uuid4().hex[:6].upper()}"
    _seed_symbol(symbol)
    with _client_inline_jobs() as client:
        _reset_paper_state(use_simulator_engine=True)
        response = client.post(
            "/api/paper/run-step",
            json={"regime": "TREND_UP", "signals": [_manual_signal(symbol, side="BUY")], "mark_prices": {}},
        )
        assert response.status_code == 200
        job = _wait_job(client, response.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"
        result = job["result_json"] or {}
        assert result.get("paper_engine") == "simulator"
        assert isinstance(result.get("engine_version"), str) and result.get("engine_version")
        assert isinstance(result.get("data_digest"), str) and len(str(result.get("data_digest"))) > 16
        assert isinstance(result.get("seed"), int)


def test_paper_simulator_matches_shadow_step_outputs() -> None:
    symbol = f"PARITY_{uuid4().hex[:6].upper()}"
    _seed_symbol(symbol)
    asof = "2026-01-05T10:00:00+05:30"
    signal = _manual_signal(symbol, side="BUY")

    with _client_inline_jobs() as client:
        _reset_paper_state(use_simulator_engine=True)
        run = client.post(
            "/api/paper/run-step",
            json={"regime": "TREND_UP", "asof": asof, "signals": [signal], "mark_prices": {}},
        )
        assert run.status_code == 200
        job = _wait_job(client, run.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"
        result = job["result_json"] or {}
        selected = (result.get("selected_signals") or [])[0]

        settings_payload = client.get("/api/settings").json()["data"]
        shadow = simulate_portfolio_step(
            signals=[signal],
            open_positions=[],
            mark_prices={},
            asof=pd.Timestamp(asof),
            cash=1_000_000.0,
            equity_reference=1_000_000.0,
            config=SimulationConfig(
                risk_per_trade=float(settings_payload.get("risk_per_trade", 0.005)),
                max_positions=int(settings_payload.get("max_positions", 3)),
                commission_bps=float(settings_payload.get("commission_bps", 5.0)),
                slippage_base_bps=float(settings_payload.get("slippage_base_bps", 2.0)),
                slippage_vol_factor=float(settings_payload.get("slippage_vol_factor", 15.0)),
                max_position_value_pct_adv=float(settings_payload.get("max_position_value_pct_adv", 0.01)),
                allow_long=True,
                allow_short=True,
                cost_model_enabled=bool(settings_payload.get("cost_model_enabled", False)),
                cost_mode=str(settings_payload.get("cost_mode", "delivery")),
                seed=7,
            ),
        )

        shadow_signal = shadow.executed_signals[0]
        assert int(selected["qty"]) == int(shadow_signal["qty"])
        assert abs(float(selected["fill_price"]) - float(shadow_signal["fill_price"])) < 1e-8
        assert abs(float(result["cost_summary"]["entry_cost_total"]) - float(shadow.entry_cost_total)) < 1e-8
        assert abs(float(result["state"]["cash"]) - float(shadow.cash)) < 1e-8
        assert abs(float(result["state"]["equity"]) - float(shadow.equity)) < 1e-8


def test_cash_short_squareoff_and_futures_margin_skip_with_simulator() -> None:
    short_symbol = f"CSHRT_{uuid4().hex[:6].upper()}"
    fut_symbol = f"FUTM_{uuid4().hex[:6].upper()}_FUT"
    _seed_symbol(short_symbol)
    _seed_symbol(fut_symbol, instrument_kind="STOCK_FUT", lot_size=50)

    with _client_inline_jobs() as client:
        _reset_paper_state(use_simulator_engine=True)
        open_step = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "asof": "2026-01-05T10:00:00+05:30",
                "signals": [_manual_signal(short_symbol, side="SELL")],
                "mark_prices": {},
            },
        )
        assert open_step.status_code == 200
        open_job = _wait_job(client, open_step.json()["data"]["job_id"])
        assert open_job["status"] == "SUCCEEDED"
        open_result = open_job["result_json"] or {}
        assert any(
            row.get("side") == "SELL" and row.get("must_exit_by_eod") is True
            for row in (open_result.get("positions") or [])
        )

        close_step = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "asof": "2026-01-05T15:25:00+05:30",
                "signals": [],
                "mark_prices": {short_symbol: 98.0},
            },
        )
        assert close_step.status_code == 200
        close_job = _wait_job(client, close_step.json()["data"]["job_id"])
        assert close_job["status"] == "SUCCEEDED"
        close_result = close_job["result_json"] or {}
        assert len(close_result.get("positions") or []) == 0
        assert any(row.get("reason") == "EOD_SQUARE_OFF" for row in (close_result.get("orders") or []))

        with Session(engine) as session:
            state = session.get(PaperState, 1)
            assert state is not None
            state.cash = 1_000.0
            state.equity = 1_000_000.0
            session.add(state)
            session.commit()

        low_margin = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "signals": [_manual_signal(fut_symbol, side="SELL", instrument_kind="STOCK_FUT")],
                "mark_prices": {},
            },
        )
        assert low_margin.status_code == 200
        margin_job = _wait_job(client, low_margin.json()["data"]["job_id"])
        assert margin_job["status"] == "SUCCEEDED"
        margin_result = margin_job["result_json"] or {}
        skipped = margin_result.get("skipped_signals") or []
        assert any(item.get("reason") == "insufficient_margin" for item in skipped)
