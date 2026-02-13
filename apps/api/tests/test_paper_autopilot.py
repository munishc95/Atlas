from __future__ import annotations

import os
import time
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import DatasetBundle, PaperOrder, PaperPosition, PaperState, Policy
from app.db.session import engine
from app.main import app
from app.services.data_store import DataStore
from app.engine.costs import estimate_equity_delivery_cost


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


def _frame(rows: int = 260, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    frame = pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 3_000_000),
        }
    )
    return frame


def _seed_dataset_symbols(
    *,
    provider: str,
    symbols: list[str],
    timeframe: str = "1d",
    bundle_name: str | None = None,
) -> tuple[int, DataStore]:
    settings = get_settings()
    store = DataStore(parquet_root=settings.parquet_root, duckdb_path=settings.duckdb_path)
    dataset_id: int | None = None
    with Session(engine) as session:
        for offset, symbol in enumerate(symbols):
            dataset = store.save_ohlcv(
                session=session,
                symbol=symbol,
                timeframe=timeframe,
                frame=_frame(start=100.0 + (offset * 2.0)),
                provider=provider,
                bundle_name=bundle_name,
            )
            if dataset_id is None:
                dataset_id = dataset.id
    assert dataset_id is not None
    return dataset_id, store


def _reset_paper_state() -> None:
    with Session(engine) as session:
        for row in session.exec(select(PaperPosition)).all():
            session.delete(row)
        for row in session.exec(select(PaperOrder)).all():
            session.delete(row)
        state = session.get(PaperState, 1)
        if state is not None:
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
            }
            session.add(state)
        session.commit()


def test_dataset_scoped_preview_excludes_outside_symbols() -> None:
    provider = f"scope-{uuid4().hex[:8]}"
    dataset_id, _ = _seed_dataset_symbols(provider=provider, symbols=["SCOPE_A", "SCOPE_B"])
    _seed_dataset_symbols(provider=f"outside-{uuid4().hex[:8]}", symbols=["OUTSIDE_A"])

    with _client_inline_jobs() as client:
        _reset_paper_state()
        response = client.post(
            "/api/paper/signals/preview",
            json={
                "dataset_id": dataset_id,
                "timeframes": ["1d"],
                "symbol_scope": "all",
                "max_symbols_scan": 10,
                "seed": 9,
                "regime": "TREND_UP",
            },
        )
        assert response.status_code == 200
        payload = response.json()["data"]
        symbols = {item["symbol"] for item in payload["signals"]}
        assert symbols
        assert symbols.issubset({"SCOPE_A", "SCOPE_B"})
        assert "OUTSIDE_A" not in symbols


def test_bundle_scoped_preview_only_returns_bundle_members() -> None:
    provider = f"bundle-{uuid4().hex[:8]}"
    _, store = _seed_dataset_symbols(
        provider=provider,
        symbols=["BUNDLE_A", "BUNDLE_B"],
        bundle_name=f"alpha-{provider}",
    )
    with Session(engine) as session:
        alpha_bundle = session.exec(
            select(DatasetBundle).where(DatasetBundle.name == f"alpha-{provider}")
        ).first()
    _seed_dataset_symbols(
        provider=provider,
        symbols=["BUNDLE_C"],
        bundle_name=f"beta-{provider}",
    )

    assert alpha_bundle is not None and alpha_bundle.id is not None
    with _client_inline_jobs() as client:
        _reset_paper_state()
        response = client.post(
            "/api/paper/signals/preview",
            json={
                "bundle_id": alpha_bundle.id,
                "timeframes": ["1d"],
                "symbol_scope": "all",
                "max_symbols_scan": 20,
                "seed": 5,
                "regime": "TREND_UP",
            },
        )
        assert response.status_code == 200
        payload = response.json()["data"]
        symbols = {item["symbol"] for item in payload["signals"]}
        assert symbols
        assert symbols.issubset({"BUNDLE_A", "BUNDLE_B"})
        assert "BUNDLE_C" not in symbols


def test_liquidity_sampling_is_deterministic_with_seed() -> None:
    provider = f"seed-{uuid4().hex[:8]}"
    dataset_id, store = _seed_dataset_symbols(
        provider=provider,
        symbols=["SEED_A", "SEED_B", "SEED_C", "SEED_D"],
    )

    with Session(engine) as session:
        first = store.sample_dataset_symbols(
            session,
            dataset_id=dataset_id,
            timeframe="1d",
            symbol_scope="all",
            max_symbols_scan=4,
            seed=13,
        )
        second = store.sample_dataset_symbols(
            session,
            dataset_id=dataset_id,
            timeframe="1d",
            symbol_scope="all",
            max_symbols_scan=4,
            seed=13,
        )
        different_seed = store.sample_dataset_symbols(
            session,
            dataset_id=dataset_id,
            timeframe="1d",
            symbol_scope="all",
            max_symbols_scan=4,
            seed=99,
        )
    assert first == second
    assert first != different_seed


def test_policy_mode_autogenerates_signals_when_missing() -> None:
    provider = f"autogen-{uuid4().hex[:8]}"
    dataset_id, _ = _seed_dataset_symbols(provider=provider, symbols=["AUTO_SIG"])

    with _client_inline_jobs() as client:
        _reset_paper_state()
        with Session(engine) as session:
            policy = Policy(
                name=f"AutoPolicy-{uuid4().hex[:6]}",
                definition_json={
                    "universe": {
                        "dataset_id": dataset_id,
                        "symbol_scope": "all",
                        "max_symbols_scan": 5,
                    },
                    "timeframes": ["1d"],
                    "ranking": {"method": "robust_score", "seed": 7, "weights": {"signal": 1.0}},
                    "cost_model": {"enabled": False, "mode": "delivery"},
                    "regime_map": {
                        "TREND_UP": {
                            "strategy_key": "trend_breakout",
                            "params": {"atr_stop_mult": 2.0},
                            "risk_scale": 1.0,
                            "max_positions_scale": 1.0,
                        }
                    },
                },
            )
            session.add(policy)
            session.commit()
            session.refresh(policy)
            state = session.get(PaperState, 1)
            assert state is not None
            state.settings_json = {
                **(state.settings_json or {}),
                "paper_mode": "policy",
                "active_policy_id": policy.id,
            }
            session.add(state)
            session.commit()

        run = client.post("/api/paper/run-step", json={"regime": "TREND_UP", "signals": []})
        assert run.status_code == 200
        job = _wait_job(client, run.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"

        result = job["result_json"] or {}
        assert result.get("signals_source") == "generated"
        assert int(result.get("generated_signals_count", 0)) >= 1
        assert int(result.get("selected_signals_count", 0)) >= 1


def test_preview_does_not_create_orders() -> None:
    provider = f"preview-{uuid4().hex[:8]}"
    dataset_id, _ = _seed_dataset_symbols(provider=provider, symbols=["PRVW_A"])

    with _client_inline_jobs() as client:
        _reset_paper_state()
        before = client.get("/api/paper/orders").json()["data"]
        response = client.post(
            "/api/paper/signals/preview",
            json={
                "dataset_id": dataset_id,
                "timeframes": ["1d"],
                "symbol_scope": "all",
                "max_symbols_scan": 5,
                "regime": "TREND_UP",
            },
        )
        assert response.status_code == 200
        payload = response.json()["data"]
        assert payload["generated_signals_count"] >= 1
        after = client.get("/api/paper/orders").json()["data"]
        assert len(after) == len(before)


def test_cost_model_changes_cash_with_estimated_costs() -> None:
    with _client_inline_jobs() as client:
        _reset_paper_state()
        update = client.put(
            "/api/settings",
            json={
                "cost_model_enabled": True,
                "cost_mode": "delivery",
                "commission_bps": 0.0,
                "slippage_base_bps": 0.0,
                "slippage_vol_factor": 0.0,
                "paper_mode": "strategy",
                "active_policy_id": None,
            },
        )
        assert update.status_code == 200

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "auto_generate_signals": False,
                "signals": [
                    {
                        "symbol": "NIFTY500",
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 100.0,
                        "stop_distance": 100.0,
                        "signal_strength": 0.9,
                        "adv": 10_000_000_000.0,
                        "vol_scale": 0.0,
                    }
                ],
                "mark_prices": {},
            },
        )
        assert run.status_code == 200
        job = _wait_job(client, run.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"
        result = job["result_json"] or {}
        state = result["state"]
        assert result.get("selected_signals_count") == 1

        qty = int(result["positions"][0]["qty"])
        fill_price = float(result["positions"][0]["avg_price"])
        notional = qty * fill_price
        settings_cfg = (client.get("/api/settings").json()["data"]) or {}
        expected_cost = estimate_equity_delivery_cost(
            notional=notional,
            side="BUY",
            config={
                "brokerage_bps": settings_cfg.get("brokerage_bps"),
                "stt_delivery_buy_bps": settings_cfg.get("stt_delivery_buy_bps"),
                "stt_delivery_sell_bps": settings_cfg.get("stt_delivery_sell_bps"),
                "stt_intraday_buy_bps": settings_cfg.get("stt_intraday_buy_bps"),
                "stt_intraday_sell_bps": settings_cfg.get("stt_intraday_sell_bps"),
                "exchange_txn_bps": settings_cfg.get("exchange_txn_bps"),
                "sebi_bps": settings_cfg.get("sebi_bps"),
                "stamp_delivery_buy_bps": settings_cfg.get("stamp_delivery_buy_bps"),
                "stamp_intraday_buy_bps": settings_cfg.get("stamp_intraday_buy_bps"),
                "gst_rate": settings_cfg.get("gst_rate"),
            },
        )
        expected_cash = 1_000_000.0 - notional - expected_cost
        assert state["cash"] < 1_000_000.0 - notional + 1e-6
        assert abs(state["cash"] - expected_cash) < 1e-4


def test_allowed_sides_blocks_sell_signal_when_disabled() -> None:
    with _client_inline_jobs() as client:
        _reset_paper_state()
        update = client.put("/api/settings", json={"allowed_sides": ["BUY"]})
        assert update.status_code == 200

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "signals": [
                    {
                        "symbol": "NIFTY500",
                        "side": "SELL",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.9,
                        "adv": 10_000_000_000.0,
                        "vol_scale": 0.0,
                    }
                ],
                "mark_prices": {},
            },
        )
        assert run.status_code == 200
        job = _wait_job(client, run.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"
        result = job["result_json"] or {}
        assert result.get("selected_signals_count") == 0
        skipped = result.get("skipped_signals", [])
        assert any(item.get("reason") == "shorts_disabled" for item in skipped)


def test_intraday_short_forced_squareoff_at_eod() -> None:
    with _client_inline_jobs() as client:
        _reset_paper_state()
        update = client.put(
            "/api/settings",
            json={"allowed_sides": ["BUY", "SELL"], "paper_short_squareoff_time": "15:20"},
        )
        assert update.status_code == 200

        open_step = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "asof": "2026-01-05T10:00:00+05:30",
                "signals": [
                    {
                        "symbol": "NIFTY500",
                        "side": "SELL",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.9,
                        "adv": 10_000_000_000.0,
                        "vol_scale": 0.0,
                    }
                ],
                "mark_prices": {},
            },
        )
        assert open_step.status_code == 200
        open_job = _wait_job(client, open_step.json()["data"]["job_id"])
        assert open_job["status"] == "SUCCEEDED"
        open_result = open_job["result_json"] or {}
        assert open_result.get("selected_signals_count") == 1
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
                "mark_prices": {"NIFTY500": 97.0},
            },
        )
        assert close_step.status_code == 200
        close_job = _wait_job(client, close_step.json()["data"]["job_id"])
        assert close_job["status"] == "SUCCEEDED"
        close_result = close_job["result_json"] or {}
        assert len(close_result.get("positions") or []) == 0
        assert any(
            row.get("reason") == "EOD_SQUARE_OFF" for row in (close_result.get("orders") or [])
        )


def test_feature_cache_matches_computed_indicators() -> None:
    provider = f"feature-{uuid4().hex[:8]}"
    _, store = _seed_dataset_symbols(provider=provider, symbols=["FEAT_A"], timeframe="1d")

    base = store.load_ohlcv(symbol="FEAT_A", timeframe="1d")
    expected = store._compute_feature_frame(base)  # noqa: SLF001
    cached = store.load_features(symbol="FEAT_A", timeframe="1d")

    assert not cached.empty
    assert len(cached) == len(expected)
    for column in ("atr_14", "rsi_4", "ema_20", "ema_50", "adx_14", "bb_width"):
        left = np.nan_to_num(cached[column].to_numpy(), nan=0.0)
        right = np.nan_to_num(expected[column].to_numpy(), nan=0.0)
        assert np.allclose(left, right, atol=1e-8)
