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
from app.engine.costs import estimate_equity_delivery_cost, estimate_futures_cost


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


def _down_frame(rows: int = 260, start: float = 320.0) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start - rows + 1, rows)
    # Force a deterministic downside breakout on the decision bar (len-2),
    # while keeping the last bar available as next-bar fill.
    close[-2] = close[-3] - 4.0
    close[-1] = close[-2] - 1.5
    frame = pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 3_200_000),
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


def _seed_equity_and_future_bundle(
    *,
    provider: str,
    underlying_symbol: str,
    lot_size: int = 25,
    downtrend: bool = False,
) -> tuple[int, int]:
    settings = get_settings()
    store = DataStore(parquet_root=settings.parquet_root, duckdb_path=settings.duckdb_path)
    bundle_name = f"bundle-{provider}"
    frame_builder = _down_frame if downtrend else _frame
    equity_start = 620.0 if downtrend else 200.0
    futures_start = 625.0 if downtrend else 205.0
    with Session(engine) as session:
        equity_ds = store.save_ohlcv(
            session=session,
            symbol=underlying_symbol,
            timeframe="1d",
            frame=frame_builder(start=equity_start),
            provider=provider,
            bundle_name=bundle_name,
            instrument_kind="EQUITY_CASH",
        )
        store.save_ohlcv(
            session=session,
            symbol=f"{underlying_symbol}_FUT",
            timeframe="1d",
            frame=frame_builder(start=futures_start),
            provider=provider,
            bundle_name=bundle_name,
            instrument_kind="STOCK_FUT",
            underlying=underlying_symbol,
            lot_size=lot_size,
        )
        assert equity_ds.bundle_id is not None
        assert equity_ds.id is not None
        return int(equity_ds.bundle_id), int(equity_ds.id)


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
                "active_ensemble_id": None,
                "allowed_sides": ["BUY"],
                "operate_mode": "offline",
                "data_quality_stale_severity": "WARN",
                "data_quality_stale_severity_override": True,
                "no_trade_enabled": False,
                "no_trade_regimes": ["HIGH_VOL"],
                "no_trade_cooldown_trading_days": 0,
                "no_trade_max_realized_vol_annual": 10.0,
                "no_trade_min_breadth_pct": 0.0,
                "no_trade_min_trend_strength": 0.0,
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
    symbol = "INTRADAY_SHORT_ONLY"
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
                        "symbol": symbol,
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
                "mark_prices": {symbol: 97.0},
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


def test_sell_prefers_stock_fut_when_available() -> None:
    provider = f"futpref-{uuid4().hex[:8]}"
    bundle_id, _ = _seed_equity_and_future_bundle(
        provider=provider,
        underlying_symbol="FUTSEL_A",
        lot_size=25,
    )

    with _client_inline_jobs() as client:
        _reset_paper_state()
        update = client.put(
            "/api/settings",
            json={
                "allowed_sides": ["BUY", "SELL"],
                "slippage_base_bps": 0.0,
                "slippage_vol_factor": 0.0,
                "commission_bps": 0.0,
            },
        )
        assert update.status_code == 200

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "signals": [
                    {
                        "symbol": "FUTSEL_A",
                        "side": "SELL",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 210.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.95,
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
        assert result.get("selected_signals_count") == 1

        positions = result.get("positions") or []
        assert len(positions) == 1
        assert positions[0].get("instrument_kind") == "STOCK_FUT"
        assert positions[0].get("symbol") == "FUTSEL_A_FUT"
        assert positions[0].get("must_exit_by_eod") is False

        selected = result.get("selected_signals") or []
        assert selected
        assert selected[0].get("instrument_choice_reason") in {
            "swing_short_requires_futures",
            "provided_futures_signal",
        }


def test_sell_without_allowed_short_instrument_returns_explicit_reason() -> None:
    provider = f"futskip-{uuid4().hex[:8]}"
    dataset_id, _ = _seed_dataset_symbols(provider=provider, symbols=["NOSHORT_A"])

    with _client_inline_jobs() as client:
        _reset_paper_state()
        policy_name = f"ShortPolicy-{uuid4().hex[:6]}"
        with Session(engine) as session:
            policy = Policy(
                name=policy_name,
                definition_json={
                    "universe": {"dataset_id": dataset_id, "symbol_scope": "all"},
                    "timeframes": ["1d"],
                    "allowed_instruments": {"BUY": ["EQUITY_CASH"], "SELL": ["STOCK_FUT"]},
                    "regime_map": {
                        "TREND_UP": {
                            "strategy_key": "trend_breakout",
                            "params": {},
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
                "allowed_sides": ["BUY", "SELL"],
            }
            session.add(state)
            session.commit()

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "dataset_id": dataset_id,
                "signals": [
                    {
                        "symbol": "NOSHORT_A",
                        "side": "SELL",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 150.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.8,
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
        assert any(item.get("reason") == "no_short_instrument_available" for item in skipped)


def test_futures_sizing_respects_lot_size_and_margin_reserve() -> None:
    provider = f"futsize-{uuid4().hex[:8]}"
    bundle_id, _ = _seed_equity_and_future_bundle(
        provider=provider,
        underlying_symbol="FUTLOT_A",
        lot_size=25,
    )

    with _client_inline_jobs() as client:
        _reset_paper_state()
        update = client.put(
            "/api/settings",
            json={
                "allowed_sides": ["BUY", "SELL"],
                "slippage_base_bps": 0.0,
                "slippage_vol_factor": 0.0,
                "commission_bps": 0.0,
                "futures_initial_margin_pct": 0.2,
                "cost_model_enabled": False,
            },
        )
        assert update.status_code == 200

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "signals": [
                    {
                        "symbol": "FUTLOT_A",
                        "side": "SELL",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 200.0,
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
        assert result.get("selected_signals_count") == 1
        position = (result.get("positions") or [])[0]
        assert position.get("instrument_kind") == "STOCK_FUT"
        assert int(position.get("qty_lots", 0)) == 40
        assert int(position.get("qty", 0)) == 1000
        fill_price = float(position.get("avg_price", 0.0))
        expected_margin = 1000 * fill_price * 0.2
        assert abs(float(position.get("margin_reserved", 0.0)) - expected_margin) < 1e-6
        state = result.get("state") or {}
        expected_cash = 1_000_000.0 - expected_margin
        assert abs(float(state.get("cash", 0.0)) - expected_cash) < 1e-4


def test_futures_cost_model_changes_cash_consistently() -> None:
    provider = f"futcost-{uuid4().hex[:8]}"
    bundle_id, _ = _seed_equity_and_future_bundle(
        provider=provider,
        underlying_symbol="FUTCOST_A",
        lot_size=50,
    )

    with _client_inline_jobs() as client:
        _reset_paper_state()
        update = client.put(
            "/api/settings",
            json={
                "allowed_sides": ["BUY", "SELL"],
                "slippage_base_bps": 0.0,
                "slippage_vol_factor": 0.0,
                "commission_bps": 0.0,
                "cost_model_enabled": True,
                "cost_mode": "delivery",
                "futures_initial_margin_pct": 0.18,
            },
        )
        assert update.status_code == 200
        settings_payload = client.get("/api/settings").json()["data"]

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "signals": [
                    {
                        "symbol": "FUTCOST_A",
                        "side": "SELL",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 220.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.95,
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
        position = (result.get("positions") or [])[0]
        qty = int(position.get("qty", 0))
        fill_price = float(position.get("avg_price", 0.0))
        notional = qty * fill_price
        margin_required = notional * 0.18

        expected_cost = estimate_futures_cost(
            notional=notional,
            side="SELL",
            config={
                "brokerage_bps": settings_payload.get("brokerage_bps"),
                "stt_delivery_buy_bps": settings_payload.get("stt_delivery_buy_bps"),
                "stt_delivery_sell_bps": settings_payload.get("stt_delivery_sell_bps"),
                "stt_intraday_buy_bps": settings_payload.get("stt_intraday_buy_bps"),
                "stt_intraday_sell_bps": settings_payload.get("stt_intraday_sell_bps"),
                "exchange_txn_bps": settings_payload.get("exchange_txn_bps"),
                "sebi_bps": settings_payload.get("sebi_bps"),
                "stamp_delivery_buy_bps": settings_payload.get("stamp_delivery_buy_bps"),
                "stamp_intraday_buy_bps": settings_payload.get("stamp_intraday_buy_bps"),
                "gst_rate": settings_payload.get("gst_rate"),
                "futures_brokerage_bps": settings_payload.get("futures_brokerage_bps"),
                "futures_stt_sell_bps": settings_payload.get("futures_stt_sell_bps"),
                "futures_exchange_txn_bps": settings_payload.get("futures_exchange_txn_bps"),
                "futures_stamp_buy_bps": settings_payload.get("futures_stamp_buy_bps"),
            },
        )
        state = result.get("state") or {}
        expected_cash = 1_000_000.0 - margin_required - expected_cost
        assert abs(float(state.get("cash", 0.0)) - expected_cash) < 1e-4


def test_policy_autopilot_generated_sell_prefers_futures() -> None:
    provider = f"auto-fut-{uuid4().hex[:8]}"
    bundle_id, _ = _seed_equity_and_future_bundle(
        provider=provider,
        underlying_symbol="AUTOFUT_A",
        lot_size=25,
        downtrend=True,
    )

    with _client_inline_jobs() as client:
        _reset_paper_state()
        with Session(engine) as session:
            policy = Policy(
                name=f"AutoFut-{uuid4().hex[:6]}",
                definition_json={
                    "universe": {"bundle_id": bundle_id, "symbol_scope": "all", "max_symbols_scan": 10},
                    "timeframes": ["1d"],
                    "regime_map": {
                        "TREND_UP": {
                            "strategy_key": "trend_breakout",
                            "params": {
                                "trend_breakout": {
                                    "direction": "both",
                                    "trend_period": 50,
                                    "breakout_lookback": 10,
                                    "atr_stop_mult": 2.0,
                                }
                            },
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
                "allowed_sides": ["BUY", "SELL"],
            }
            session.add(state)
            session.commit()

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "auto_generate_signals": True,
                "signals": [],
                "mark_prices": {},
            },
        )
        assert run.status_code == 200
        job = _wait_job(client, run.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"
        result = job["result_json"] or {}
        assert result.get("signals_source") == "generated"
        assert int(result.get("generated_signals_count", 0)) >= 1
        selected = result.get("selected_signals") or []
        assert selected
        assert any(
            row.get("side") == "SELL" and row.get("instrument_kind") == "STOCK_FUT"
            for row in selected
        )
