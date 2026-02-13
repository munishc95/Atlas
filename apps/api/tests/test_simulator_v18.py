from __future__ import annotations

from datetime import date
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import get_settings
from app.db.models import Policy
from app.db.session import engine
from app.engine.backtester import BacktestConfig, run_backtest
from app.engine.simulator import SimulationConfig, run_simulation
from app.services.data_store import DataStore
from app.services.policy_simulation import simulate_policy_on_bundle
from app.strategies.templates import generate_signal_sides


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(*, rows: int = 90, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    close = np.array([start + step * i for i in range(rows)], dtype=float)
    frame = pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 3_000_000, dtype=float),
        }
    )
    return frame


def _sell_signal(index: pd.DatetimeIndex, at: int) -> dict[str, pd.Series]:
    buy = pd.Series(False, index=index)
    sell = pd.Series(False, index=index)
    sell.iloc[at] = True
    return {"BUY": buy, "SELL": sell}


def test_simulator_deterministic_for_same_inputs() -> None:
    frame = _frame(rows=80, start=120.0, step=0.4)
    entries = _sell_signal(pd.DatetimeIndex(frame["datetime"]), at=20)
    cfg = SimulationConfig(
        initial_equity=500_000.0,
        allow_long=True,
        allow_short=True,
        instrument_kind="EQUITY_CASH",
        equity_short_intraday_only=False,
        min_notional=0.0,
        slippage_base_bps=0.0,
        slippage_vol_factor=0.0,
        commission_bps=0.0,
        seed=42,
    )

    first = run_simulation(price_df=frame, entries=entries, symbol="DET_A", config=cfg)
    second = run_simulation(price_df=frame, entries=entries, symbol="DET_A", config=cfg)

    assert first.metadata["engine_version"] == second.metadata["engine_version"]
    assert first.metadata["seed"] == second.metadata["seed"] == 42
    assert first.metadata["data_digest"] == second.metadata["data_digest"]
    assert first.trades.to_dict("records") == second.trades.to_dict("records")
    assert first.equity_curve.to_dict("records") == second.equity_curve.to_dict("records")


def test_shadow_and_research_paths_share_execution_assumptions() -> None:
    settings = get_settings()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"SIMPAR_{unique[:4]}"
    provider = f"sim-parity-{unique}"
    frame = _frame(rows=180, start=200.0, step=0.3)

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame,
            provider=provider,
            bundle_name=f"bundle-{provider}",
        )
        assert dataset.bundle_id is not None
        policy = Policy(
            name=f"Policy-{unique}",
            definition_json={
                "universe": {"bundle_id": int(dataset.bundle_id), "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "regime_map": {
                    "TREND_UP": {
                        "strategy_key": "trend_breakout",
                        "params": {
                            "direction": "both",
                            "trend_period": 50,
                            "breakout_lookback": 15,
                            "atr_stop_mult": 2.0,
                            "atr_trail_mult": 2.0,
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

        shadow = simulate_policy_on_bundle(
            session=session,
            store=store,
            settings=settings,
            policy=policy,
            bundle_id=int(dataset.bundle_id),
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            regime="TREND_UP",
            seed=11,
        )

        scoped = store.load_ohlcv(symbol=symbol, timeframe="1d")
        params = {
            "direction": "both",
            "trend_period": 50,
            "breakout_lookback": 15,
            "atr_stop_mult": 2.0,
            "atr_trail_mult": 2.0,
        }
        signals = generate_signal_sides("trend_breakout", scoped, params=params)
        research = run_backtest(
            price_df=scoped,
            entries=signals,
            symbol=symbol,
            config=BacktestConfig(
                allow_long=True,
                allow_short=True,
                min_notional=0.0,
                cost_model_enabled=settings.cost_model_enabled,
                cost_mode=settings.cost_mode,
                risk_per_trade=settings.risk_per_trade,
                max_positions=settings.max_positions,
                atr_stop_mult=2.0,
                atr_trail_mult=2.0,
                seed=11,
            ),
        )

    assert shadow["engine_version"] == research.metadata["engine_version"]
    assert shadow["seed"] == research.metadata["seed"] == 11
    assert shadow["symbol_count"] == 1
    assert shadow["symbol_rows"][0]["trade_count"] == len(research.trades)
    research_period_return = float(
        research.equity_curve["equity"].iloc[-1] / research.equity_curve["equity"].iloc[0] - 1.0
    )
    assert abs(float(shadow["symbol_rows"][0]["period_return"]) - research_period_return) < 1e-8


def test_cash_short_intraday_forces_same_day_squareoff() -> None:
    frame = _frame(rows=40, start=250.0, step=-0.8)
    entries = _sell_signal(pd.DatetimeIndex(frame["datetime"]), at=20)
    result = run_simulation(
        price_df=frame,
        entries=entries,
        symbol="SHORT_EQ",
        config=SimulationConfig(
            allow_long=False,
            allow_short=True,
            instrument_kind="EQUITY_CASH",
            equity_short_intraday_only=True,
            min_notional=0.0,
                slippage_base_bps=0.0,
                slippage_vol_factor=0.0,
                commission_bps=0.0,
                atr_period=5,
            ),
        )
    assert not result.trades.empty
    trade = result.trades.iloc[0]
    assert trade["side"] == "SHORT"
    assert trade["reason"] == "EOD_SQUARE_OFF"
    assert pd.Timestamp(trade["entry_dt"]).date() == pd.Timestamp(trade["exit_dt"]).date()


def test_futures_short_can_hold_overnight_and_margin_released() -> None:
    frame = _frame(rows=50, start=300.0, step=-1.0)
    entries = _sell_signal(pd.DatetimeIndex(frame["datetime"]), at=20)
    initial = 1_000_000.0
    result = run_simulation(
        price_df=frame,
        entries=entries,
        symbol="SHORT_FUT",
        config=SimulationConfig(
            initial_equity=initial,
            allow_long=False,
            allow_short=True,
            instrument_kind="STOCK_FUT",
            lot_size=25,
            futures_initial_margin_pct=0.18,
            equity_short_intraday_only=False,
            min_notional=0.0,
                slippage_base_bps=0.0,
                slippage_vol_factor=0.0,
                commission_bps=0.0,
                atr_period=5,
            ),
        )
    assert not result.trades.empty
    trade = result.trades.iloc[0]
    assert trade["side"] == "SHORT"
    assert trade["instrument_kind"] == "STOCK_FUT"
    assert pd.Timestamp(trade["exit_dt"]).date() > pd.Timestamp(trade["entry_dt"]).date()
    assert trade["pnl"] > 0
    assert int(trade["qty"]) % 25 == 0
    final_equity = float(result.equity_curve.iloc[-1]["equity"])
    assert final_equity > initial


def test_futures_lot_rounding_and_insufficient_margin_skip_reason() -> None:
    frame = _frame(rows=35, start=150.0, step=0.2)
    entries = _sell_signal(pd.DatetimeIndex(frame["datetime"]), at=20)

    ok = run_simulation(
        price_df=frame,
        entries=entries,
        symbol="LOT_OK",
        config=SimulationConfig(
            initial_equity=500_000.0,
            allow_long=False,
            allow_short=True,
            instrument_kind="STOCK_FUT",
            lot_size=75,
            futures_initial_margin_pct=0.18,
            equity_short_intraday_only=False,
            min_notional=0.0,
                slippage_base_bps=0.0,
                slippage_vol_factor=0.0,
                commission_bps=0.0,
                atr_period=5,
            ),
        )
    assert not ok.trades.empty
    assert int(ok.trades.iloc[0]["qty"]) % 75 == 0

    low_cash = run_simulation(
        price_df=frame,
        entries=entries,
        symbol="LOT_LOW",
        config=SimulationConfig(
            initial_equity=1_000.0,
            risk_per_trade=1.0,
            allow_long=False,
            allow_short=True,
            instrument_kind="STOCK_FUT",
            lot_size=75,
            futures_initial_margin_pct=0.18,
            equity_short_intraday_only=False,
            min_notional=0.0,
                slippage_base_bps=0.0,
                slippage_vol_factor=0.0,
                commission_bps=0.0,
                atr_period=5,
            ),
        )
    assert low_cash.trades.empty
    assert any(row["reason"] == "insufficient_margin" for row in low_cash.skipped_signals)
