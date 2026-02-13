from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlmodel import Session

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import Backtest, Trade
from app.engine.backtester import BacktestConfig, run_backtest
from app.services.jobs import append_job_log
from app.services.data_store import DataStore
from app.strategies.templates import generate_signals


def _build_config(settings: Settings, payload_config: dict[str, Any]) -> BacktestConfig:
    cost_params = {
        "brokerage_bps": float(payload_config.get("brokerage_bps", settings.brokerage_bps)),
        "stt_delivery_buy_bps": float(
            payload_config.get("stt_delivery_buy_bps", settings.stt_delivery_buy_bps)
        ),
        "stt_delivery_sell_bps": float(
            payload_config.get("stt_delivery_sell_bps", settings.stt_delivery_sell_bps)
        ),
        "stt_intraday_buy_bps": float(
            payload_config.get("stt_intraday_buy_bps", settings.stt_intraday_buy_bps)
        ),
        "stt_intraday_sell_bps": float(
            payload_config.get("stt_intraday_sell_bps", settings.stt_intraday_sell_bps)
        ),
        "exchange_txn_bps": float(
            payload_config.get("exchange_txn_bps", settings.exchange_txn_bps)
        ),
        "sebi_bps": float(payload_config.get("sebi_bps", settings.sebi_bps)),
        "stamp_delivery_buy_bps": float(
            payload_config.get("stamp_delivery_buy_bps", settings.stamp_delivery_buy_bps)
        ),
        "stamp_intraday_buy_bps": float(
            payload_config.get("stamp_intraday_buy_bps", settings.stamp_intraday_buy_bps)
        ),
        "gst_rate": float(payload_config.get("gst_rate", settings.gst_rate)),
        "futures_brokerage_bps": float(
            payload_config.get("futures_brokerage_bps", settings.futures_brokerage_bps)
        ),
        "futures_stt_sell_bps": float(
            payload_config.get("futures_stt_sell_bps", settings.futures_stt_sell_bps)
        ),
        "futures_exchange_txn_bps": float(
            payload_config.get("futures_exchange_txn_bps", settings.futures_exchange_txn_bps)
        ),
        "futures_stamp_buy_bps": float(
            payload_config.get("futures_stamp_buy_bps", settings.futures_stamp_buy_bps)
        ),
    }
    return BacktestConfig(
        risk_per_trade=float(payload_config.get("risk_per_trade", settings.risk_per_trade)),
        max_positions=int(payload_config.get("max_positions", settings.max_positions)),
        initial_equity=float(payload_config.get("initial_equity", 1_000_000.0)),
        commission_bps=float(payload_config.get("commission_bps", settings.commission_bps)),
        slippage_base_bps=float(
            payload_config.get("slippage_base_bps", settings.slippage_base_bps)
        ),
        slippage_vol_factor=float(
            payload_config.get("slippage_vol_factor", settings.slippage_vol_factor)
        ),
        atr_period=int(payload_config.get("atr_period", 14)),
        atr_stop_mult=float(
            payload_config.get("atr_stop_mult", payload_config.get("atr_stop", 2.0))
        ),
        atr_trail_mult=float(
            payload_config.get("atr_trail_mult", payload_config.get("atr_trail", 2.0))
        ),
        take_profit_r=(
            None
            if payload_config.get("take_profit_r") is None
            else float(payload_config["take_profit_r"])
        ),
        time_stop_bars=(
            None
            if payload_config.get("time_stop_bars") is None
            else int(payload_config["time_stop_bars"])
        ),
        min_notional=float(payload_config.get("min_notional", 2_000_000.0)),
        max_position_value_pct_adv=float(
            payload_config.get("max_position_value_pct_adv", settings.max_position_value_pct_adv)
        ),
        adv_lookback=int(payload_config.get("adv_lookback", 20)),
        allow_long=bool(payload_config.get("allow_long", True)),
        cost_model_enabled=bool(
            payload_config.get("cost_model_enabled", settings.cost_model_enabled)
        ),
        cost_mode=str(payload_config.get("cost_mode", settings.cost_mode)),
        cost_params=cost_params,
    )


def _to_iso_records(df: pd.DataFrame, time_col: str) -> list[dict[str, Any]]:
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        out = row.copy()
        if isinstance(out.get(time_col), (pd.Timestamp, datetime)):
            out[time_col] = pd.Timestamp(out[time_col]).isoformat()
        rows.append(out)
    return rows


def execute_backtest(
    session: Session,
    store: DataStore,
    settings: Settings,
    payload: dict[str, Any],
    job_id: str | None = None,
) -> dict[str, Any]:
    symbol = str(payload["symbol"]).upper()
    timeframe = str(payload.get("timeframe", "1d"))
    strategy_template = str(payload["strategy_template"])
    params = dict(payload.get("params", {}))

    start = payload.get("start")
    end = payload.get("end")
    start_dt = pd.Timestamp(start, tz="UTC").to_pydatetime() if start else None
    end_dt = pd.Timestamp(end, tz="UTC").to_pydatetime() if end else None

    if job_id:
        append_job_log(session, job_id, f"Loading OHLCV for {symbol} ({timeframe})")

    frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start_dt, end=end_dt)
    if frame.empty:
        raise APIError(
            code="missing_data", message="No data available for requested symbol/timeframe"
        )

    if job_id:
        append_job_log(session, job_id, f"Generating signals: {strategy_template}")

    signals = generate_signals(strategy_template, frame, params=params)
    bt_config = _build_config(settings, dict(payload.get("config", {}), **params))

    if job_id:
        append_job_log(session, job_id, "Running realistic backtest engine")

    result = run_backtest(price_df=frame, entries=signals, symbol=symbol, config=bt_config)

    if result.equity_curve.empty:
        raise APIError(code="insufficient_data", message="Backtest produced no equity points")

    backtest = Backtest(
        strategy_id=payload.get("strategy_id"),
        symbol=symbol,
        timeframe=timeframe,
        start_date=pd.Timestamp(frame["datetime"].min()).date(),
        end_date=pd.Timestamp(frame["datetime"].max()).date(),
        config_json={
            "strategy_template": strategy_template,
            "params": params,
            "config": bt_config.__dict__,
            "job_id": job_id,
            "equity_curve": _to_iso_records(result.equity_curve, "datetime"),
        },
        metrics_json=result.metrics,
    )
    session.add(backtest)
    session.commit()
    session.refresh(backtest)

    for row in result.trades.to_dict(orient="records"):
        session.add(
            Trade(
                backtest_id=backtest.id,
                symbol=symbol,
                entry_dt=pd.Timestamp(row["entry_dt"]).to_pydatetime(),
                exit_dt=pd.Timestamp(row["exit_dt"]).to_pydatetime(),
                qty=int(row["qty"]),
                entry_px=float(row["entry_px"]),
                exit_px=float(row["exit_px"]),
                pnl=float(row["pnl"]),
                r_multiple=float(row["r_multiple"]),
                reason=str(row["reason"]),
            )
        )
    session.commit()

    if job_id:
        append_job_log(session, job_id, f"Backtest complete. Trades: {len(result.trades)}")

    return {
        "backtest_id": backtest.id,
        "symbol": symbol,
        "timeframe": timeframe,
        "metrics": result.metrics,
        "trade_count": int(len(result.trades)),
    }
