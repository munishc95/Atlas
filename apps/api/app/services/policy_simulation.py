from __future__ import annotations

from datetime import date, datetime, time, timezone
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import Policy
from app.engine.backtester import BacktestConfig, run_backtest
from app.services.data_store import DataStore
from app.strategies.templates import generate_signals


def _utc_datetime(value: date, *, end: bool = False) -> datetime:
    if end:
        return datetime.combine(value, time.max, tzinfo=timezone.utc)
    return datetime.combine(value, time.min, tzinfo=timezone.utc)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _downsample_series(frame: pd.DataFrame, *, limit: int = 1500) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    if len(frame) <= limit:
        sampled = frame
    else:
        step = max(1, len(frame) // limit)
        sampled = frame.iloc[::step].copy()
        if sampled.iloc[-1]["datetime"] != frame.iloc[-1]["datetime"]:
            sampled = pd.concat([sampled, frame.tail(1)], ignore_index=True)
    rows: list[dict[str, Any]] = []
    for row in sampled.to_dict(orient="records"):
        dt = pd.Timestamp(row["datetime"])
        rows.append({"datetime": dt.isoformat(), "equity": float(row["equity"])})
    return rows


def _resolve_policy_setup(
    policy: Policy,
    *,
    regime: str | None,
    fallback_timeframe: str = "1d",
) -> tuple[str, dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    regime_map = definition.get("regime_map", {})
    if not isinstance(regime_map, dict) or not regime_map:
        raise APIError(
            code="invalid_policy",
            message=f"Policy {policy.id} has no regime_map for simulation.",
        )

    selected_regime = regime or "TREND_UP"
    regime_cfg = regime_map.get(selected_regime)
    if not isinstance(regime_cfg, dict):
        # Fallback to first valid regime map entry.
        for key, value in regime_map.items():
            if isinstance(value, dict):
                selected_regime = str(key)
                regime_cfg = value
                break
    if not isinstance(regime_cfg, dict):
        raise APIError(
            code="invalid_policy",
            message=f"Policy {policy.id} has no usable regime config.",
        )

    strategy_key = regime_cfg.get("strategy_key")
    allowed = regime_cfg.get("allowed_templates")
    if not isinstance(strategy_key, str) or not strategy_key:
        if isinstance(allowed, list) and allowed:
            strategy_key = str(allowed[0])
        else:
            strategy_key = "trend_breakout"

    raw_params = regime_cfg.get("params", {})
    params: dict[str, Any]
    if isinstance(raw_params, dict) and isinstance(raw_params.get(strategy_key), dict):
        params = dict(raw_params.get(strategy_key, {}))
    elif isinstance(raw_params, dict):
        params = dict(raw_params)
    else:
        params = {}

    timeframes = definition.get("timeframes", [])
    if isinstance(timeframes, list) and timeframes:
        timeframe = str(regime_cfg.get("timeframe", timeframes[0]))
    else:
        timeframe = str(regime_cfg.get("timeframe", fallback_timeframe))
    if not timeframe:
        timeframe = fallback_timeframe

    return str(strategy_key), params, timeframe, definition, regime_cfg


def _build_bt_config(
    settings: Settings,
    *,
    params: dict[str, Any],
    regime_cfg: dict[str, Any],
    definition: dict[str, Any],
) -> BacktestConfig:
    risk_scale = _safe_float(regime_cfg.get("risk_scale"), 1.0)
    max_positions_scale = _safe_float(regime_cfg.get("max_positions_scale"), 1.0)
    cost_model = definition.get("cost_model", {})
    cost_model = cost_model if isinstance(cost_model, dict) else {}
    cost_params = {
        "brokerage_bps": settings.brokerage_bps,
        "stt_delivery_buy_bps": settings.stt_delivery_buy_bps,
        "stt_delivery_sell_bps": settings.stt_delivery_sell_bps,
        "stt_intraday_buy_bps": settings.stt_intraday_buy_bps,
        "stt_intraday_sell_bps": settings.stt_intraday_sell_bps,
        "exchange_txn_bps": settings.exchange_txn_bps,
        "sebi_bps": settings.sebi_bps,
        "stamp_delivery_buy_bps": settings.stamp_delivery_buy_bps,
        "stamp_intraday_buy_bps": settings.stamp_intraday_buy_bps,
        "gst_rate": settings.gst_rate,
        "futures_brokerage_bps": settings.futures_brokerage_bps,
        "futures_stt_sell_bps": settings.futures_stt_sell_bps,
        "futures_exchange_txn_bps": settings.futures_exchange_txn_bps,
        "futures_stamp_buy_bps": settings.futures_stamp_buy_bps,
    }
    return BacktestConfig(
        risk_per_trade=max(0.0001, settings.risk_per_trade * max(0.0, risk_scale)),
        max_positions=max(1, int(round(settings.max_positions * max(0.1, max_positions_scale)))),
        initial_equity=1_000_000.0,
        commission_bps=settings.commission_bps,
        slippage_base_bps=settings.slippage_base_bps,
        slippage_vol_factor=settings.slippage_vol_factor,
        atr_period=int(params.get("atr_period", 14)),
        atr_stop_mult=_safe_float(params.get("atr_stop_mult", params.get("atr_stop", 2.0)), 2.0),
        atr_trail_mult=_safe_float(params.get("atr_trail_mult", params.get("atr_trail", 2.0)), 2.0),
        take_profit_r=(
            _safe_float(params.get("take_profit_r"), 0.0)
            if params.get("take_profit_r") is not None
            else None
        ),
        time_stop_bars=(
            int(params["time_stop_bars"]) if params.get("time_stop_bars") is not None else None
        ),
        min_notional=_safe_float(params.get("min_notional", 2_000_000.0), 2_000_000.0),
        max_position_value_pct_adv=_safe_float(
            params.get("max_position_value_pct_adv", settings.max_position_value_pct_adv),
            settings.max_position_value_pct_adv,
        ),
        adv_lookback=int(params.get("adv_lookback", 20)),
        allow_long=bool(params.get("allow_long", True)),
        cost_model_enabled=bool(cost_model.get("enabled", settings.cost_model_enabled)),
        cost_mode=str(cost_model.get("mode", settings.cost_mode)),
        cost_params=cost_params,
    )


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def _cvar_95(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    threshold = returns.quantile(0.05)
    tail = returns[returns <= threshold]
    if tail.empty:
        return float(threshold)
    return float(tail.mean())


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def simulate_policy_on_bundle(
    *,
    session: Session,
    store: DataStore,
    settings: Settings,
    policy: Policy,
    bundle_id: int,
    start_date: date,
    end_date: date,
    regime: str | None = None,
    seed: int = 7,
    max_symbols: int | None = None,
) -> dict[str, Any]:
    strategy_key, params, timeframe, definition, regime_cfg = _resolve_policy_setup(
        policy,
        regime=regime,
    )
    universe = definition.get("universe", {})
    universe = universe if isinstance(universe, dict) else {}
    symbol_scope = str(universe.get("symbol_scope", "liquid"))
    max_scan = int(universe.get("max_symbols_scan", 50))
    if max_symbols is not None:
        max_scan = min(max_scan, max_symbols) if max_scan > 0 else max_symbols
    max_scan = max(1, max_scan)

    symbols = store.sample_bundle_symbols(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe,
        symbol_scope=symbol_scope,
        max_symbols_scan=max_scan,
        seed=seed,
    )
    if not symbols:
        raise APIError(
            code="missing_data",
            message=f"No symbols available in bundle {bundle_id} for timeframe {timeframe}.",
        )

    config = _build_bt_config(
        settings,
        params=params,
        regime_cfg=regime_cfg,
        definition=definition,
    )
    start_dt = _utc_datetime(start_date, end=False)
    end_dt = _utc_datetime(end_date, end=True)

    symbol_rows: list[dict[str, Any]] = []
    normalized_curves: list[pd.Series] = []
    for symbol in sorted(symbols):
        frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe, start=start_dt, end=end_dt)
        if len(frame) < 60:
            continue
        signals = generate_signals(strategy_key, frame, params=params)
        result = run_backtest(price_df=frame, entries=signals, symbol=symbol, config=config)
        if result.equity_curve.empty:
            continue

        eq = result.equity_curve.copy()
        eq["datetime"] = pd.to_datetime(eq["datetime"], utc=True)
        eq = eq.sort_values("datetime")
        base = float(eq["equity"].iloc[0])
        if base <= 0:
            continue
        norm = eq.set_index("datetime")["equity"] / base
        normalized_curves.append(norm)

        metrics = result.metrics if isinstance(result.metrics, dict) else {}
        symbol_rows.append(
            {
                "symbol": symbol,
                "trade_count": int(len(result.trades)),
                "period_return": float(eq["equity"].iloc[-1] / base - 1.0),
                "max_drawdown": _safe_float(metrics.get("max_drawdown"), 0.0),
                "calmar": _safe_float(metrics.get("calmar"), 0.0),
                "cvar_95": _safe_float(metrics.get("cvar_95"), 0.0),
                "win_rate": _safe_float(metrics.get("win_rate"), 0.0),
                "profit_factor": _safe_float(metrics.get("profit_factor"), 0.0),
                "turnover": _safe_float(metrics.get("turnover"), 0.0),
                "exposure_pct": _safe_float(metrics.get("exposure_pct"), 0.0),
            }
        )

    if not symbol_rows or not normalized_curves:
        summary = {
            "policy_id": policy.id,
            "policy_name": policy.name,
            "strategy_key": strategy_key,
            "timeframe": timeframe,
            "regime": regime,
            "symbol_count": 0,
            "symbols": [],
            "metrics": {
                "period_return": 0.0,
                "max_drawdown": 0.0,
                "calmar": 0.0,
                "cvar_95": 0.0,
                "turnover": 0.0,
                "cost_ratio": 0.0,
                "score": -1.0,
            },
            "equity_curve": [],
            "reasons": ["No symbols produced valid shadow simulation output."],
        }
        summary["digest"] = _stable_hash(summary)
        return summary

    merged = pd.concat(normalized_curves, axis=1).sort_index().ffill().dropna(how="all")
    portfolio_norm = merged.mean(axis=1)
    portfolio_eq = pd.Series(portfolio_norm * 1_000_000.0, index=portfolio_norm.index)
    returns = portfolio_eq.pct_change().dropna()
    max_dd = _max_drawdown(portfolio_eq)
    period_return = float(portfolio_eq.iloc[-1] / portfolio_eq.iloc[0] - 1.0)
    calmar = period_return / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
    cvar_95 = _cvar_95(returns)
    turnover = float(np.mean([_safe_float(row["turnover"]) for row in symbol_rows]))
    win_rate = float(np.mean([_safe_float(row["win_rate"]) for row in symbol_rows]))
    profit_factor = float(np.mean([_safe_float(row["profit_factor"]) for row in symbol_rows]))
    exposure_pct = float(np.mean([_safe_float(row["exposure_pct"]) for row in symbol_rows]))
    cost_ratio = max(0.0, turnover * (settings.commission_bps / 10_000))
    score = float(
        (1.1 * calmar)
        + (2.0 * period_return)
        - (1.5 * abs(max_dd))
        - (3.0 * abs(cvar_95))
        - (0.6 * cost_ratio)
        - (0.05 * turnover)
    )

    equity_frame = pd.DataFrame(
        {"datetime": portfolio_eq.index.to_pydatetime(), "equity": portfolio_eq.to_numpy()}
    )
    summary = {
        "policy_id": policy.id,
        "policy_name": policy.name,
        "strategy_key": strategy_key,
        "timeframe": timeframe,
        "regime": regime,
        "symbol_count": len(symbol_rows),
        "symbols": [row["symbol"] for row in symbol_rows],
        "symbol_rows": symbol_rows,
        "metrics": {
            "period_return": period_return,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "cvar_95": cvar_95,
            "turnover": turnover,
            "cost_ratio": cost_ratio,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "exposure_pct": exposure_pct,
            "score": score,
        },
        "equity_curve": _downsample_series(equity_frame),
        "engine_inputs": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "seed": int(seed),
            "bundle_id": int(bundle_id),
            "symbol_scope": symbol_scope,
            "max_symbols_scan": max_scan,
            "strategy_params": params,
        },
    }
    summary["digest"] = _stable_hash(summary)
    return summary
