from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from app.engine.costs import estimate_equity_delivery_cost, estimate_futures_cost, estimate_intraday_cost
from app.engine.indicators import atr
from app.engine.metrics import calculate_metrics

ENGINE_VERSION = "atlas-sim-v1.8"
FUTURE_KINDS = {"STOCK_FUT", "INDEX_FUT"}


@dataclass
class SimulationConfig:
    risk_per_trade: float = 0.005
    max_positions: int = 3
    initial_equity: float = 1_000_000.0
    commission_bps: float = 5.0
    slippage_base_bps: float = 2.0
    slippage_vol_factor: float = 15.0
    atr_period: int = 14
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 2.0
    take_profit_r: float | None = None
    time_stop_bars: int | None = None
    min_notional: float = 2_000_000.0
    max_position_value_pct_adv: float = 0.01
    adv_lookback: int = 20
    allow_long: bool = True
    allow_short: bool = False
    instrument_kind: str = "EQUITY_CASH"
    lot_size: int = 1
    futures_initial_margin_pct: float = 0.18
    equity_short_intraday_only: bool = True
    squareoff_time: str = "15:20"
    cost_model_enabled: bool = False
    cost_mode: str = "delivery"
    cost_params: dict[str, float] = field(default_factory=dict)
    seed: int = 7


@dataclass
class SimulationPosition:
    symbol: str
    side: str
    instrument_kind: str
    entry_idx: int
    entry_time: pd.Timestamp
    qty: int
    lot_size: int
    qty_lots: int
    entry_price: float
    stop_price: float
    trail_price: float
    target_price: float | None
    stop_distance: float
    entry_cost: float
    entry_notional: float
    margin_reserved: float
    force_eod: bool
    bars_held: int = 0


@dataclass
class SimulationResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, Any]
    skipped_signals: list[dict[str, Any]]


def _is_futures(instrument_kind: str) -> bool:
    return str(instrument_kind).upper() in FUTURE_KINDS


def _slippage_bps(atr_value: float, close: float, config: SimulationConfig) -> float:
    if close <= 0 or np.isnan(atr_value):
        return config.slippage_base_bps
    return config.slippage_base_bps + config.slippage_vol_factor * (atr_value / close)


def _transaction_cost(
    *,
    notional: float,
    side: str,
    instrument_kind: str,
    config: SimulationConfig,
) -> float:
    if notional <= 0:
        return 0.0
    if config.cost_model_enabled:
        mode = str(config.cost_mode).lower()
        if _is_futures(instrument_kind):
            return estimate_futures_cost(notional=notional, side=side, config=config.cost_params)
        if mode == "intraday":
            return estimate_intraday_cost(notional=notional, side=side, config=config.cost_params)
        return estimate_equity_delivery_cost(notional=notional, side=side, config=config.cost_params)
    return notional * config.commission_bps / 10_000


def _safe_series(entries: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if len(entries) == len(index):
        aligned = pd.Series(entries.to_numpy(), index=index)
    else:
        aligned = entries.reindex(index)
    return pd.Series(aligned, index=index, dtype="boolean").fillna(False).astype(bool)


def _normalize_signals(
    entries: pd.Series | dict[str, pd.Series],
    index: pd.DatetimeIndex,
) -> dict[str, pd.Series]:
    if isinstance(entries, pd.Series):
        return {
            "BUY": _safe_series(entries, index),
            "SELL": pd.Series(False, index=index),
        }

    buy = entries.get("BUY", pd.Series(False, index=index))
    sell = entries.get("SELL", pd.Series(False, index=index))
    return {
        "BUY": _safe_series(pd.Series(buy), index),
        "SELL": _safe_series(pd.Series(sell), index),
    }


def _canonical_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    frame = price_df.copy()
    if frame.empty:
        return frame
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        frame = frame.set_index("datetime")
    frame = frame.sort_index()
    return frame


def _config_hash(config: SimulationConfig) -> str:
    payload = {
        "risk_per_trade": float(config.risk_per_trade),
        "max_positions": int(config.max_positions),
        "initial_equity": float(config.initial_equity),
        "commission_bps": float(config.commission_bps),
        "slippage_base_bps": float(config.slippage_base_bps),
        "slippage_vol_factor": float(config.slippage_vol_factor),
        "atr_period": int(config.atr_period),
        "atr_stop_mult": float(config.atr_stop_mult),
        "atr_trail_mult": float(config.atr_trail_mult),
        "take_profit_r": config.take_profit_r,
        "time_stop_bars": config.time_stop_bars,
        "min_notional": float(config.min_notional),
        "max_position_value_pct_adv": float(config.max_position_value_pct_adv),
        "adv_lookback": int(config.adv_lookback),
        "allow_long": bool(config.allow_long),
        "allow_short": bool(config.allow_short),
        "instrument_kind": str(config.instrument_kind).upper(),
        "lot_size": int(config.lot_size),
        "futures_initial_margin_pct": float(config.futures_initial_margin_pct),
        "equity_short_intraday_only": bool(config.equity_short_intraday_only),
        "squareoff_time": str(config.squareoff_time),
        "cost_model_enabled": bool(config.cost_model_enabled),
        "cost_mode": str(config.cost_mode),
        "cost_params": dict(config.cost_params),
        "seed": int(config.seed),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _data_digest(
    frame: pd.DataFrame,
    signals: dict[str, pd.Series],
    config_hash: str,
) -> str:
    sha = hashlib.sha256()
    ts = pd.to_datetime(frame.index, utc=True).astype("int64").to_numpy()
    sha.update(ts.tobytes())
    for col in ("open", "high", "low", "close", "volume"):
        values = np.nan_to_num(frame[col].to_numpy(dtype=float), nan=0.0)
        sha.update(values.tobytes())
    sha.update(signals["BUY"].to_numpy(dtype=np.int8).tobytes())
    sha.update(signals["SELL"].to_numpy(dtype=np.int8).tobytes())
    sha.update(config_hash.encode("utf-8"))
    return sha.hexdigest()


def _side_label(side: str) -> str:
    return "LONG" if side == "BUY" else "SHORT"


def _exit_side_for_cost(side_label: str) -> str:
    return "SELL" if side_label == "LONG" else "BUY"


def _entry_side_for_cost(side_label: str) -> str:
    return "BUY" if side_label == "LONG" else "SELL"


def _gross_unrealized(position: SimulationPosition, mark_price: float) -> float:
    if position.side == "LONG":
        return (mark_price - position.entry_price) * position.qty
    return (position.entry_price - mark_price) * position.qty


def _mark_to_market(position: SimulationPosition, mark_price: float) -> float:
    if position.side == "LONG" and not _is_futures(position.instrument_kind):
        return mark_price * position.qty
    return position.margin_reserved + _gross_unrealized(position, mark_price)


def _target_hit(position: SimulationPosition, high: float, low: float) -> bool:
    if position.target_price is None:
        return False
    if position.side == "LONG":
        return high >= position.target_price
    return low <= position.target_price


def _stop_hit(position: SimulationPosition, high: float, low: float, trigger: float) -> bool:
    if position.side == "LONG":
        return low <= trigger
    return high >= trigger


def _trailing_candidate(position: SimulationPosition, close: float, atr_value: float, config: SimulationConfig) -> float:
    if position.side == "LONG":
        return close - config.atr_trail_mult * atr_value
    return close + config.atr_trail_mult * atr_value


def _updated_trail(position: SimulationPosition, candidate: float) -> float:
    if np.isnan(candidate):
        return position.trail_price
    if position.side == "LONG":
        return max(position.trail_price, candidate)
    return min(position.trail_price, candidate)


def _stop_trigger(position: SimulationPosition) -> float:
    if position.side == "LONG":
        return max(position.stop_price, position.trail_price)
    return min(position.stop_price, position.trail_price)


def _exit_price(
    *,
    position: SimulationPosition,
    bar: pd.Series,
    atr_value: float,
    reason: str,
    trigger: float | None,
    config: SimulationConfig,
) -> float:
    close = float(bar["close"])
    open_px = float(bar["open"])
    slippage = _slippage_bps(atr_value, close, config) / 10_000
    if reason == "STOP_HIT":
        ref = max(0.0, float(trigger if trigger is not None else position.trail_price))
        if position.side == "LONG":
            stop_ref = min(open_px, ref)
            return max(0.0, stop_ref) * (1 - slippage)
        stop_ref = max(open_px, ref)
        return max(0.0, stop_ref) * (1 + slippage)
    if reason == "TARGET_HIT" and position.target_price is not None:
        if position.side == "LONG":
            return max(0.0, position.target_price) * (1 - slippage)
        return max(0.0, position.target_price) * (1 + slippage)
    if position.side == "LONG":
        return max(0.0, close) * (1 - slippage)
    return max(0.0, close) * (1 + slippage)


def _exit_position(
    *,
    position: SimulationPosition,
    bar: pd.Series,
    timestamp: pd.Timestamp,
    atr_value: float,
    reason: str,
    trigger: float | None,
    config: SimulationConfig,
) -> tuple[float, dict[str, Any]]:
    px = _exit_price(
        position=position,
        bar=bar,
        atr_value=atr_value,
        reason=reason,
        trigger=trigger,
        config=config,
    )
    notional = position.qty * px
    exit_cost = _transaction_cost(
        notional=notional,
        side=_exit_side_for_cost(position.side),
        instrument_kind=position.instrument_kind,
        config=config,
    )
    gross_pnl = (
        (px - position.entry_price) * position.qty
        if position.side == "LONG"
        else (position.entry_price - px) * position.qty
    )
    pnl = gross_pnl - position.entry_cost - exit_cost

    if position.side == "LONG" and not _is_futures(position.instrument_kind):
        cash_delta = notional - exit_cost
    else:
        cash_delta = position.margin_reserved + gross_pnl - exit_cost

    trade = {
        "symbol": position.symbol,
        "side": position.side,
        "instrument_kind": position.instrument_kind,
        "lot_size": position.lot_size,
        "qty_lots": position.qty_lots,
        "entry_dt": position.entry_time,
        "exit_dt": timestamp,
        "qty": position.qty,
        "entry_px": position.entry_price,
        "exit_px": px,
        "entry_cost": position.entry_cost,
        "exit_cost": exit_cost,
        "gross_pnl": gross_pnl,
        "pnl": pnl,
        "r_multiple": pnl / (position.stop_distance * position.qty) if position.stop_distance > 0 else 0.0,
        "reason": reason,
        "holding_bars": position.bars_held,
        "notional": position.entry_notional,
        "metadata": {
            "force_eod": bool(position.force_eod),
            "margin_reserved": float(position.margin_reserved),
        },
    }
    return cash_delta, trade


def _squareoff_cutoff(value: str) -> time:
    try:
        hour, minute = value.strip().split(":", 1)
        return time(hour=int(hour), minute=int(minute))
    except Exception:  # noqa: BLE001
        return time(hour=15, minute=20)


def run_simulation(
    *,
    price_df: pd.DataFrame,
    entries: pd.Series | dict[str, pd.Series],
    symbol: str,
    config: SimulationConfig,
) -> SimulationResult:
    if price_df.empty:
        metadata = {
            "engine_version": ENGINE_VERSION,
            "seed": int(config.seed),
            "data_digest": "",
            "config_digest": _config_hash(config),
        }
        return SimulationResult(
            trades=pd.DataFrame(),
            equity_curve=pd.DataFrame(),
            metrics={},
            metadata=metadata,
            skipped_signals=[],
        )

    frame = _canonical_frame(price_df)
    signals = _normalize_signals(entries, frame.index)
    atr_series = atr(frame, config.atr_period)
    adv_notional = (frame["close"] * frame["volume"]).rolling(config.adv_lookback, min_periods=1).mean()

    config_digest = _config_hash(config)
    data_digest = _data_digest(frame, signals, config_digest)
    metadata = {
        "engine_version": ENGINE_VERSION,
        "seed": int(config.seed),
        "data_digest": data_digest,
        "config_digest": config_digest,
    }

    cash = float(config.initial_equity)
    positions: list[SimulationPosition] = []
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    open_counts: list[int] = []
    skipped_signals: list[dict[str, Any]] = []
    cutoff = _squareoff_cutoff(config.squareoff_time)
    instrument_kind = str(config.instrument_kind).upper()
    lot_size = max(1, int(config.lot_size))

    for i in range(1, len(frame)):
        bar = frame.iloc[i]
        timestamp = frame.index[i]
        atr_now = float(atr_series.iloc[i])

        survivors: list[SimulationPosition] = []
        for position in positions:
            position.bars_held += 1
            trail_candidate = _trailing_candidate(position, float(bar["close"]), atr_now, config)
            position.trail_price = _updated_trail(position, trail_candidate)

            reason: str | None = None
            trigger = _stop_trigger(position)
            if _stop_hit(position, float(bar["high"]), float(bar["low"]), trigger):
                reason = "STOP_HIT"
            elif _target_hit(position, float(bar["high"]), float(bar["low"])):
                reason = "TARGET_HIT"
            elif config.time_stop_bars is not None and position.bars_held >= config.time_stop_bars:
                reason = "TIME_STOP"

            if reason is None:
                survivors.append(position)
                continue

            cash_delta, trade = _exit_position(
                position=position,
                bar=bar,
                timestamp=timestamp,
                atr_value=atr_now,
                reason=reason,
                trigger=trigger if reason == "STOP_HIT" else None,
                config=config,
            )
            cash += cash_delta
            trades.append(trade)

        positions = survivors

        signal_idx = i - 1
        signal_bar = frame.iloc[signal_idx]
        liquidity_ok = (float(signal_bar["close"]) * float(signal_bar["volume"])) >= config.min_notional

        # Deterministic order for same-bar opposite signals.
        for side in ("BUY", "SELL"):
            if side == "BUY" and not config.allow_long:
                continue
            if side == "SELL" and not config.allow_short:
                continue
            if not bool(signals[side].iloc[signal_idx]):
                continue
            if len(positions) >= config.max_positions:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "reason": "max_positions_reached",
                        "timestamp": str(timestamp),
                    }
                )
                continue
            if not liquidity_ok:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "reason": "liquidity_filter",
                        "timestamp": str(timestamp),
                    }
                )
                continue

            stop_distance = config.atr_stop_mult * float(atr_series.iloc[signal_idx])
            if stop_distance <= 0 or np.isnan(stop_distance):
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "reason": "invalid_stop_distance",
                        "timestamp": str(timestamp),
                    }
                )
                continue

            risk_amount = cash * config.risk_per_trade
            qty_risk = int(np.floor(risk_amount / stop_distance))
            if qty_risk <= 0:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "reason": "risk_budget_too_small",
                        "timestamp": str(timestamp),
                    }
                )
                continue

            slip = _slippage_bps(float(atr_series.iloc[signal_idx]), float(signal_bar["close"]), config) / 10_000
            open_px = float(bar["open"])
            if open_px <= 0:
                skipped_signals.append(
                    {"symbol": symbol, "side": side, "reason": "invalid_open_price", "timestamp": str(timestamp)}
                )
                continue
            entry_price = open_px * (1 + slip) if side == "BUY" else open_px * (1 - slip)
            if entry_price <= 0:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "reason": "invalid_entry_price",
                        "timestamp": str(timestamp),
                    }
                )
                continue

            signal_adv = float(adv_notional.iloc[signal_idx])
            qty_adv = qty_risk
            if config.max_position_value_pct_adv > 0 and signal_adv > 0:
                max_position_value = signal_adv * config.max_position_value_pct_adv
                qty_adv = int(np.floor(max_position_value / entry_price))
            qty = min(qty_risk, qty_adv)

            qty_lots = 1
            if _is_futures(instrument_kind):
                qty_lots = qty // lot_size
                qty = qty_lots * lot_size
            if qty <= 0:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "reason": "adv_cap_too_small",
                        "timestamp": str(timestamp),
                    }
                )
                continue
            if _is_futures(instrument_kind) and qty_lots <= 0:
                skipped_signals.append(
                    {"symbol": symbol, "side": side, "reason": "lot_rounding_zero", "timestamp": str(timestamp)}
                )
                continue

            notional = qty * entry_price
            entry_cost = _transaction_cost(
                notional=notional,
                side=_entry_side_for_cost(_side_label(side)),
                instrument_kind=instrument_kind,
                config=config,
            )

            margin_reserved = 0.0
            if _is_futures(instrument_kind):
                margin_reserved = notional * max(0.0, config.futures_initial_margin_pct)
                required = margin_reserved + entry_cost
            elif side == "SELL":
                margin_reserved = notional
                required = margin_reserved + entry_cost
            else:
                required = notional + entry_cost

            if cash < required:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "side": side,
                        "reason": "insufficient_margin"
                        if (_is_futures(instrument_kind) or side == "SELL")
                        else "insufficient_cash",
                        "timestamp": str(timestamp),
                    }
                )
                continue

            cash -= required
            side_label = _side_label(side)
            target_price: float | None = None
            if config.take_profit_r is not None:
                if side == "BUY":
                    target_price = entry_price + config.take_profit_r * stop_distance
                else:
                    target_price = max(0.0, entry_price - config.take_profit_r * stop_distance)
            force_eod = bool(
                side == "SELL"
                and instrument_kind == "EQUITY_CASH"
                and config.equity_short_intraday_only
            )
            start_stop = entry_price - stop_distance if side == "BUY" else entry_price + stop_distance
            positions.append(
                SimulationPosition(
                    symbol=symbol,
                    side=side_label,
                    instrument_kind=instrument_kind,
                    entry_idx=i,
                    entry_time=timestamp,
                    qty=qty,
                    lot_size=lot_size if _is_futures(instrument_kind) else 1,
                    qty_lots=qty_lots,
                    entry_price=entry_price,
                    stop_price=start_stop,
                    trail_price=start_stop,
                    target_price=target_price,
                    stop_distance=stop_distance,
                    entry_cost=entry_cost,
                    entry_notional=notional,
                    margin_reserved=margin_reserved,
                    force_eod=force_eod,
                )
            )

        # Intraday cash shorts cannot carry overnight.
        next_ts = frame.index[i + 1] if i + 1 < len(frame) else None
        end_of_day = next_ts is None or next_ts.date() != timestamp.date() or timestamp.time() >= cutoff
        if end_of_day:
            carry: list[SimulationPosition] = []
            for position in positions:
                if not position.force_eod:
                    carry.append(position)
                    continue
                cash_delta, trade = _exit_position(
                    position=position,
                    bar=bar,
                    timestamp=timestamp,
                    atr_value=atr_now,
                    reason="EOD_SQUARE_OFF",
                    trigger=None,
                    config=config,
                )
                cash += cash_delta
                trades.append(trade)
            positions = carry

        mark_to_market = sum(_mark_to_market(position, float(bar["close"])) for position in positions)
        equity_rows.append({"datetime": timestamp, "equity": cash + mark_to_market})
        open_counts.append(len(positions))

    if equity_rows and positions:
        final_bar = frame.iloc[-1]
        final_ts = frame.index[-1]
        atr_final = float(atr_series.iloc[-1])
        for position in positions:
            cash_delta, trade = _exit_position(
                position=position,
                bar=final_bar,
                timestamp=final_ts,
                atr_value=atr_final,
                reason="EOD_CLOSE",
                trigger=None,
                config=config,
            )
            cash += cash_delta
            trades.append(trade)
        equity_rows[-1]["equity"] = cash

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    equity_series = equity_df.set_index("datetime")["equity"] if not equity_df.empty else pd.Series(dtype=float)
    open_count_series = pd.Series(open_counts, dtype=float)
    metrics = calculate_metrics(
        equity=equity_series,
        trades=trades_df
        if not trades_df.empty
        else pd.DataFrame(columns=["pnl", "notional", "holding_bars"]),
        open_position_count=open_count_series,
    )
    return SimulationResult(
        trades=trades_df,
        equity_curve=equity_df,
        metrics=metrics,
        metadata=metadata,
        skipped_signals=skipped_signals,
    )
