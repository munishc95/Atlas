from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, time
import hashlib
import json
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app.engine.costs import estimate_equity_delivery_cost, estimate_futures_cost, estimate_intraday_cost
from app.engine.indicators import atr
from app.engine.metrics import calculate_metrics

ENGINE_VERSION = "atlas-sim-v1.9"
FUTURE_KINDS = {"STOCK_FUT", "INDEX_FUT"}
IST_ZONE = ZoneInfo("Asia/Kolkata")


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
    risk_overlay_enabled: bool = False
    risk_overlay_scale: float = 1.0
    risk_overlay_realized_vol: float = 0.0
    risk_overlay_target_vol: float = 0.0
    risk_overlay_max_gross_exposure_pct: float = 1.0
    risk_overlay_max_single_name_exposure_pct: float = 0.12
    risk_overlay_max_sector_exposure_pct: float = 0.30
    risk_overlay_corr_clamp_enabled: bool = False
    risk_overlay_corr_threshold: float = 0.65
    risk_overlay_corr_reduce_factor: float = 0.5
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
    metadata_json: dict[str, Any] = field(default_factory=dict)
    source_position_id: int | None = None
    bars_held: int = 0


@dataclass
class SimulationResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]
    metadata: dict[str, Any]
    skipped_signals: list[dict[str, Any]]


@dataclass
class PortfolioStepResult:
    positions: list[dict[str, Any]]
    orders: list[dict[str, Any]]
    trades: list[dict[str, Any]]
    executed_signals: list[dict[str, Any]]
    skipped_signals: list[dict[str, Any]]
    cash: float
    equity: float
    entry_cost_total: float
    exit_cost_total: float
    entry_slippage_cost: float
    exit_slippage_cost: float
    traded_notional_total: float
    metadata: dict[str, Any]


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
        "risk_overlay_enabled": bool(config.risk_overlay_enabled),
        "risk_overlay_scale": float(config.risk_overlay_scale),
        "risk_overlay_realized_vol": float(config.risk_overlay_realized_vol),
        "risk_overlay_target_vol": float(config.risk_overlay_target_vol),
        "risk_overlay_max_gross_exposure_pct": float(config.risk_overlay_max_gross_exposure_pct),
        "risk_overlay_max_single_name_exposure_pct": float(
            config.risk_overlay_max_single_name_exposure_pct
        ),
        "risk_overlay_max_sector_exposure_pct": float(config.risk_overlay_max_sector_exposure_pct),
        "risk_overlay_corr_clamp_enabled": bool(config.risk_overlay_corr_clamp_enabled),
        "risk_overlay_corr_threshold": float(config.risk_overlay_corr_threshold),
        "risk_overlay_corr_reduce_factor": float(config.risk_overlay_corr_reduce_factor),
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


def _is_squareoff_due(asof: datetime, cutoff: time) -> bool:
    return asof.astimezone(IST_ZONE).time() >= cutoff


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


def _step_signal_sort_key(signal: dict[str, Any], seed: int) -> tuple[Any, ...]:
    symbol = str(signal.get("symbol", "")).upper()
    side = str(signal.get("side", "BUY")).upper()
    template = str(signal.get("template", ""))
    signal_strength = float(signal.get("signal_strength", 0.0))
    tie = hashlib.sha1(f"{seed}:{symbol}:{side}:{template}".encode("utf-8")).hexdigest()
    return (-signal_strength, symbol, side, template, tie)


def _step_slippage_bps(vol_scale: float, config: SimulationConfig) -> float:
    return config.slippage_base_bps + (config.slippage_vol_factor * max(0.0, float(vol_scale)))


def _to_internal_position(position: dict[str, Any]) -> SimulationPosition:
    side = str(position.get("side", "BUY")).upper()
    side_label = _side_label(side)
    qty = max(0, int(position.get("qty", 0)))
    avg_price = max(0.0, float(position.get("avg_price", 0.0)))
    stop_price_raw = position.get("stop_price")
    stop_price = float(stop_price_raw) if isinstance(stop_price_raw, (int, float)) else avg_price
    target_raw = position.get("target_price")
    target_price = float(target_raw) if isinstance(target_raw, (int, float)) else None
    lot_size = max(1, int(position.get("lot_size", 1)))
    qty_lots = max(1, int(position.get("qty_lots", max(1, qty // max(1, lot_size)))))
    stop_distance = abs(avg_price - stop_price)
    metadata = position.get("metadata_json", {})
    if not isinstance(metadata, dict):
        metadata = {}
    source_id = position.get("id")
    return SimulationPosition(
        symbol=str(position.get("symbol", "")).upper(),
        side=side_label,
        instrument_kind=str(position.get("instrument_kind", "EQUITY_CASH")).upper(),
        entry_idx=0,
        entry_time=pd.Timestamp(position.get("opened_at") or pd.Timestamp.utcnow()),
        qty=qty,
        lot_size=lot_size,
        qty_lots=qty_lots,
        entry_price=avg_price,
        stop_price=stop_price,
        trail_price=stop_price,
        target_price=target_price,
        stop_distance=stop_distance,
        entry_cost=0.0,
        entry_notional=qty * avg_price,
        margin_reserved=float(position.get("margin_reserved", 0.0) or 0.0),
        force_eod=bool(position.get("must_exit_by_eod", False)),
        metadata_json=metadata,
        source_position_id=int(source_id) if isinstance(source_id, int) else None,
        bars_held=0,
    )


def _from_internal_position(position: SimulationPosition, *, source_id: int | None) -> dict[str, Any]:
    side = "BUY" if position.side == "LONG" else "SELL"
    return {
        "source_position_id": source_id,
        "symbol": position.symbol,
        "side": side,
        "instrument_kind": position.instrument_kind,
        "lot_size": int(position.lot_size),
        "qty_lots": int(position.qty_lots),
        "margin_reserved": float(position.margin_reserved),
        "must_exit_by_eod": bool(position.force_eod),
        "qty": int(position.qty),
        "avg_price": float(position.entry_price),
        "stop_price": float(position.stop_price) if position.stop_price is not None else None,
        "target_price": float(position.target_price) if position.target_price is not None else None,
        "opened_at": position.entry_time.to_pydatetime().isoformat(),
        "metadata_json": dict(position.metadata_json),
    }


def _step_cost_config(config: SimulationConfig, *, intraday: bool) -> SimulationConfig:
    if not intraday:
        return config
    if str(config.cost_mode).lower() == "intraday":
        return config
    return replace(config, cost_mode="intraday")


def _step_data_digest(
    *,
    signals: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    mark_prices: dict[str, float],
    cash: float,
    equity_reference: float,
    asof: pd.Timestamp,
    config: SimulationConfig,
) -> str:
    payload = {
        "signals": signals,
        "positions": positions,
        "mark_prices": {str(key): float(value) for key, value in sorted(mark_prices.items())},
        "cash": float(cash),
        "equity_reference": float(equity_reference),
        "asof": asof.isoformat(),
        "config_digest": _config_hash(config),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _signal_underlying(signal: dict[str, Any], symbol: str) -> str:
    token = str(signal.get("underlying_symbol", symbol)).strip().upper()
    return token or symbol


def _signal_sector(signal: dict[str, Any]) -> str:
    raw = signal.get("sector", "UNKNOWN")
    token = str(raw).strip().upper()
    return token or "UNKNOWN"


def _position_underlying(position: SimulationPosition) -> str:
    if isinstance(position.metadata_json, dict):
        token = str(position.metadata_json.get("underlying_symbol", position.symbol)).strip().upper()
        if token:
            return token
    return position.symbol


def _position_sector(position: SimulationPosition) -> str:
    if isinstance(position.metadata_json, dict):
        token = str(position.metadata_json.get("sector", "UNKNOWN")).strip().upper()
        if token:
            return token
    return "UNKNOWN"


def _position_exposure_notional(
    position: SimulationPosition,
    mark_prices: dict[str, float],
) -> float:
    mark = float(mark_prices.get(position.symbol, position.entry_price))
    return abs(mark * float(position.qty))


def simulate_portfolio_step(
    *,
    signals: list[dict[str, Any]],
    open_positions: list[dict[str, Any]],
    mark_prices: dict[str, float],
    asof: pd.Timestamp,
    cash: float,
    equity_reference: float,
    config: SimulationConfig,
) -> PortfolioStepResult:
    ordered_signals = sorted([dict(item) for item in signals], key=lambda row: _step_signal_sort_key(row, config.seed))
    positions: list[SimulationPosition] = []
    normalized_positions: list[dict[str, Any]] = []
    for row in open_positions:
        pos = _to_internal_position(row)
        positions.append(pos)
        normalized_positions.append(
            {
                "id": row.get("id"),
                "symbol": pos.symbol,
                "side": "BUY" if pos.side == "LONG" else "SELL",
                "instrument_kind": pos.instrument_kind,
                "qty": pos.qty,
                "avg_price": pos.entry_price,
                "stop_price": pos.stop_price,
                "target_price": pos.target_price,
                "lot_size": pos.lot_size,
                "qty_lots": pos.qty_lots,
                "margin_reserved": pos.margin_reserved,
                "must_exit_by_eod": pos.force_eod,
            }
        )

    entry_cost_total = 0.0
    exit_cost_total = 0.0
    entry_slippage_cost_total = 0.0
    exit_slippage_cost_total = 0.0
    traded_notional_total = 0.0
    skipped: list[dict[str, Any]] = []
    executed_signals: list[dict[str, Any]] = []
    orders: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    overlay_enabled = bool(config.risk_overlay_enabled)
    overlay_scale = (
        max(0.0, float(config.risk_overlay_scale)) if overlay_enabled else 1.0
    )
    gross_cap_notional = max(0.0, float(config.risk_overlay_max_gross_exposure_pct)) * max(
        0.0, float(equity_reference)
    )
    single_name_cap_notional = max(
        0.0, float(config.risk_overlay_max_single_name_exposure_pct)
    ) * max(0.0, float(equity_reference))
    sector_cap_notional = max(0.0, float(config.risk_overlay_max_sector_exposure_pct)) * max(
        0.0, float(equity_reference)
    )
    gross_exposure_notional = 0.0
    single_name_exposure_notional: dict[str, float] = {}
    sector_exposure_notional: dict[str, float] = {}
    for position in positions:
        position_notional = _position_exposure_notional(position, mark_prices)
        gross_exposure_notional += position_notional
        underlying = _position_underlying(position)
        sector = _position_sector(position)
        single_name_exposure_notional[underlying] = (
            single_name_exposure_notional.get(underlying, 0.0) + position_notional
        )
        sector_exposure_notional[sector] = (
            sector_exposure_notional.get(sector, 0.0) + position_notional
        )

    for signal in ordered_signals:
        side = str(signal.get("side", "BUY")).upper()
        symbol = str(signal.get("symbol", "")).upper()
        underlying_symbol = _signal_underlying(signal, symbol)
        sector = _signal_sector(signal)
        instrument_kind = str(signal.get("instrument_kind", "EQUITY_CASH")).upper()
        lot_size = max(1, int(signal.get("lot_size", 1)))
        stop_distance = float(signal.get("stop_distance", 0.0))
        base_price = float(signal.get("price", 0.0))
        vol_scale = float(signal.get("vol_scale", 0.0))
        if side not in {"BUY", "SELL"}:
            skipped.append({**signal, "reason": "invalid_side"})
            continue
        if base_price <= 0 or stop_distance <= 0:
            skipped.append({**signal, "reason": "invalid_price_or_size"})
            continue
        if len(positions) >= config.max_positions:
            skipped.append({**signal, "reason": "max_positions_reached"})
            continue

        risk_amount = max(
            0.0,
            float(equity_reference) * float(config.risk_per_trade) * overlay_scale,
        )
        qty_risk = int(np.floor(risk_amount / stop_distance))
        if _is_futures(instrument_kind):
            qty_lots = qty_risk // lot_size
            qty = qty_lots * lot_size
        else:
            qty = qty_risk
            qty_lots = max(1, int(np.floor(qty / lot_size))) if qty > 0 else 0
        if qty <= 0 or qty_lots <= 0:
            skipped.append({**signal, "reason": "invalid_price_or_size"})
            continue

        adv = float(signal.get("adv", 0.0))
        if adv > 0 and config.max_position_value_pct_adv > 0:
            max_notional = adv * config.max_position_value_pct_adv
            if _is_futures(instrument_kind):
                qty_adv_lots = int(np.floor(max_notional / max(base_price * lot_size, 1e-9)))
                qty_lots = min(qty_lots, qty_adv_lots)
                qty = qty_lots * lot_size
            else:
                qty_adv = int(np.floor(max_notional / max(base_price, 1e-9)))
                qty = min(qty, qty_adv)
                qty_lots = max(1, int(np.floor(qty / max(1, lot_size)))) if qty > 0 else 0
            if qty <= 0 or qty_lots <= 0:
                skipped.append({**signal, "reason": "adv_cap_zero_qty"})
                continue

        if overlay_enabled and config.risk_overlay_corr_clamp_enabled and positions:
            correlations = signal.get("correlations")
            max_corr = 0.0
            if isinstance(correlations, dict):
                for position in positions:
                    raw = correlations.get(position.symbol)
                    if raw is None:
                        raw = correlations.get(_position_underlying(position))
                    if raw is None:
                        continue
                    try:
                        max_corr = max(max_corr, abs(float(raw)))
                    except (TypeError, ValueError):
                        continue
            if max_corr >= float(config.risk_overlay_corr_threshold):
                reduced_qty = int(np.floor(qty * max(0.0, float(config.risk_overlay_corr_reduce_factor))))
                if _is_futures(instrument_kind):
                    qty_lots = reduced_qty // lot_size
                    qty = qty_lots * lot_size
                else:
                    qty = reduced_qty
                    qty_lots = (
                        max(1, int(np.floor(qty / max(1, lot_size)))) if qty > 0 else 0
                    )
                if qty <= 0 or qty_lots <= 0:
                    skipped.append(
                        {
                            **signal,
                            "underlying_symbol": underlying_symbol,
                            "sector": sector,
                            "reason": "risk_overlay_corr_clamp",
                            "correlation": max_corr,
                            "threshold": float(config.risk_overlay_corr_threshold),
                        }
                    )
                    continue

        slippage_bps = _step_slippage_bps(vol_scale, config) / 10_000
        fill_price = base_price * (1 + slippage_bps) if side == "BUY" else base_price * (1 - slippage_bps)
        if fill_price <= 0:
            skipped.append({**signal, "reason": "invalid_entry_price"})
            continue

        side_label = _side_label(side)
        notional = qty * fill_price

        intraday_cash_short = side == "SELL" and instrument_kind == "EQUITY_CASH" and config.equity_short_intraday_only
        entry_cfg = _step_cost_config(config, intraday=intraday_cash_short)
        entry_cost = _transaction_cost(
            notional=notional,
            side=side,
            instrument_kind=instrument_kind,
            config=entry_cfg,
        )
        margin_required = (
            notional * max(0.0, config.futures_initial_margin_pct)
            if _is_futures(instrument_kind)
            else (notional if side == "SELL" else 0.0)
        )
        required_cash = margin_required + entry_cost if (side == "SELL" or _is_futures(instrument_kind)) else (notional + entry_cost)
        if cash < required_cash:
            skipped.append(
                {
                    **signal,
                    "reason": "insufficient_margin"
                    if (side == "SELL" or _is_futures(instrument_kind))
                    else "insufficient_cash",
                }
            )
            continue

        if overlay_enabled:
            projected_gross = gross_exposure_notional + notional
            if gross_cap_notional > 0 and projected_gross > gross_cap_notional + 1e-9:
                skipped.append(
                    {
                        **signal,
                        "underlying_symbol": underlying_symbol,
                        "sector": sector,
                        "reason": "risk_overlay_gross_exposure_cap",
                        "projected_gross_notional": projected_gross,
                        "cap_notional": gross_cap_notional,
                    }
                )
                continue
            projected_single = single_name_exposure_notional.get(underlying_symbol, 0.0) + notional
            if single_name_cap_notional > 0 and projected_single > single_name_cap_notional + 1e-9:
                skipped.append(
                    {
                        **signal,
                        "underlying_symbol": underlying_symbol,
                        "sector": sector,
                        "reason": "risk_overlay_single_name_cap",
                        "projected_notional": projected_single,
                        "cap_notional": single_name_cap_notional,
                    }
                )
                continue
            projected_sector = sector_exposure_notional.get(sector, 0.0) + notional
            if sector_cap_notional > 0 and projected_sector > sector_cap_notional + 1e-9:
                skipped.append(
                    {
                        **signal,
                        "underlying_symbol": underlying_symbol,
                        "sector": sector,
                        "reason": "risk_overlay_sector_cap",
                        "projected_notional": projected_sector,
                        "cap_notional": sector_cap_notional,
                    }
                )
                continue

        cash -= required_cash
        traded_notional_total += float(notional)
        entry_slippage_cost = ((fill_price - base_price) * qty) if side == "BUY" else ((base_price - fill_price) * qty)
        entry_slippage_cost_total += max(0.0, float(entry_slippage_cost))
        if overlay_enabled:
            gross_exposure_notional += notional
            single_name_exposure_notional[underlying_symbol] = (
                single_name_exposure_notional.get(underlying_symbol, 0.0) + notional
            )
            sector_exposure_notional[sector] = (
                sector_exposure_notional.get(sector, 0.0) + notional
            )
        entry_cost_total += float(entry_cost)
        target_raw = signal.get("target_price")
        target_price = float(target_raw) if isinstance(target_raw, (int, float)) else None
        start_stop = fill_price - stop_distance if side == "BUY" else fill_price + stop_distance
        position = SimulationPosition(
            symbol=symbol,
            side=side_label,
            instrument_kind=instrument_kind,
            entry_idx=0,
            entry_time=asof,
            qty=qty,
            lot_size=lot_size if _is_futures(instrument_kind) else max(1, lot_size),
            qty_lots=qty_lots,
            entry_price=fill_price,
            stop_price=start_stop,
            trail_price=start_stop,
            target_price=target_price,
            stop_distance=stop_distance,
            entry_cost=float(entry_cost),
            entry_notional=notional,
            margin_reserved=float(margin_required),
            force_eod=intraday_cash_short,
            metadata_json={
                "template": str(signal.get("template", "")),
                "underlying_symbol": underlying_symbol,
                "sector": sector,
                "instrument_choice_reason": str(signal.get("instrument_choice_reason", "provided")),
                "vol_scale": float(vol_scale),
            },
            source_position_id=None,
            bars_held=0,
        )
        positions.append(position)
        orders.append(
            {
                "symbol": symbol,
                "side": side,
                "instrument_kind": instrument_kind,
                "lot_size": lot_size,
                "qty_lots": qty_lots,
                "qty": qty,
                "fill_price": fill_price,
                "status": "FILLED",
                "reason": "SIGNAL_SHORT" if side == "SELL" else "SIGNAL",
            }
        )
        executed_signals.append(
            {
                **signal,
                "symbol": symbol,
                "underlying_symbol": underlying_symbol,
                "sector": sector,
                "instrument_kind": instrument_kind,
                "lot_size": lot_size,
                "qty_lots": qty_lots,
                "qty": qty,
                "fill_price": fill_price,
                "entry_cost": float(entry_cost),
                "margin_reserved": float(margin_required),
                "must_exit_by_eod": bool(intraday_cash_short),
            }
        )

    survivors: list[SimulationPosition] = []
    for position in positions:
        mark = mark_prices.get(position.symbol)
        if mark is None:
            survivors.append(position)
            continue
        mark_val = float(mark)
        should_close = False
        reason = ""
        trigger = _stop_trigger(position)
        if _stop_hit(position, mark_val, mark_val, trigger):
            should_close = True
            reason = "STOP_HIT"
        elif _target_hit(position, mark_val, mark_val):
            should_close = True
            reason = "EXITED"

        if not should_close:
            survivors.append(position)
            continue
        pseudo_bar = pd.Series({"open": mark_val, "high": mark_val, "low": mark_val, "close": mark_val})
        vol_scale = 0.0
        if isinstance(position.metadata_json, dict):
            vol_scale = float(position.metadata_json.get("vol_scale", 0.0) or 0.0)
        atr_value = mark_val * max(0.0, vol_scale)
        exit_cfg = _step_cost_config(
            config,
            intraday=(position.side == "SHORT" and position.instrument_kind == "EQUITY_CASH" and position.force_eod),
        )
        cash_delta, trade = _exit_position(
            position=position,
            bar=pseudo_bar,
            timestamp=asof,
            atr_value=atr_value,
            reason=reason,
            trigger=trigger if reason == "STOP_HIT" else None,
            config=exit_cfg,
        )
        cash += cash_delta
        exit_cost_total += float(trade["exit_cost"])
        adverse_slip = ((mark_val - trade["exit_px"]) * position.qty) if position.side == "LONG" else ((trade["exit_px"] - mark_val) * position.qty)
        exit_slippage_cost_total += max(0.0, float(adverse_slip))
        orders.append(
            {
                "symbol": position.symbol,
                "side": _exit_side_for_cost(position.side),
                "instrument_kind": position.instrument_kind,
                "lot_size": position.lot_size,
                "qty_lots": position.qty_lots,
                "qty": position.qty,
                "fill_price": float(trade["exit_px"]),
                "status": reason,
                "reason": reason,
            }
        )
        trades.append(trade)

    positions = survivors

    if _is_squareoff_due(asof.to_pydatetime(), _squareoff_cutoff(config.squareoff_time)):
        carry: list[SimulationPosition] = []
        for position in positions:
            if not (position.side == "SHORT" and position.force_eod):
                carry.append(position)
                continue
            mark_val = float(mark_prices.get(position.symbol, position.entry_price))
            pseudo_bar = pd.Series({"open": mark_val, "high": mark_val, "low": mark_val, "close": mark_val})
            exit_cfg = _step_cost_config(config, intraday=True)
            cash_delta, trade = _exit_position(
                position=position,
                bar=pseudo_bar,
                timestamp=asof,
                atr_value=0.0,
                reason="EOD_SQUARE_OFF",
                trigger=None,
                config=exit_cfg,
            )
            cash += cash_delta
            exit_cost_total += float(trade["exit_cost"])
            orders.append(
                {
                    "symbol": position.symbol,
                    "side": _exit_side_for_cost(position.side),
                    "instrument_kind": position.instrument_kind,
                    "lot_size": position.lot_size,
                    "qty_lots": position.qty_lots,
                    "qty": position.qty,
                    "fill_price": float(trade["exit_px"]),
                    "status": "EOD_SQUARE_OFF",
                    "reason": "EOD_SQUARE_OFF",
                }
            )
            trades.append(trade)
        positions = carry

    equity = float(cash)
    for position in positions:
        mark = float(mark_prices.get(position.symbol, position.entry_price))
        equity += _mark_to_market(position, mark)

    result_positions: list[dict[str, Any]] = []
    for position in positions:
        result_positions.append(_from_internal_position(position, source_id=position.source_position_id))

    metadata = {
        "engine_version": ENGINE_VERSION,
        "seed": int(config.seed),
        "data_digest": _step_data_digest(
            signals=ordered_signals,
            positions=normalized_positions,
            mark_prices=mark_prices,
            cash=float(cash),
            equity_reference=float(equity_reference),
            asof=asof,
            config=config,
        ),
        "risk_overlay": {
            "enabled": bool(overlay_enabled),
            "risk_scale": float(overlay_scale),
            "realized_vol": float(config.risk_overlay_realized_vol),
            "target_vol": float(config.risk_overlay_target_vol),
            "caps": {
                "max_gross_exposure_pct": float(config.risk_overlay_max_gross_exposure_pct),
                "max_single_name_exposure_pct": float(config.risk_overlay_max_single_name_exposure_pct),
                "max_sector_exposure_pct": float(config.risk_overlay_max_sector_exposure_pct),
            },
            "corr_clamp": {
                "enabled": bool(config.risk_overlay_corr_clamp_enabled),
                "threshold": float(config.risk_overlay_corr_threshold),
                "reduce_factor": float(config.risk_overlay_corr_reduce_factor),
            },
        },
    }

    return PortfolioStepResult(
        positions=result_positions,
        orders=orders,
        trades=trades,
        executed_signals=executed_signals,
        skipped_signals=skipped,
        cash=float(cash),
        equity=equity,
        entry_cost_total=float(entry_cost_total),
        exit_cost_total=float(exit_cost_total),
        entry_slippage_cost=float(entry_slippage_cost_total),
        exit_slippage_cost=float(exit_slippage_cost_total),
        traded_notional_total=float(traded_notional_total),
        metadata=metadata,
    )
