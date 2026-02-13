from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.engine.costs import estimate_equity_delivery_cost, estimate_intraday_cost
from app.engine.indicators import atr
from app.engine.metrics import calculate_metrics


@dataclass
class BacktestConfig:
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
    cost_model_enabled: bool = False
    cost_mode: str = "delivery"
    cost_params: dict[str, float] = field(default_factory=dict)


@dataclass
class Position:
    symbol: str
    entry_idx: int
    entry_time: pd.Timestamp
    qty: int
    entry_price: float
    stop_price: float
    trail_price: float
    target_price: float | None
    stop_distance: float
    entry_commission: float
    bars_held: int = 0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


def _slippage_bps(atr_value: float, close: float, config: BacktestConfig) -> float:
    if close <= 0 or np.isnan(atr_value):
        return config.slippage_base_bps
    return config.slippage_base_bps + config.slippage_vol_factor * (atr_value / close)


def _transaction_cost(notional: float, side: str, config: BacktestConfig) -> float:
    if notional <= 0:
        return 0.0
    if config.cost_model_enabled:
        if str(config.cost_mode).lower() == "intraday":
            return estimate_intraday_cost(notional=notional, side=side, config=config.cost_params)
        return estimate_equity_delivery_cost(
            notional=notional, side=side, config=config.cost_params
        )
    return notional * config.commission_bps / 10_000


def _exit_position(
    bar: pd.Series,
    pos: Position,
    config: BacktestConfig,
    atr_value: float,
    reason: str,
    timestamp: pd.Timestamp,
    stop_trigger: float | None = None,
) -> tuple[float, dict[str, float | int | str]]:
    slippage = _slippage_bps(atr_value, float(bar["close"]), config) / 10_000
    if reason == "STOP_HIT":
        stop_ref = max(0.0, stop_trigger if stop_trigger is not None else pos.trail_price)
        # Gap-through-stop realism: if the bar opens below stop, fill at the worse open price.
        stop_fill_ref = min(float(bar["open"]), stop_ref)
        exit_px = max(0.0, stop_fill_ref) * (1 - slippage)
    elif reason == "TARGET_HIT" and pos.target_price is not None:
        exit_px = pos.target_price * (1 - slippage)
    else:
        exit_px = float(bar["close"]) * (1 - slippage)

    exit_notional = pos.qty * exit_px
    exit_cost = _transaction_cost(exit_notional, "SELL", config)
    pnl = (exit_px - pos.entry_price) * pos.qty - (pos.entry_commission + exit_cost)
    cash_delta = exit_notional - exit_cost

    trade = {
        "symbol": pos.symbol,
        "entry_dt": pos.entry_time,
        "exit_dt": timestamp,
        "qty": pos.qty,
        "entry_px": pos.entry_price,
        "exit_px": exit_px,
        "pnl": pnl,
        "r_multiple": pnl / (pos.stop_distance * pos.qty) if pos.stop_distance > 0 else 0.0,
        "reason": reason,
        "holding_bars": pos.bars_held,
        "notional": pos.entry_price * pos.qty,
    }
    return cash_delta, trade


def run_backtest(
    price_df: pd.DataFrame,
    entries: pd.Series,
    symbol: str,
    config: BacktestConfig,
) -> BacktestResult:
    """Run a realistic long-only backtest with next-bar fills and transaction costs."""
    if price_df.empty:
        return BacktestResult(trades=pd.DataFrame(), equity_curve=pd.DataFrame(), metrics={})

    frame = price_df.copy()
    if "datetime" in frame.columns:
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
        frame = frame.set_index("datetime")
    frame = frame.sort_index()

    if len(entries) == len(frame):
        entries_aligned = pd.Series(entries.to_numpy(), index=frame.index)
    else:
        entries_aligned = entries.reindex(frame.index)
    entries = (
        pd.Series(entries_aligned, index=frame.index, dtype="boolean").fillna(False).astype(bool)
    )
    atr_series = atr(frame, config.atr_period)
    adv_notional = (
        (frame["close"] * frame["volume"]).rolling(config.adv_lookback, min_periods=1).mean()
    )

    cash = config.initial_equity
    positions: list[Position] = []
    trade_rows: list[dict[str, float | int | str]] = []
    equity_rows: list[dict[str, float]] = []
    open_counts: list[int] = []

    for i in range(1, len(frame)):
        bar = frame.iloc[i]
        timestamp = frame.index[i]

        survivors: list[Position] = []
        for pos in positions:
            pos.bars_held += 1
            trail_candidate = float(bar["close"]) - config.atr_trail_mult * float(
                atr_series.iloc[i]
            )
            if not np.isnan(trail_candidate):
                pos.trail_price = max(pos.trail_price, trail_candidate)

            exit_reason: str | None = None
            stop_trigger = max(pos.stop_price, pos.trail_price)
            if float(bar["low"]) <= stop_trigger:
                pos.trail_price = stop_trigger
                exit_reason = "STOP_HIT"
            elif pos.target_price is not None and float(bar["high"]) >= pos.target_price:
                exit_reason = "TARGET_HIT"
            elif config.time_stop_bars is not None and pos.bars_held >= config.time_stop_bars:
                exit_reason = "TIME_STOP"

            if exit_reason is None:
                survivors.append(pos)
                continue

            cash_delta, trade = _exit_position(
                bar=bar,
                pos=pos,
                config=config,
                atr_value=float(atr_series.iloc[i]),
                reason=exit_reason,
                timestamp=timestamp,
                stop_trigger=stop_trigger if exit_reason == "STOP_HIT" else None,
            )
            cash += cash_delta
            trade_rows.append(trade)

        positions = survivors

        signal_bar_idx = i - 1
        signal_on_close = bool(entries.iloc[signal_bar_idx])
        liquidity_ok = (
            float(frame.iloc[signal_bar_idx]["close"]) * float(frame.iloc[signal_bar_idx]["volume"])
            >= config.min_notional
        )

        if (
            signal_on_close
            and liquidity_ok
            and len(positions) < config.max_positions
            and config.allow_long
        ):
            stop_distance = config.atr_stop_mult * float(atr_series.iloc[signal_bar_idx])
            if stop_distance > 0 and not np.isnan(stop_distance):
                risk_amount = cash * config.risk_per_trade
                qty_risk = int(np.floor(risk_amount / stop_distance))
                signal_adv = float(adv_notional.iloc[signal_bar_idx])
                qty_adv = qty_risk
                if config.max_position_value_pct_adv > 0 and signal_adv > 0:
                    max_position_value = signal_adv * config.max_position_value_pct_adv
                    qty_adv = int(np.floor(max_position_value / float(bar["open"])))
                qty = min(qty_risk, qty_adv)
                if qty > 0:
                    slip = (
                        _slippage_bps(
                            float(atr_series.iloc[signal_bar_idx]),
                            float(frame.iloc[signal_bar_idx]["close"]),
                            config,
                        )
                        / 10_000
                    )
                    entry_px = float(bar["open"]) * (1 + slip)
                    entry_value = qty * entry_px
                    entry_commission = _transaction_cost(entry_value, "BUY", config)
                    if cash >= (entry_value + entry_commission):
                        cash -= entry_value + entry_commission
                        target = None
                        if config.take_profit_r is not None:
                            target = entry_px + config.take_profit_r * stop_distance
                        positions.append(
                            Position(
                                symbol=symbol,
                                entry_idx=i,
                                entry_time=timestamp,
                                qty=qty,
                                entry_price=entry_px,
                                stop_price=entry_px - stop_distance,
                                trail_price=entry_px - stop_distance,
                                target_price=target,
                                stop_distance=stop_distance,
                                entry_commission=entry_commission,
                            )
                        )

        mark_to_market = sum(pos.qty * float(bar["close"]) for pos in positions)
        equity_rows.append({"datetime": timestamp, "equity": cash + mark_to_market})
        open_counts.append(len(positions))

    if equity_rows and positions:
        final_bar = frame.iloc[-1]
        final_ts = frame.index[-1]
        for pos in positions:
            cash_delta, trade = _exit_position(
                bar=final_bar,
                pos=pos,
                config=config,
                atr_value=float(atr_series.iloc[-1]),
                reason="EOD_CLOSE",
                timestamp=final_ts,
            )
            cash += cash_delta
            trade_rows.append(trade)

        equity_rows[-1]["equity"] = cash

    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame(equity_rows)
    equity_series = (
        equity_df.set_index("datetime")["equity"] if not equity_df.empty else pd.Series()
    )
    open_count_series = pd.Series(open_counts)

    metrics = calculate_metrics(
        equity=equity_series,
        trades=trades_df
        if not trades_df.empty
        else pd.DataFrame(columns=["pnl", "notional", "holding_bars"]),
        open_position_count=open_count_series,
    )

    return BacktestResult(trades=trades_df, equity_curve=equity_df, metrics=metrics)
