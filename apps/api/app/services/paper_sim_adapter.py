from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from sqlmodel import Session

from app.core.config import Settings
from app.db.models import PaperOrder, PaperPosition, PaperState
from app.engine.simulator import SimulationConfig, simulate_portfolio_step


@dataclass
class PaperSimulatorExecution:
    executed_signals: list[dict[str, Any]]
    skipped_signals: list[dict[str, Any]]
    positions_after: list[dict[str, Any]]
    orders_generated: list[dict[str, Any]]
    trades_generated: list[dict[str, Any]]
    entry_cost_total: float
    exit_cost_total: float
    entry_slippage_cost_total: float
    exit_slippage_cost_total: float
    traded_notional_total: float
    cash: float
    equity: float
    metadata: dict[str, Any]


def _asof_timestamp(asof_dt: datetime) -> pd.Timestamp:
    if asof_dt.tzinfo is None:
        return pd.Timestamp(asof_dt.replace(tzinfo=timezone.utc))
    return pd.Timestamp(asof_dt.astimezone(timezone.utc))


def _position_payload(position: PaperPosition) -> dict[str, Any]:
    return {
        "id": int(position.id) if position.id is not None else None,
        "symbol": position.symbol,
        "side": position.side,
        "instrument_kind": position.instrument_kind,
        "lot_size": int(position.lot_size),
        "qty_lots": int(position.qty_lots),
        "margin_reserved": float(position.margin_reserved),
        "must_exit_by_eod": bool(position.must_exit_by_eod),
        "qty": int(position.qty),
        "avg_price": float(position.avg_price),
        "stop_price": float(position.stop_price) if position.stop_price is not None else None,
        "target_price": float(position.target_price) if position.target_price is not None else None,
        "metadata_json": dict(position.metadata_json or {}),
        "opened_at": position.opened_at.isoformat(),
    }


def _position_payload_from_dict(position: dict[str, Any]) -> dict[str, Any]:
    opened_at = position.get("opened_at")
    return {
        "id": position.get("id"),
        "symbol": str(position.get("symbol", "")).upper(),
        "side": str(position.get("side", "BUY")).upper(),
        "instrument_kind": str(position.get("instrument_kind", "EQUITY_CASH")).upper(),
        "lot_size": int(position.get("lot_size", 1) or 1),
        "qty_lots": int(position.get("qty_lots", 1) or 1),
        "margin_reserved": float(position.get("margin_reserved", 0.0) or 0.0),
        "must_exit_by_eod": bool(position.get("must_exit_by_eod", False)),
        "qty": int(position.get("qty", 0) or 0),
        "avg_price": float(position.get("avg_price", 0.0) or 0.0),
        "stop_price": (
            float(position["stop_price"]) if position.get("stop_price") is not None else None
        ),
        "target_price": (
            float(position["target_price"]) if position.get("target_price") is not None else None
        ),
        "metadata_json": dict(position.get("metadata_json") or {}),
        "opened_at": str(opened_at) if opened_at else datetime.now(tz=timezone.utc).isoformat(),
    }


def _build_simulation_config(
    *,
    settings: Settings,
    state_settings: dict[str, Any],
    policy: dict[str, Any],
    equity: float,
    seed: int,
) -> SimulationConfig:
    cost_model = policy.get("cost_model", {}) if isinstance(policy.get("cost_model"), dict) else {}
    cost_enabled = bool(
        cost_model.get("enabled", state_settings.get("cost_model_enabled", settings.cost_model_enabled))
    )
    cost_mode = str(cost_model.get("mode", state_settings.get("cost_mode", settings.cost_mode)))
    cost_params = {
        "brokerage_bps": float(state_settings.get("brokerage_bps", settings.brokerage_bps)),
        "stt_delivery_buy_bps": float(
            state_settings.get("stt_delivery_buy_bps", settings.stt_delivery_buy_bps)
        ),
        "stt_delivery_sell_bps": float(
            state_settings.get("stt_delivery_sell_bps", settings.stt_delivery_sell_bps)
        ),
        "stt_intraday_buy_bps": float(
            state_settings.get("stt_intraday_buy_bps", settings.stt_intraday_buy_bps)
        ),
        "stt_intraday_sell_bps": float(
            state_settings.get("stt_intraday_sell_bps", settings.stt_intraday_sell_bps)
        ),
        "exchange_txn_bps": float(state_settings.get("exchange_txn_bps", settings.exchange_txn_bps)),
        "sebi_bps": float(state_settings.get("sebi_bps", settings.sebi_bps)),
        "stamp_delivery_buy_bps": float(
            state_settings.get("stamp_delivery_buy_bps", settings.stamp_delivery_buy_bps)
        ),
        "stamp_intraday_buy_bps": float(
            state_settings.get("stamp_intraday_buy_bps", settings.stamp_intraday_buy_bps)
        ),
        "gst_rate": float(state_settings.get("gst_rate", settings.gst_rate)),
        "futures_brokerage_bps": float(
            state_settings.get("futures_brokerage_bps", settings.futures_brokerage_bps)
        ),
        "futures_stt_sell_bps": float(
            state_settings.get("futures_stt_sell_bps", settings.futures_stt_sell_bps)
        ),
        "futures_exchange_txn_bps": float(
            state_settings.get("futures_exchange_txn_bps", settings.futures_exchange_txn_bps)
        ),
        "futures_stamp_buy_bps": float(
            state_settings.get("futures_stamp_buy_bps", settings.futures_stamp_buy_bps)
        ),
    }

    return SimulationConfig(
        risk_per_trade=float(policy.get("risk_per_trade", settings.risk_per_trade)),
        max_positions=int(policy.get("max_positions", settings.max_positions)),
        initial_equity=float(equity),
        commission_bps=float(state_settings.get("commission_bps", settings.commission_bps)),
        slippage_base_bps=float(state_settings.get("slippage_base_bps", settings.slippage_base_bps)),
        slippage_vol_factor=float(
            state_settings.get("slippage_vol_factor", settings.slippage_vol_factor)
        ),
        max_position_value_pct_adv=float(
            state_settings.get("max_position_value_pct_adv", settings.max_position_value_pct_adv)
        ),
        allow_long=True,
        allow_short=True,
        futures_initial_margin_pct=float(
            state_settings.get("futures_initial_margin_pct", settings.futures_initial_margin_pct)
        ),
        equity_short_intraday_only=True,
        squareoff_time=str(
            state_settings.get("paper_short_squareoff_time", settings.paper_short_squareoff_time)
        ),
        cost_model_enabled=cost_enabled,
        cost_mode=cost_mode,
        cost_params=cost_params,
        seed=int(seed),
    )


def execute_paper_step_with_simulator(
    *,
    session: Session,
    settings: Settings,
    state: PaperState | dict[str, Any],
    state_settings: dict[str, Any],
    policy: dict[str, Any],
    asof_dt: datetime,
    selected_signals: list[dict[str, Any]],
    mark_prices: dict[str, float],
    open_positions: list[PaperPosition] | list[dict[str, Any]],
    seed: int,
    persist_live_state: bool = True,
) -> PaperSimulatorExecution:
    state_cash = float(state.cash) if isinstance(state, PaperState) else float(state.get("cash", 0.0))
    state_equity = (
        float(state.equity) if isinstance(state, PaperState) else float(state.get("equity", state_cash))
    )
    config = _build_simulation_config(
        settings=settings,
        state_settings=state_settings,
        policy=policy,
        equity=state_equity,
        seed=seed,
    )
    if open_positions and isinstance(open_positions[0], PaperPosition):
        open_payload = [_position_payload(item) for item in open_positions if isinstance(item, PaperPosition)]
    else:
        open_payload = [
            _position_payload_from_dict(item)
            for item in open_positions
            if isinstance(item, dict)
        ]

    step = simulate_portfolio_step(
        signals=selected_signals,
        open_positions=open_payload,
        mark_prices={str(key): float(value) for key, value in mark_prices.items()},
        asof=_asof_timestamp(asof_dt),
        cash=state_cash,
        equity_reference=state_equity,
        config=config,
    )

    if persist_live_state:
        existing_by_id = {
            int(row.id): row
            for row in open_positions
            if isinstance(row, PaperPosition) and row.id is not None
        }
        surviving_ids = {
            int(item["source_position_id"])
            for item in step.positions
            if item.get("source_position_id") is not None
        }
        for row_id, row in existing_by_id.items():
            if row_id in surviving_ids:
                continue
            session.delete(row)

        for item in step.positions:
            source_id = item.get("source_position_id")
            position: PaperPosition | None = None
            if isinstance(source_id, int):
                position = existing_by_id.get(source_id)

            if position is None:
                position = PaperPosition(
                    symbol=str(item["symbol"]),
                    side=str(item["side"]),
                    instrument_kind=str(item["instrument_kind"]),
                    lot_size=int(item["lot_size"]),
                    qty_lots=int(item["qty_lots"]),
                    margin_reserved=float(item["margin_reserved"]),
                    must_exit_by_eod=bool(item["must_exit_by_eod"]),
                    qty=int(item["qty"]),
                    avg_price=float(item["avg_price"]),
                    stop_price=(
                        float(item["stop_price"]) if item.get("stop_price") is not None else None
                    ),
                    target_price=(
                        float(item["target_price"]) if item.get("target_price") is not None else None
                    ),
                    metadata_json=dict(item.get("metadata_json") or {}),
                    opened_at=(
                        datetime.fromisoformat(str(item.get("opened_at")))
                        if item.get("opened_at")
                        else asof_dt
                    ),
                )
            else:
                position.symbol = str(item["symbol"])
                position.side = str(item["side"])
                position.instrument_kind = str(item["instrument_kind"])
                position.lot_size = int(item["lot_size"])
                position.qty_lots = int(item["qty_lots"])
                position.margin_reserved = float(item["margin_reserved"])
                position.must_exit_by_eod = bool(item["must_exit_by_eod"])
                position.qty = int(item["qty"])
                position.avg_price = float(item["avg_price"])
                position.stop_price = (
                    float(item["stop_price"]) if item.get("stop_price") is not None else None
                )
                position.target_price = (
                    float(item["target_price"]) if item.get("target_price") is not None else None
                )
                position.metadata_json = dict(item.get("metadata_json") or {})
            session.add(position)

        now = datetime.now(tz=timezone.utc)
        for item in step.orders:
            session.add(
                PaperOrder(
                    symbol=str(item["symbol"]),
                    side=str(item["side"]),
                    instrument_kind=str(item["instrument_kind"]),
                    lot_size=int(item["lot_size"]),
                    qty_lots=int(item["qty_lots"]),
                    qty=int(item["qty"]),
                    fill_price=float(item["fill_price"]) if item.get("fill_price") is not None else None,
                    status=str(item["status"]),
                    reason=str(item.get("reason")) if item.get("reason") else None,
                    created_at=now,
                    updated_at=now,
                )
            )

        if isinstance(state, PaperState):
            state.cash = float(step.cash)
            state.equity = float(step.equity)
            session.add(state)

    return PaperSimulatorExecution(
        executed_signals=list(step.executed_signals),
        skipped_signals=list(step.skipped_signals),
        positions_after=list(step.positions),
        orders_generated=list(step.orders),
        trades_generated=list(step.trades),
        entry_cost_total=float(step.entry_cost_total),
        exit_cost_total=float(step.exit_cost_total),
        entry_slippage_cost_total=float(step.entry_slippage_cost),
        exit_slippage_cost_total=float(step.exit_slippage_cost),
        traded_notional_total=float(step.traded_notional_total),
        cash=float(step.cash),
        equity=float(step.equity),
        metadata=dict(step.metadata),
    )
