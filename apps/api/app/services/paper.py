from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any

from fastapi.encoders import jsonable_encoder
import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import (
    AuditLog,
    Dataset,
    PaperOrder,
    PaperPosition,
    PaperState,
    Policy,
    Symbol,
)
from app.engine.costs import estimate_equity_delivery_cost, estimate_intraday_cost
from app.engine.signal_engine import generate_signals_for_policy
from app.services.data_store import DataStore
from app.services.regime import regime_policy


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def get_or_create_paper_state(session: Session, settings: Settings) -> PaperState:
    state = session.get(PaperState, 1)
    if state is not None:
        return state

    state = PaperState(
        id=1,
        equity=1_000_000.0,
        cash=1_000_000.0,
        peak_equity=1_000_000.0,
        drawdown=0.0,
        kill_switch_active=False,
        cooldown_days_left=0,
        settings_json={
            "risk_per_trade": settings.risk_per_trade,
            "max_positions": settings.max_positions,
            "kill_switch_dd": settings.kill_switch_drawdown,
            "max_position_value_pct_adv": settings.max_position_value_pct_adv,
            "diversification_corr_threshold": settings.diversification_corr_threshold,
            "cost_model_enabled": settings.cost_model_enabled,
            "cost_mode": settings.cost_mode,
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
            "paper_mode": "strategy",
            "active_policy_id": None,
        },
    )
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


def get_positions(session: Session) -> list[PaperPosition]:
    return list(session.exec(select(PaperPosition).order_by(PaperPosition.opened_at.desc())).all())


def get_orders(session: Session) -> list[PaperOrder]:
    return list(session.exec(select(PaperOrder).order_by(PaperOrder.created_at.desc())).all())


def _log(session: Session, event_type: str, payload: dict[str, Any]) -> None:
    session.add(AuditLog(type=event_type, payload_json=payload))


def _dump_model(value: Any) -> Any:
    fields = getattr(type(value), "model_fields", None)
    if isinstance(fields, dict):
        return jsonable_encoder({key: getattr(value, key, None) for key in fields})
    return jsonable_encoder(value)


def _position_size(equity: float, risk_per_trade: float, stop_distance: float) -> int:
    if stop_distance <= 0:
        return 0
    risk_amount = equity * risk_per_trade
    return int(np.floor(risk_amount / stop_distance))


def _correlation_reject(
    candidate: dict[str, Any],
    selected_symbols: set[str],
    threshold: float,
) -> bool:
    matrix = candidate.get("correlations", {})
    if not isinstance(matrix, dict):
        return False
    for symbol in selected_symbols:
        corr = matrix.get(symbol)
        if corr is None:
            continue
        try:
            if abs(float(corr)) >= threshold:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _cost_settings(state_settings: dict[str, Any], settings: Settings) -> dict[str, float]:
    return {
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
        "exchange_txn_bps": float(
            state_settings.get("exchange_txn_bps", settings.exchange_txn_bps)
        ),
        "sebi_bps": float(state_settings.get("sebi_bps", settings.sebi_bps)),
        "stamp_delivery_buy_bps": float(
            state_settings.get("stamp_delivery_buy_bps", settings.stamp_delivery_buy_bps)
        ),
        "stamp_intraday_buy_bps": float(
            state_settings.get("stamp_intraday_buy_bps", settings.stamp_intraday_buy_bps)
        ),
        "gst_rate": float(state_settings.get("gst_rate", settings.gst_rate)),
    }


def _transaction_cost(
    *,
    notional: float,
    side: str,
    settings: Settings,
    state_settings: dict[str, Any],
    override_cost_model: dict[str, Any] | None,
) -> float:
    if notional <= 0:
        return 0.0

    cost_model = dict(override_cost_model or {})
    enabled = bool(cost_model.get("enabled", state_settings.get("cost_model_enabled", False)))
    mode = str(cost_model.get("mode", state_settings.get("cost_mode", settings.cost_mode))).lower()
    if enabled:
        cfg = _cost_settings(state_settings, settings)
        cfg.update(
            {
                key: float(value)
                for key, value in cost_model.items()
                if key in cfg and isinstance(value, (int, float))
            }
        )
        if mode == "intraday":
            return estimate_intraday_cost(notional=notional, side=side, config=cfg)
        return estimate_equity_delivery_cost(notional=notional, side=side, config=cfg)

    commission_bps = float(state_settings.get("commission_bps", settings.commission_bps))
    return max(0.0, notional * commission_bps / 10_000)


def _parse_asof(value: Any) -> datetime | date | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    return None


def _resolve_execution_policy(
    session: Session,
    state: PaperState,
    settings: Settings,
    regime: str,
    policy_override_id: int | None = None,
) -> dict[str, Any]:
    base = regime_policy(regime, settings.risk_per_trade, settings.max_positions)
    resolved: dict[str, Any] = {
        **base,
        "mode": "strategy",
        "policy_id": None,
        "policy_name": None,
        "policy_definition": {},
        "selection_reason": f"Default regime policy for {regime}.",
        "params": {},
        "cost_model": {},
    }

    state_settings = state.settings_json or {}
    if policy_override_id is None and str(state_settings.get("paper_mode", "strategy")) != "policy":
        return resolved

    if policy_override_id is not None:
        policy_int = int(policy_override_id)
    else:
        policy_id = state_settings.get("active_policy_id")
        try:
            policy_int = int(policy_id)
        except (TypeError, ValueError):
            resolved["selection_reason"] = "Policy mode requested but active policy id is invalid."
            return resolved

    policy = session.get(Policy, policy_int)
    if policy is None:
        resolved["selection_reason"] = "Policy mode requested but policy record was not found."
        return resolved

    policy_definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}
    regime_map = policy_definition.get("regime_map", {})
    if not isinstance(regime_map, dict):
        resolved["selection_reason"] = (
            "Policy has no valid regime map; using default regime policy."
        )
        return resolved

    regime_config = regime_map.get(regime)
    if not isinstance(regime_config, dict):
        return {
            "allowed_templates": [],
            "risk_per_trade": 0.0,
            "max_positions": 0,
            "mode": "policy",
            "policy_id": policy.id,
            "policy_name": policy.name,
            "policy_definition": policy_definition,
            "selection_reason": f"Policy {policy.name} blocks new entries in {regime}.",
            "params": {},
            "cost_model": dict(policy_definition.get("cost_model", {})),
        }

    strategy_key = regime_config.get("strategy_key")
    risk_scale = float(regime_config.get("risk_scale", 1.0))
    max_positions_scale = float(regime_config.get("max_positions_scale", 1.0))
    allowed_templates = (
        [str(strategy_key)]
        if isinstance(strategy_key, str) and strategy_key
        else [str(item) for item in regime_config.get("allowed_templates", []) if str(item)]
    )
    return {
        "allowed_templates": allowed_templates,
        "risk_per_trade": max(0.0, settings.risk_per_trade * risk_scale),
        "max_positions": max(0, int(round(settings.max_positions * max_positions_scale))),
        "mode": "policy",
        "policy_id": policy.id,
        "policy_name": policy.name,
        "policy_definition": policy_definition,
        "selection_reason": str(
            regime_config.get("reason", f"Policy {policy.name} selected for regime {regime}.")
        ),
        "params": dict(regime_config.get("params", {})),
        "cost_model": dict(policy_definition.get("cost_model", {})),
    }


def _resolve_dataset_id(
    session: Session,
    payload: dict[str, Any],
    policy: dict[str, Any],
    timeframes: list[str],
) -> int | None:
    explicit_id = payload.get("dataset_id")
    if isinstance(explicit_id, int):
        return explicit_id if session.get(Dataset, explicit_id) is not None else None
    try:
        explicit_id_int = int(explicit_id)
        if session.get(Dataset, explicit_id_int) is not None:
            return explicit_id_int
    except (TypeError, ValueError):
        pass

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        universe = definition.get("universe", {})
        if isinstance(universe, dict):
            dataset_id = universe.get("dataset_id")
            try:
                dataset_id_int = int(dataset_id)
                if session.get(Dataset, dataset_id_int) is not None:
                    return dataset_id_int
            except (TypeError, ValueError):
                pass

    preferred_timeframe = timeframes[0] if timeframes else "1d"
    candidate = session.exec(
        select(Dataset)
        .where(Dataset.timeframe == preferred_timeframe)
        .order_by(Dataset.created_at.desc())
    ).first()
    return candidate.id if candidate is not None else None


def _resolve_timeframes(payload: dict[str, Any], policy: dict[str, Any]) -> list[str]:
    payload_timeframes = payload.get("timeframes")
    if isinstance(payload_timeframes, list):
        values = [str(value).strip() for value in payload_timeframes if str(value).strip()]
        if values:
            return values

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        value = definition.get("timeframes", [])
        if isinstance(value, list):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                return values
    return ["1d"]


def _resolve_symbol_scope(payload: dict[str, Any], policy: dict[str, Any]) -> str:
    explicit = payload.get("symbol_scope")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        universe = definition.get("universe", {})
        if isinstance(universe, dict):
            scope = universe.get("symbol_scope")
            if isinstance(scope, str) and scope.strip():
                return scope.strip()
    return "liquid"


def _resolve_max_symbols_scan(payload: dict[str, Any], policy: dict[str, Any]) -> int:
    explicit = payload.get("max_symbols_scan")
    if isinstance(explicit, int):
        return max(1, explicit)
    try:
        return max(1, int(explicit))
    except (TypeError, ValueError):
        pass

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        universe = definition.get("universe", {})
        if isinstance(universe, dict):
            value = universe.get("max_symbols_scan")
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                pass
    return 50


def _resolve_seed(payload: dict[str, Any], policy: dict[str, Any]) -> int:
    explicit = payload.get("seed")
    if isinstance(explicit, int):
        return explicit
    try:
        return int(explicit)
    except (TypeError, ValueError):
        pass

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        ranking = definition.get("ranking", {})
        if isinstance(ranking, dict):
            value = ranking.get("seed")
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return 7


def run_paper_step(
    session: Session,
    settings: Settings,
    payload: dict[str, Any],
    store: DataStore | None = None,
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    regime = str(payload.get("regime", "TREND_UP"))
    policy = _resolve_execution_policy(session, state, settings, regime)
    state_settings = state.settings_json or {}

    if state.kill_switch_active:
        _log(session, "kill_switch", {"active": True})
        session.commit()
        return jsonable_encoder(
            {
                "status": "kill_switch_active",
                "regime": regime,
                "policy": policy,
                "policy_mode": policy.get("mode"),
                "policy_selection_reason": policy.get("selection_reason"),
                "signals_source": "provided",
                "generated_signals_count": 0,
                "selected_signals_count": 0,
                "positions": [_dump_model(p) for p in get_positions(session)],
                "orders": [_dump_model(o) for o in get_orders(session)],
            }
        )

    provided_signals = [dict(item) for item in payload.get("signals", []) if isinstance(item, dict)]
    auto_generate = bool(payload.get("auto_generate_signals", False))
    should_generate = auto_generate or (
        policy.get("mode") == "policy" and len(provided_signals) == 0
    )
    signals_source = "provided"
    generated_signals_count = 0

    if should_generate:
        store = store or DataStore(
            parquet_root=settings.parquet_root, duckdb_path=settings.duckdb_path
        )
        timeframes = _resolve_timeframes(payload, policy)
        dataset_id = _resolve_dataset_id(session, payload, policy, timeframes)
        symbol_scope = _resolve_symbol_scope(payload, policy)
        max_symbols_scan = _resolve_max_symbols_scan(payload, policy)
        seed = _resolve_seed(payload, policy)

        if dataset_id is None:
            provided_signals = []
            _log(
                session,
                "signals_generated",
                {
                    "mode": "policy_autopilot",
                    "generated_count": 0,
                    "dataset_id": None,
                    "reason": "dataset_not_found",
                },
            )
        else:
            provided_signals = generate_signals_for_policy(
                session=session,
                store=store,
                dataset_id=dataset_id,
                asof=_parse_asof(payload.get("asof")),
                timeframes=timeframes,
                allowed_templates=policy.get("allowed_templates", []),
                params_overrides=policy.get("params", {}),
                max_symbols_scan=max_symbols_scan,
                seed=seed,
                mode="paper",
                symbol_scope=symbol_scope,
            )
            generated_signals_count = len(provided_signals)
            _log(
                session,
                "signals_generated",
                {
                    "mode": "policy_autopilot",
                    "dataset_id": dataset_id,
                    "timeframes": timeframes,
                    "symbol_scope": symbol_scope,
                    "generated_count": generated_signals_count,
                    "allowed_templates": policy.get("allowed_templates", []),
                    "policy_id": policy.get("policy_id"),
                    "policy_name": policy.get("policy_name"),
                },
            )
        signals_source = "generated"

    current_positions = get_positions(session)
    open_symbols = {p.symbol for p in current_positions}
    max_positions = int(policy["max_positions"])
    correlation_threshold = float(
        state_settings.get(
            "diversification_corr_threshold", settings.diversification_corr_threshold
        )
    )
    max_position_value_pct_adv = float(
        state_settings.get("max_position_value_pct_adv", settings.max_position_value_pct_adv)
    )
    skipped_signals: list[dict[str, Any]] = []
    selected_signals: list[dict[str, Any]] = []

    candidates = sorted(
        [dict(item) for item in provided_signals],
        key=lambda item: float(item.get("signal_strength", 0.0)),
        reverse=True,
    )
    sectors = {
        row.symbol: (row.sector or "UNKNOWN")
        for row in session.exec(
            select(Symbol).where(
                Symbol.symbol.in_([str(s.get("symbol", "")).upper() for s in candidates])
            )
        ).all()
    }
    sector_counts: dict[str, int] = {}
    for pos in current_positions:
        sector = sectors.get(pos.symbol, "UNKNOWN")
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    selected_symbols = set(open_symbols)

    for signal in candidates:
        symbol = str(signal.get("symbol", "")).upper()
        side = str(signal.get("side", "BUY")).upper()
        template = str(signal.get("template", "trend_breakout"))
        base_meta = {
            "symbol": symbol,
            "template": template,
            "policy_mode": policy.get("mode"),
            "policy_id": policy.get("policy_id"),
            "policy_name": policy.get("policy_name"),
        }

        if len(current_positions) + len(selected_signals) >= max_positions:
            skipped_signals.append({**base_meta, "reason": "max_positions_reached"})
            continue
        if side != "BUY":
            skipped_signals.append({**base_meta, "reason": "non_buy_not_supported_yet"})
            continue
        if symbol in open_symbols:
            skipped_signals.append({**base_meta, "reason": "already_open"})
            continue
        if template not in policy["allowed_templates"]:
            skipped_signals.append(
                {
                    **base_meta,
                    "reason": "template_blocked_by_policy"
                    if policy.get("mode") == "policy"
                    else "template_blocked_by_regime",
                }
            )
            continue

        sector = sectors.get(symbol, "UNKNOWN")
        if sector_counts.get(sector, 0) >= 2:
            skipped_signals.append({**base_meta, "reason": "sector_concentration"})
            continue
        if _correlation_reject(
            signal, selected_symbols=selected_symbols, threshold=correlation_threshold
        ):
            skipped_signals.append({**base_meta, "reason": "correlation_threshold"})
            continue

        selected_signals.append(signal)
        selected_symbols.add(symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    entry_cost_total = 0.0
    exit_cost_total = 0.0
    executed_count = 0

    for signal in selected_signals:
        symbol = str(signal.get("symbol", "")).upper()
        template = str(signal.get("template", "trend_breakout"))
        price = float(signal.get("price", 0.0))
        stop_distance = float(signal.get("stop_distance", 0.0))
        qty = _position_size(state.equity, float(policy["risk_per_trade"]), stop_distance)
        if price <= 0 or qty <= 0:
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "reason": "invalid_price_or_size",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        slippage = (settings.slippage_base_bps / 10_000) * (1 + float(signal.get("vol_scale", 0.0)))
        fill_price = price * (1 + slippage)
        adv_notional = float(signal.get("adv", 0.0))
        if adv_notional > 0 and max_position_value_pct_adv > 0:
            max_notional = adv_notional * max_position_value_pct_adv
            qty_adv = int(np.floor(max_notional / max(fill_price, 1e-9)))
            if qty_adv <= 0:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "template": template,
                        "reason": "adv_cap_zero_qty",
                        "policy_mode": policy.get("mode"),
                        "policy_id": policy.get("policy_id"),
                    }
                )
                continue
            qty = min(qty, qty_adv)

        notional = qty * fill_price
        entry_cost = _transaction_cost(
            notional=notional,
            side="BUY",
            settings=settings,
            state_settings=state_settings,
            override_cost_model=policy.get("cost_model"),
        )
        if notional + entry_cost > state.cash:
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "reason": "insufficient_cash",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        order = PaperOrder(
            symbol=symbol,
            side="BUY",
            qty=qty,
            fill_price=fill_price,
            status="FILLED",
            reason="SIGNAL",
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        position = PaperPosition(
            symbol=symbol,
            qty=qty,
            avg_price=fill_price,
            stop_price=max(0.0, fill_price - stop_distance),
            target_price=(
                float(signal.get("target_price"))
                if isinstance(signal.get("target_price"), (int, float))
                and float(signal.get("target_price")) > 0
                else None
            ),
            opened_at=_utc_now(),
        )
        state.cash -= notional + entry_cost
        entry_cost_total += entry_cost
        executed_count += 1

        session.add(order)
        session.add(position)
        _log(
            session,
            "signal_selected",
            {
                "symbol": symbol,
                "template": template,
                "side": "BUY",
                "qty": qty,
                "fill_price": fill_price,
                "signal_strength": float(signal.get("signal_strength", 0.0)),
                "selection_reason": policy.get("selection_reason"),
                "policy_mode": policy.get("mode"),
                "policy_id": policy.get("policy_id"),
            },
        )
        _log(
            session,
            "paper_buy",
            {
                "symbol": symbol,
                "qty": qty,
                "fill_price": fill_price,
                "entry_cost": entry_cost,
                "selection_reason": policy.get("selection_reason"),
                "policy_mode": policy.get("mode"),
            },
        )
        current_positions.append(position)
        open_symbols.add(symbol)

    mark_prices = payload.get("mark_prices", {})
    for position in list(current_positions):
        if position.symbol not in mark_prices:
            continue
        mark = float(mark_prices[position.symbol])
        stop_hit = position.stop_price is not None and mark <= position.stop_price
        target_hit = position.target_price is not None and mark >= position.target_price
        if not stop_hit and not target_hit:
            continue

        reason = "STOP_HIT" if stop_hit else "EXITED"
        order = PaperOrder(
            symbol=position.symbol,
            side="SELL",
            qty=position.qty,
            fill_price=mark,
            status=reason,
            reason=reason,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        exit_notional = position.qty * mark
        exit_cost = _transaction_cost(
            notional=exit_notional,
            side="SELL",
            settings=settings,
            state_settings=state_settings,
            override_cost_model=policy.get("cost_model"),
        )
        state.cash += exit_notional - exit_cost
        exit_cost_total += exit_cost
        session.add(order)
        session.delete(position)
        _log(
            session,
            "paper_sell",
            {
                "symbol": position.symbol,
                "qty": position.qty,
                "fill_price": mark,
                "exit_cost": exit_cost,
                "reason": reason,
            },
        )

    live_positions = get_positions(session)
    mtm = 0.0
    for position in live_positions:
        mark = float(mark_prices.get(position.symbol, position.avg_price))
        mtm += position.qty * mark

    state.equity = state.cash + mtm
    state.peak_equity = max(state.peak_equity, state.equity)
    state.drawdown = (state.equity / state.peak_equity - 1.0) if state.peak_equity > 0 else 0.0

    dd_limit = float(state.settings_json.get("kill_switch_dd", settings.kill_switch_drawdown))
    if state.drawdown <= -dd_limit:
        state.kill_switch_active = True
        state.cooldown_days_left = settings.kill_switch_cooldown_days
        _log(session, "kill_switch_activated", {"drawdown": state.drawdown, "limit": dd_limit})

    session.add(state)
    session.commit()
    session.refresh(state)

    for skip in skipped_signals:
        _log(session, "signal_skipped", skip)
        _log(session, "paper_skip", skip)
    session.commit()

    return jsonable_encoder(
        {
            "status": "ok",
            "regime": regime,
            "policy": policy,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "risk_scaled": bool(
                float(policy["risk_per_trade"]) < settings.risk_per_trade
                or int(policy["max_positions"]) < settings.max_positions
            ),
            "signals_source": signals_source,
            "generated_signals_count": generated_signals_count,
            "selected_signals_count": executed_count,
            "selected_signals": selected_signals,
            "skipped_signals": skipped_signals,
            "cost_summary": {
                "entry_cost_total": entry_cost_total,
                "exit_cost_total": exit_cost_total,
                "total_cost": entry_cost_total + exit_cost_total,
                "cost_model_enabled": bool(
                    policy.get("cost_model", {}).get(
                        "enabled",
                        state_settings.get("cost_model_enabled", settings.cost_model_enabled),
                    )
                ),
                "cost_mode": str(
                    policy.get("cost_model", {}).get(
                        "mode",
                        state_settings.get("cost_mode", settings.cost_mode),
                    )
                ),
            },
            "state": _dump_model(state),
            "positions": [_dump_model(p) for p in get_positions(session)],
            "orders": [_dump_model(o) for o in get_orders(session)],
        }
    )


def preview_policy_signals(
    session: Session,
    settings: Settings,
    payload: dict[str, Any],
    store: DataStore | None = None,
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    regime = str(payload.get("regime", "TREND_UP"))
    policy_override: int | None = None
    try:
        if payload.get("policy_id") is not None:
            policy_override = int(payload.get("policy_id"))
    except (TypeError, ValueError):
        policy_override = None
    policy = _resolve_execution_policy(
        session,
        state,
        settings,
        regime,
        policy_override_id=policy_override,
    )
    store = store or DataStore(parquet_root=settings.parquet_root, duckdb_path=settings.duckdb_path)

    timeframes = _resolve_timeframes(payload, policy)
    dataset_id = _resolve_dataset_id(session, payload, policy, timeframes)
    if dataset_id is None:
        return {
            "regime": regime,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "signals_source": "generated",
            "signals": [],
            "generated_signals_count": 0,
            "selected_signals_count": 0,
            "skipped_signals": [
                {"reason": "dataset_not_found", "message": "No dataset available for preview."}
            ],
        }

    symbol_scope = _resolve_symbol_scope(payload, policy)
    max_symbols_scan = _resolve_max_symbols_scan(payload, policy)
    seed = _resolve_seed(payload, policy)
    signals = generate_signals_for_policy(
        session=session,
        store=store,
        dataset_id=dataset_id,
        asof=_parse_asof(payload.get("asof")),
        timeframes=timeframes,
        allowed_templates=policy.get("allowed_templates", []),
        params_overrides=policy.get("params", {}),
        max_symbols_scan=max_symbols_scan,
        seed=seed,
        mode="preview",
        symbol_scope=symbol_scope,
    )

    return {
        "regime": regime,
        "policy_mode": policy.get("mode"),
        "policy_selection_reason": policy.get("selection_reason"),
        "signals_source": "generated",
        "generated_signals_count": len(signals),
        "selected_signals_count": 0,
        "signals": signals,
        "dataset_id": dataset_id,
        "timeframes": timeframes,
        "symbol_scope": symbol_scope,
    }


def get_paper_state_payload(session: Session, settings: Settings) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    return {
        "state": _dump_model(state),
        "positions": [_dump_model(p) for p in get_positions(session)],
        "orders": [_dump_model(o) for o in get_orders(session)],
    }


def update_runtime_settings(
    session: Session, settings: Settings, payload: dict[str, Any]
) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    merged = dict(state.settings_json)
    merged.update(payload)
    state.settings_json = merged
    session.add(state)
    session.commit()
    session.refresh(state)
    _log(session, "settings_updated", {"keys": sorted(payload.keys())})
    session.commit()
    return {"settings": state.settings_json}


def activate_policy_mode(session: Session, settings: Settings, policy: Policy) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    merged = dict(state.settings_json or {})
    merged["paper_mode"] = "policy"
    merged["active_policy_id"] = policy.id
    merged["active_policy_name"] = policy.name
    state.settings_json = merged
    session.add(state)
    _log(
        session,
        "policy_promoted_to_paper",
        {"policy_id": policy.id, "policy_name": policy.name, "paper_mode": "policy"},
    )
    session.commit()
    session.refresh(state)
    return {
        "paper_mode": "policy",
        "active_policy_id": policy.id,
        "active_policy_name": policy.name,
    }
