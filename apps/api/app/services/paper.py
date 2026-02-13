from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi.encoders import jsonable_encoder
import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import AuditLog, PaperOrder, PaperPosition, PaperState, Policy, Symbol
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


def _resolve_execution_policy(
    session: Session,
    state: PaperState,
    settings: Settings,
    regime: str,
) -> dict[str, Any]:
    base = regime_policy(regime, settings.risk_per_trade, settings.max_positions)
    resolved: dict[str, Any] = {
        **base,
        "mode": "strategy",
        "policy_id": None,
        "policy_name": None,
        "selection_reason": f"Default regime policy for {regime}.",
    }

    state_settings = state.settings_json or {}
    if str(state_settings.get("paper_mode", "strategy")) != "policy":
        return resolved

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

    regime_map = policy.definition_json.get("regime_map", {})
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
            "selection_reason": f"Policy {policy.name} blocks new entries in {regime}.",
            "params": {},
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
        "selection_reason": str(
            regime_config.get("reason", f"Policy {policy.name} selected for regime {regime}.")
        ),
        "params": dict(regime_config.get("params", {})),
    }


def run_paper_step(session: Session, settings: Settings, payload: dict[str, Any]) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    regime = str(payload.get("regime", "TREND_UP"))
    policy = _resolve_execution_policy(session, state, settings, regime)

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
                "positions": [p.model_dump() for p in get_positions(session)],
                "orders": [o.model_dump() for o in get_orders(session)],
            }
        )

    current_positions = get_positions(session)
    open_symbols = {p.symbol for p in current_positions}
    max_positions = int(policy["max_positions"])
    settings_json = state.settings_json or {}
    correlation_threshold = float(
        settings_json.get("diversification_corr_threshold", settings.diversification_corr_threshold)
    )
    max_position_value_pct_adv = float(
        settings_json.get("max_position_value_pct_adv", settings.max_position_value_pct_adv)
    )
    skipped_signals: list[dict[str, Any]] = []

    candidates = sorted(
        [dict(item) for item in payload.get("signals", [])],
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

    selected: list[dict[str, Any]] = []
    selected_symbols = set(open_symbols)

    for signal in candidates:
        if len(current_positions) + len(selected) >= max_positions:
            break

        symbol = str(signal["symbol"]).upper()
        side = str(signal.get("side", "BUY")).upper()
        if side != "BUY" or symbol in open_symbols:
            skipped_signals.append({"symbol": symbol, "reason": "duplicate_or_non_buy"})
            continue

        template = str(signal.get("template", "trend_breakout"))
        if template not in policy["allowed_templates"]:
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "reason": "template_blocked_by_policy"
                    if policy.get("mode") == "policy"
                    else "template_blocked_by_regime",
                }
            )
            continue

        sector = sectors.get(symbol, "UNKNOWN")
        if sector_counts.get(sector, 0) >= 2:
            skipped_signals.append({"symbol": symbol, "reason": "sector_concentration"})
            continue
        if _correlation_reject(
            signal, selected_symbols=selected_symbols, threshold=correlation_threshold
        ):
            skipped_signals.append({"symbol": symbol, "reason": "correlation_threshold"})
            continue

        selected.append(signal)
        selected_symbols.add(symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    for signal in selected:
        symbol = str(signal["symbol"]).upper()
        price = float(signal.get("price", 0.0))
        stop_distance = float(signal.get("stop_distance", 0.0))
        qty = _position_size(state.equity, float(policy["risk_per_trade"]), stop_distance)
        if price <= 0 or qty <= 0:
            skipped_signals.append({"symbol": symbol, "reason": "invalid_price_or_size"})
            continue

        slippage = (settings.slippage_base_bps / 10_000) * (1 + float(signal.get("vol_scale", 0.0)))
        fill_price = price * (1 + slippage)
        adv_notional = float(signal.get("adv", 0.0))
        if adv_notional > 0 and max_position_value_pct_adv > 0:
            max_notional = adv_notional * max_position_value_pct_adv
            qty_adv = int(np.floor(max_notional / max(fill_price, 1e-9)))
            if qty_adv <= 0:
                skipped_signals.append({"symbol": symbol, "reason": "adv_cap_zero_qty"})
                continue
            qty = min(qty, qty_adv)

        notional = qty * fill_price
        if notional > state.cash:
            skipped_signals.append({"symbol": symbol, "reason": "insufficient_cash"})
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
            target_price=float(signal.get("target_price", 0.0)) or None,
            opened_at=_utc_now(),
        )
        state.cash -= notional

        session.add(order)
        session.add(position)
        _log(
            session,
            "paper_buy",
            {
                "symbol": symbol,
                "qty": qty,
                "fill_price": fill_price,
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
        state.cash += position.qty * mark
        session.add(order)
        session.delete(position)
        _log(
            session,
            "paper_sell",
            {"symbol": position.symbol, "qty": position.qty, "fill_price": mark},
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
            "skipped_signals": skipped_signals,
            "state": state.model_dump(),
            "positions": [p.model_dump() for p in get_positions(session)],
            "orders": [o.model_dump() for o in get_orders(session)],
        }
    )


def get_paper_state_payload(session: Session, settings: Settings) -> dict[str, Any]:
    state = get_or_create_paper_state(session, settings)
    return {
        "state": state.model_dump(),
        "positions": [p.model_dump() for p in get_positions(session)],
        "orders": [o.model_dump() for o in get_orders(session)],
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
