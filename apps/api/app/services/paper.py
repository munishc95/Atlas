from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any
from zoneinfo import ZoneInfo

from fastapi.encoders import jsonable_encoder
import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import (
    AuditLog,
    Dataset,
    DatasetBundle,
    PaperOrder,
    PaperPosition,
    PaperState,
    Policy,
    Symbol,
)
from app.engine.costs import (
    estimate_equity_delivery_cost,
    estimate_futures_cost,
    estimate_intraday_cost,
)
from app.engine.signal_engine import SignalGenerationResult, generate_signals_for_policy
from app.services.data_store import DataStore
from app.services.regime import regime_policy

IST_ZONE = ZoneInfo("Asia/Kolkata")


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
            "allowed_sides": settings.allowed_sides,
            "paper_short_squareoff_time": settings.paper_short_squareoff_time,
            "autopilot_max_symbols_scan": settings.autopilot_max_symbols_scan,
            "autopilot_max_runtime_seconds": settings.autopilot_max_runtime_seconds,
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
            "futures_brokerage_bps": settings.futures_brokerage_bps,
            "futures_stt_sell_bps": settings.futures_stt_sell_bps,
            "futures_exchange_txn_bps": settings.futures_exchange_txn_bps,
            "futures_stamp_buy_bps": settings.futures_stamp_buy_bps,
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


def _transaction_cost(
    *,
    notional: float,
    side: str,
    instrument_kind: str,
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
        kind = instrument_kind.upper()
        if kind in {"STOCK_FUT", "INDEX_FUT"}:
            return estimate_futures_cost(notional=notional, side=side, config=cfg)
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
        "allowed_instruments": {"BUY": ["EQUITY_CASH"], "SELL": ["EQUITY_CASH"]},
        "ranking_weights": {"signal": 1.0},
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
            "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
            "ranking_weights": dict(policy_definition.get("ranking", {}).get("weights", {})),
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
        "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
        "ranking_weights": dict(policy_definition.get("ranking", {}).get("weights", {})),
    }


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


def _resolve_bundle_id(
    session: Session,
    payload: dict[str, Any],
    policy: dict[str, Any],
) -> int | None:
    explicit_id = payload.get("bundle_id")
    if isinstance(explicit_id, int):
        return explicit_id if session.get(DatasetBundle, explicit_id) is not None else None
    try:
        explicit_id_int = int(explicit_id)
        if session.get(DatasetBundle, explicit_id_int) is not None:
            return explicit_id_int
    except (TypeError, ValueError):
        pass

    definition = policy.get("policy_definition", {})
    if isinstance(definition, dict):
        universe = definition.get("universe", {})
        if isinstance(universe, dict):
            bundle_id = universe.get("bundle_id")
            try:
                bundle_id_int = int(bundle_id)
                if session.get(DatasetBundle, bundle_id_int) is not None:
                    return bundle_id_int
            except (TypeError, ValueError):
                pass
    return None


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


def _resolve_max_symbols_scan(
    payload: dict[str, Any],
    policy: dict[str, Any],
    state_settings: dict[str, Any],
    settings: Settings,
) -> int:
    explicit = payload.get("max_symbols_scan")
    if isinstance(explicit, int):
        requested = explicit
    else:
        try:
            requested = int(explicit)
        except (TypeError, ValueError):
            requested = None

    if requested is None:
        definition = policy.get("policy_definition", {})
        if isinstance(definition, dict):
            universe = definition.get("universe", {})
            if isinstance(universe, dict):
                value = universe.get("max_symbols_scan")
                try:
                    requested = int(value)
                except (TypeError, ValueError):
                    requested = None
    if requested is None:
        requested = 50

    hard_cap = int(
        state_settings.get("autopilot_max_symbols_scan", settings.autopilot_max_symbols_scan)
    )
    return max(1, min(max(1, int(requested)), max(1, hard_cap)))


def _resolve_max_runtime_seconds(
    payload: dict[str, Any],
    state_settings: dict[str, Any],
    settings: Settings,
) -> int:
    explicit = payload.get("max_runtime_seconds")
    if isinstance(explicit, int):
        requested = explicit
    else:
        try:
            requested = int(explicit)
        except (TypeError, ValueError):
            requested = None
    if requested is None:
        requested = int(
            state_settings.get(
                "autopilot_max_runtime_seconds", settings.autopilot_max_runtime_seconds
            )
        )
    hard_cap = int(
        state_settings.get("autopilot_max_runtime_seconds", settings.autopilot_max_runtime_seconds)
    )
    return max(1, min(max(1, int(requested)), max(1, hard_cap)))


def _normalize_allowed_sides(state_settings: dict[str, Any], settings: Settings) -> set[str]:
    value = state_settings.get("allowed_sides", settings.allowed_sides)
    sides: list[str] = []
    if isinstance(value, list):
        sides = [str(item).upper().strip() for item in value if str(item).strip()]
    elif isinstance(value, str):
        sides = [item.strip().upper() for item in value.split(",") if item.strip()]
    if not sides:
        sides = ["BUY"]
    return set(sides)


def _parse_cutoff(cutoff: str | None, default_cutoff: str) -> time:
    raw = cutoff if isinstance(cutoff, str) and cutoff.strip() else default_cutoff
    try:
        hour, minute = [int(part) for part in raw.split(":", maxsplit=1)]
        return time(hour=max(0, min(23, hour)), minute=max(0, min(59, minute)))
    except Exception:  # noqa: BLE001
        return time(15, 20)


def _asof_datetime(payload: dict[str, Any]) -> datetime:
    asof = _parse_asof(payload.get("asof"))
    if isinstance(asof, date) and not isinstance(asof, datetime):
        return datetime.combine(asof, time(0, 0), tzinfo=timezone.utc)
    if isinstance(asof, datetime):
        return asof if asof.tzinfo is not None else asof.replace(tzinfo=timezone.utc)
    return _utc_now()


def _is_squareoff_due(asof: datetime, cutoff: time) -> bool:
    return asof.astimezone(IST_ZONE).time() >= cutoff


def _adjust_qty_for_lot(qty: int, lot_size: int) -> int:
    if lot_size <= 1:
        return qty
    return (qty // lot_size) * lot_size


def _mark_to_market_component(position: PaperPosition, mark: float) -> float:
    if position.side.upper() == "SELL":
        return -position.qty * mark
    return position.qty * mark


def _position_exit_condition(position: PaperPosition, mark: float) -> tuple[bool, str]:
    if position.side.upper() == "SELL":
        stop_hit = position.stop_price is not None and mark >= position.stop_price
        target_hit = position.target_price is not None and mark <= position.target_price
    else:
        stop_hit = position.stop_price is not None and mark <= position.stop_price
        target_hit = position.target_price is not None and mark >= position.target_price
    if stop_hit:
        return True, "STOP_HIT"
    if target_hit:
        return True, "EXITED"
    return False, ""


def _close_position(
    *,
    session: Session,
    settings: Settings,
    state_settings: dict[str, Any],
    policy: dict[str, Any],
    state: PaperState,
    position: PaperPosition,
    price: float,
    reason: str,
) -> float:
    side = position.side.upper()
    exit_side = "BUY" if side == "SELL" else "SELL"
    notional = position.qty * price
    cost_override = dict(policy.get("cost_model", {}))
    if side == "SELL" and position.instrument_kind == "EQUITY_CASH":
        cost_override["mode"] = "intraday"
    exit_cost = _transaction_cost(
        notional=notional,
        side=exit_side,
        instrument_kind=position.instrument_kind,
        settings=settings,
        state_settings=state_settings,
        override_cost_model=cost_override,
    )

    if side == "SELL":
        state.cash -= notional + exit_cost
    else:
        state.cash += notional - exit_cost

    session.add(
        PaperOrder(
            symbol=position.symbol,
            side=exit_side,
            instrument_kind=position.instrument_kind,
            lot_size=position.lot_size,
            qty=position.qty,
            fill_price=price,
            status=reason,
            reason=reason,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
    )
    session.delete(position)
    _log(
        session,
        "paper_exit",
        {
            "symbol": position.symbol,
            "position_side": side,
            "exit_side": exit_side,
            "qty": position.qty,
            "fill_price": price,
            "exit_cost": exit_cost,
            "reason": reason,
            "instrument_kind": position.instrument_kind,
        },
    )
    return exit_cost


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
    asof_dt = _asof_datetime(payload)

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
    generated_meta = SignalGenerationResult(
        signals=[],
        scan_truncated=False,
        scanned_symbols=0,
        evaluated_candidates=0,
        total_symbols=0,
    )
    resolved_bundle_id: int | None = None
    resolved_dataset_id: int | None = None
    resolved_timeframes: list[str] = []

    if should_generate:
        store = store or DataStore(
            parquet_root=settings.parquet_root,
            duckdb_path=settings.duckdb_path,
            feature_cache_root=settings.feature_cache_root,
        )
        resolved_timeframes = _resolve_timeframes(payload, policy)
        resolved_bundle_id = _resolve_bundle_id(session, payload, policy)
        resolved_dataset_id = _resolve_dataset_id(session, payload, policy, resolved_timeframes)
        symbol_scope = _resolve_symbol_scope(payload, policy)
        max_symbols_scan = _resolve_max_symbols_scan(payload, policy, state_settings, settings)
        max_runtime_seconds = _resolve_max_runtime_seconds(payload, state_settings, settings)
        seed = _resolve_seed(payload, policy)
        ranking_weights = policy.get("ranking_weights", {})

        if resolved_bundle_id is None and resolved_dataset_id is None:
            provided_signals = []
            _log(
                session,
                "signals_generated",
                {
                    "mode": "policy_autopilot",
                    "generated_count": 0,
                    "dataset_id": None,
                    "bundle_id": None,
                    "reason": "dataset_or_bundle_not_found",
                },
            )
        else:
            generated_meta = generate_signals_for_policy(
                session=session,
                store=store,
                dataset_id=resolved_dataset_id,
                bundle_id=resolved_bundle_id,
                asof=asof_dt,
                timeframes=resolved_timeframes,
                allowed_templates=policy.get("allowed_templates", []),
                params_overrides=policy.get("params", {}),
                max_symbols_scan=max_symbols_scan,
                seed=seed,
                mode="paper",
                symbol_scope=symbol_scope,
                ranking_weights=ranking_weights if isinstance(ranking_weights, dict) else None,
                max_runtime_seconds=max_runtime_seconds,
            )
            provided_signals = list(generated_meta.signals)
            generated_signals_count = len(provided_signals)
            _log(
                session,
                "signals_generated",
                {
                    "mode": "policy_autopilot",
                    "dataset_id": resolved_dataset_id,
                    "bundle_id": resolved_bundle_id,
                    "timeframes": resolved_timeframes,
                    "symbol_scope": symbol_scope,
                    "generated_count": generated_signals_count,
                    "allowed_templates": policy.get("allowed_templates", []),
                    "policy_id": policy.get("policy_id"),
                    "policy_name": policy.get("policy_name"),
                    "scan_truncated": generated_meta.scan_truncated,
                    "scanned_symbols": generated_meta.scanned_symbols,
                    "evaluated_candidates": generated_meta.evaluated_candidates,
                    "total_symbols": generated_meta.total_symbols,
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
    allowed_sides = _normalize_allowed_sides(state_settings, settings)
    policy_allowed_instruments = policy.get("allowed_instruments", {})
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
        instrument_kind = str(signal.get("instrument_kind", "EQUITY_CASH")).upper()
        base_meta = {
            "symbol": symbol,
            "template": template,
            "side": side,
            "instrument_kind": instrument_kind,
            "policy_mode": policy.get("mode"),
            "policy_id": policy.get("policy_id"),
            "policy_name": policy.get("policy_name"),
        }

        if len(current_positions) + len(selected_signals) >= max_positions:
            skipped_signals.append({**base_meta, "reason": "max_positions_reached"})
            continue
        if symbol in open_symbols:
            skipped_signals.append({**base_meta, "reason": "already_open"})
            continue
        if side not in {"BUY", "SELL"}:
            skipped_signals.append({**base_meta, "reason": "invalid_side"})
            continue
        if side not in allowed_sides:
            skipped_signals.append(
                {
                    **base_meta,
                    "reason": "shorts_disabled" if side == "SELL" else "side_not_allowed",
                }
            )
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

        if isinstance(policy_allowed_instruments, dict):
            allowed_for_side = policy_allowed_instruments.get(side)
            if isinstance(allowed_for_side, list) and allowed_for_side:
                allowed_set = {str(item).upper() for item in allowed_for_side}
                if instrument_kind not in allowed_set:
                    skipped_signals.append(
                        {
                            **base_meta,
                            "reason": "instrument_blocked_by_policy",
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
        side = str(signal.get("side", "BUY")).upper()
        template = str(signal.get("template", "trend_breakout"))
        instrument_kind = str(signal.get("instrument_kind", "EQUITY_CASH")).upper()
        lot_size = max(1, int(signal.get("lot_size", 1)))
        price = float(signal.get("price", 0.0))
        stop_distance = float(signal.get("stop_distance", 0.0))
        qty = _position_size(state.equity, float(policy["risk_per_trade"]), stop_distance)
        qty = _adjust_qty_for_lot(qty, lot_size)
        if price <= 0 or qty <= 0:
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "invalid_price_or_size",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        slippage = (settings.slippage_base_bps / 10_000) * (1 + float(signal.get("vol_scale", 0.0)))
        fill_price = price * (1 + slippage) if side == "BUY" else price * (1 - slippage)
        adv_notional = float(signal.get("adv", 0.0))
        if adv_notional > 0 and max_position_value_pct_adv > 0:
            max_notional = adv_notional * max_position_value_pct_adv
            qty_adv = int(np.floor(max_notional / max(fill_price, 1e-9)))
            qty_adv = _adjust_qty_for_lot(qty_adv, lot_size)
            if qty_adv <= 0:
                skipped_signals.append(
                    {
                        "symbol": symbol,
                        "template": template,
                        "side": side,
                        "instrument_kind": instrument_kind,
                        "reason": "adv_cap_zero_qty",
                        "policy_mode": policy.get("mode"),
                        "policy_id": policy.get("policy_id"),
                    }
                )
                continue
            qty = min(qty, qty_adv)

        if qty <= 0:
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "qty_zero_after_caps",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        must_exit_by_eod = side == "SELL" and instrument_kind == "EQUITY_CASH"
        notional = qty * fill_price
        cost_override = dict(policy.get("cost_model", {}))
        if must_exit_by_eod:
            cost_override["mode"] = "intraday"
        entry_cost = _transaction_cost(
            notional=notional,
            side=side,
            instrument_kind=instrument_kind,
            settings=settings,
            state_settings=state_settings,
            override_cost_model=cost_override,
        )
        if side == "BUY" and (notional + entry_cost > state.cash):
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "insufficient_cash",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue

        entry_reason = "SIGNAL_SHORT" if side == "SELL" else "SIGNAL"
        order = PaperOrder(
            symbol=symbol,
            side=side,
            instrument_kind=instrument_kind,
            lot_size=lot_size,
            qty=qty,
            fill_price=fill_price,
            status="FILLED",
            reason=entry_reason,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        if side == "BUY":
            stop_price = max(0.0, fill_price - stop_distance)
            if (
                isinstance(signal.get("target_price"), (int, float))
                and float(signal.get("target_price")) > 0
            ):
                target_price = float(signal.get("target_price"))
            else:
                target_price = None
            state.cash -= notional + entry_cost
        else:
            stop_price = fill_price + stop_distance
            if (
                isinstance(signal.get("target_price"), (int, float))
                and float(signal.get("target_price")) > 0
            ):
                target_price = float(signal.get("target_price"))
            else:
                target_price = None
            state.cash += notional - entry_cost

        position = PaperPosition(
            symbol=symbol,
            side=side,
            instrument_kind=instrument_kind,
            lot_size=lot_size,
            must_exit_by_eod=must_exit_by_eod,
            qty=qty,
            avg_price=fill_price,
            stop_price=stop_price,
            target_price=target_price,
            metadata_json={"template": template},
            opened_at=_utc_now(),
        )
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
                "side": side,
                "instrument_kind": instrument_kind,
                "qty": qty,
                "fill_price": fill_price,
                "signal_strength": float(signal.get("signal_strength", 0.0)),
                "selection_reason": policy.get("selection_reason"),
                "policy_mode": policy.get("mode"),
                "policy_id": policy.get("policy_id"),
                "must_exit_by_eod": must_exit_by_eod,
            },
        )
        if must_exit_by_eod:
            _log(
                session,
                "short_intraday_opened",
                {
                    "symbol": symbol,
                    "qty": qty,
                    "fill_price": fill_price,
                    "squareoff_cutoff": state_settings.get(
                        "paper_short_squareoff_time", settings.paper_short_squareoff_time
                    ),
                },
            )
        current_positions.append(position)
        open_symbols.add(symbol)

    mark_prices = payload.get("mark_prices", {})
    for position in list(current_positions):
        if position.symbol not in mark_prices:
            continue
        mark = float(mark_prices[position.symbol])
        should_close, reason = _position_exit_condition(position, mark)
        if not should_close:
            continue
        exit_cost_total += _close_position(
            session=session,
            settings=settings,
            state_settings=state_settings,
            policy=policy,
            state=state,
            position=position,
            price=mark,
            reason=reason,
        )
        if position in current_positions:
            current_positions.remove(position)

    cutoff = _parse_cutoff(
        state_settings.get("paper_short_squareoff_time"),
        settings.paper_short_squareoff_time,
    )
    if _is_squareoff_due(asof_dt, cutoff):
        for position in list(current_positions):
            if not (position.side.upper() == "SELL" and position.must_exit_by_eod):
                continue
            mark = float(mark_prices.get(position.symbol, position.avg_price))
            exit_cost_total += _close_position(
                session=session,
                settings=settings,
                state_settings=state_settings,
                policy=policy,
                state=state,
                position=position,
                price=mark,
                reason="EOD_SQUARE_OFF",
            )
            _log(
                session,
                "forced_squareoff",
                {
                    "symbol": position.symbol,
                    "qty": position.qty,
                    "fill_price": mark,
                    "cutoff": cutoff.isoformat(timespec="minutes"),
                },
            )
            if position in current_positions:
                current_positions.remove(position)

    live_positions = get_positions(session)
    mtm = 0.0
    for position in live_positions:
        mark = float(mark_prices.get(position.symbol, position.avg_price))
        mtm += _mark_to_market_component(position, mark)

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
            "scan_truncated": generated_meta.scan_truncated,
            "scanned_symbols": generated_meta.scanned_symbols,
            "evaluated_candidates": generated_meta.evaluated_candidates,
            "total_symbols": generated_meta.total_symbols,
            "bundle_id": resolved_bundle_id,
            "dataset_id": resolved_dataset_id,
            "timeframes": resolved_timeframes,
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
    store = store or DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )

    timeframes = _resolve_timeframes(payload, policy)
    state_settings = state.settings_json or {}
    bundle_id = _resolve_bundle_id(session, payload, policy)
    dataset_id = _resolve_dataset_id(session, payload, policy, timeframes)
    if bundle_id is None and dataset_id is None:
        return {
            "regime": regime,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "signals_source": "generated",
            "signals": [],
            "generated_signals_count": 0,
            "selected_signals_count": 0,
            "scan_truncated": False,
            "scanned_symbols": 0,
            "evaluated_candidates": 0,
            "total_symbols": 0,
            "skipped_signals": [
                {
                    "reason": "dataset_or_bundle_not_found",
                    "message": "No dataset/bundle available for preview.",
                }
            ],
        }

    symbol_scope = _resolve_symbol_scope(payload, policy)
    max_symbols_scan = _resolve_max_symbols_scan(payload, policy, state_settings, settings)
    max_runtime_seconds = _resolve_max_runtime_seconds(payload, state_settings, settings)
    seed = _resolve_seed(payload, policy)
    generated = generate_signals_for_policy(
        session=session,
        store=store,
        dataset_id=dataset_id,
        bundle_id=bundle_id,
        asof=_asof_datetime(payload),
        timeframes=timeframes,
        allowed_templates=policy.get("allowed_templates", []),
        params_overrides=policy.get("params", {}),
        max_symbols_scan=max_symbols_scan,
        seed=seed,
        mode="preview",
        symbol_scope=symbol_scope,
        ranking_weights=policy.get("ranking_weights", {}),
        max_runtime_seconds=max_runtime_seconds,
    )

    return {
        "regime": regime,
        "policy_mode": policy.get("mode"),
        "policy_selection_reason": policy.get("selection_reason"),
        "signals_source": "generated",
        "generated_signals_count": len(generated.signals),
        "selected_signals_count": 0,
        "signals": generated.signals,
        "bundle_id": bundle_id,
        "dataset_id": dataset_id,
        "timeframes": timeframes,
        "symbol_scope": symbol_scope,
        "scan_truncated": generated.scan_truncated,
        "scanned_symbols": generated.scanned_symbols,
        "evaluated_candidates": generated.evaluated_candidates,
        "total_symbols": generated.total_symbols,
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
