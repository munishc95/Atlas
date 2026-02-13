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
    Instrument,
    PolicyHealthSnapshot,
    PaperRun,
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
from app.services.policy_health import (
    DEGRADED,
    HEALTHY,
    PAUSED,
    RETIRED,
    WARNING,
    apply_policy_health_actions,
    get_policy_health_snapshot,
    latest_policy_health_snapshot_for_policy,
    policy_status,
    select_fallback_policy,
)
from app.services.reports import generate_daily_report
from app.services.regime import regime_policy

IST_ZONE = ZoneInfo("Asia/Kolkata")
FUTURE_KINDS = {"STOCK_FUT", "INDEX_FUT"}


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
            "reports_auto_generate_daily": settings.reports_auto_generate_daily,
            "health_window_days_short": settings.health_window_days_short,
            "health_window_days_long": settings.health_window_days_long,
            "drift_maxdd_multiplier": settings.drift_maxdd_multiplier,
            "drift_negative_return_cost_ratio_threshold": settings.drift_negative_return_cost_ratio_threshold,
            "drift_win_rate_drop_pct": settings.drift_win_rate_drop_pct,
            "drift_return_delta_threshold": settings.drift_return_delta_threshold,
            "drift_warning_risk_scale": settings.drift_warning_risk_scale,
            "drift_degraded_risk_scale": settings.drift_degraded_risk_scale,
            "drift_degraded_action": settings.drift_degraded_action,
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
            "futures_initial_margin_pct": settings.futures_initial_margin_pct,
            "futures_symbol_mapping_strategy": settings.futures_symbol_mapping_strategy,
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


def _position_size_lots(
    equity: float,
    risk_per_trade: float,
    stop_distance: float,
    lot_size: int,
) -> int:
    if stop_distance <= 0 or lot_size <= 0:
        return 0
    risk_amount = equity * risk_per_trade
    return int(np.floor(risk_amount / (stop_distance * lot_size)))


def _is_futures_kind(instrument_kind: str) -> bool:
    return str(instrument_kind).upper() in FUTURE_KINDS


def _futures_margin_pct(state_settings: dict[str, Any], settings: Settings) -> float:
    return max(
        0.0,
        float(state_settings.get("futures_initial_margin_pct", settings.futures_initial_margin_pct)),
    )


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
    asof_date: date | None = None,
) -> dict[str, Any]:
    state_settings = state.settings_json or {}
    base_risk = float(state_settings.get("risk_per_trade", settings.risk_per_trade))
    base_max_positions = int(state_settings.get("max_positions", settings.max_positions))
    base = regime_policy(regime, base_risk, base_max_positions)
    resolved: dict[str, Any] = {
        **base,
        "mode": "strategy",
        "policy_id": None,
        "policy_name": None,
        "policy_definition": {},
        "selection_reason": f"Default regime policy for {regime}.",
        "policy_status": "ACTIVE",
        "health_status": HEALTHY,
        "health_reasons": [],
        "risk_scale_override": 1.0,
        "params": {},
        "cost_model": {},
        "allowed_instruments": {
            "BUY": ["EQUITY_CASH", "STOCK_FUT", "INDEX_FUT"],
            "SELL": ["EQUITY_CASH", "STOCK_FUT", "INDEX_FUT"],
        },
        "ranking_weights": {"signal": 1.0},
    }

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

    selected_policy = policy
    health_status = HEALTHY
    health_reasons: list[str] = []
    risk_scale_override = 1.0
    selection_reasons: list[str] = []
    short_window = max(
        5,
        int(
            state_settings.get(
                "health_window_days_short",
                settings.health_window_days_short,
            )
        ),
    )
    for _ in range(3):
        current_status = policy_status(selected_policy)
        if current_status in {PAUSED, RETIRED}:
            fallback = select_fallback_policy(
                session,
                current_policy_id=int(selected_policy.id),
                regime=regime,
            )
            if fallback is None:
                policy_definition = (
                    selected_policy.definition_json
                    if isinstance(selected_policy.definition_json, dict)
                    else {}
                )
                return {
                    "allowed_templates": [],
                    "risk_per_trade": 0.0,
                    "max_positions": 0,
                    "mode": "policy",
                    "policy_id": selected_policy.id,
                    "policy_name": selected_policy.name,
                    "policy_definition": policy_definition,
                    "selection_reason": (
                        f"Policy {selected_policy.name} is {current_status}; "
                        "no fallback policy available. Running shadow-only mode."
                    ),
                    "policy_status": current_status,
                    "health_status": current_status,
                    "health_reasons": [f"Policy status is {current_status}."],
                    "risk_scale_override": 0.0,
                    "params": {},
                    "cost_model": dict(policy_definition.get("cost_model", {})),
                    "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
                    "ranking_weights": dict(policy_definition.get("ranking", {}).get("weights", {})),
                }
            selection_reasons.append(
                f"Policy {selected_policy.name} is {current_status}; fallback selected {fallback.name}."
            )
            selected_policy = fallback
            continue

        latest_snapshot = latest_policy_health_snapshot_for_policy(
            session,
            policy_id=int(selected_policy.id),
            window_days=short_window,
            asof_date=asof_date,
        )
        if latest_snapshot is not None:
            health_status = latest_snapshot.status
            health_reasons = list(latest_snapshot.reasons_json or [])
        else:
            health_status = HEALTHY
            health_reasons = []

        if health_status == DEGRADED:
            fallback = select_fallback_policy(
                session,
                current_policy_id=int(selected_policy.id),
                regime=regime,
            )
            if fallback is None:
                policy_definition = (
                    selected_policy.definition_json
                    if isinstance(selected_policy.definition_json, dict)
                    else {}
                )
                return {
                    "allowed_templates": [],
                    "risk_per_trade": 0.0,
                    "max_positions": 0,
                    "mode": "policy",
                    "policy_id": selected_policy.id,
                    "policy_name": selected_policy.name,
                    "policy_definition": policy_definition,
                    "selection_reason": (
                        f"Policy {selected_policy.name} health is DEGRADED and no fallback exists. "
                        "Running shadow-only mode."
                    ),
                    "policy_status": policy_status(selected_policy),
                    "health_status": health_status,
                    "health_reasons": health_reasons,
                    "risk_scale_override": 0.0,
                    "params": {},
                    "cost_model": dict(policy_definition.get("cost_model", {})),
                    "allowed_instruments": dict(policy_definition.get("allowed_instruments", {})),
                    "ranking_weights": dict(policy_definition.get("ranking", {}).get("weights", {})),
                }
            selection_reasons.append(
                f"Policy {selected_policy.name} health is DEGRADED; fallback selected {fallback.name}."
            )
            selected_policy = fallback
            continue

        if health_status == WARNING:
            risk_scale_override = max(
                0.0,
                float(
                    state_settings.get(
                        "drift_warning_risk_scale",
                        settings.drift_warning_risk_scale,
                    )
                ),
            )
        break

    policy = selected_policy
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
            "selection_reason": " ".join(
                selection_reasons + [f"Policy {policy.name} blocks new entries in {regime}."]
            ),
            "policy_status": policy_status(policy),
            "health_status": health_status,
            "health_reasons": health_reasons,
            "risk_scale_override": risk_scale_override,
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
    selection_reason = str(
        regime_config.get("reason", f"Policy {policy.name} selected for regime {regime}.")
    )
    if selection_reasons:
        selection_reason = " ".join(selection_reasons + [selection_reason])
    return {
        "allowed_templates": allowed_templates,
        "risk_per_trade": max(0.0, base_risk * risk_scale * risk_scale_override),
        "max_positions": max(0, int(round(base_max_positions * max_positions_scale))),
        "mode": "policy",
        "policy_id": policy.id,
        "policy_name": policy.name,
        "policy_definition": policy_definition,
        "selection_reason": selection_reason,
        "policy_status": policy_status(policy),
        "health_status": health_status,
        "health_reasons": health_reasons,
        "risk_scale_override": risk_scale_override,
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


def _reason_histogram(rows: list[dict[str, Any]], key: str = "reason") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get(key, "")).strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _selection_reason_histogram(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("instrument_choice_reason", row.get("selection_reason", ""))).strip()
        if not reason:
            continue
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _positions_notional(positions: list[PaperPosition]) -> float:
    total = 0.0
    for position in positions:
        total += float(position.qty) * float(position.avg_price)
    return total


def _average_holding_days(positions: list[PaperPosition], asof: datetime) -> float:
    if not positions:
        return 0.0
    days: list[float] = []
    for position in positions:
        opened = position.opened_at
        if opened.tzinfo is None:
            opened = opened.replace(tzinfo=timezone.utc)
        delta = asof - opened
        days.append(max(0.0, delta.total_seconds() / 86_400))
    return float(np.mean(days)) if days else 0.0


def _mark_to_market_component(position: PaperPosition, mark: float) -> float:
    if _is_futures_kind(position.instrument_kind):
        margin_reserved = float(getattr(position, "margin_reserved", 0.0) or 0.0)
        if position.side.upper() == "SELL":
            pnl = (position.avg_price - mark) * position.qty
        else:
            pnl = (mark - position.avg_price) * position.qty
        return margin_reserved + pnl
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
    qty_lots = max(1, int(getattr(position, "qty_lots", max(1, position.qty // max(1, position.lot_size)))))
    is_futures = _is_futures_kind(position.instrument_kind)
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

    if is_futures:
        entry_notional = position.qty * position.avg_price
        margin_reserved = float(getattr(position, "margin_reserved", 0.0) or 0.0)
        pnl = (entry_notional - notional) if side == "SELL" else (notional - entry_notional)
        state.cash += margin_reserved + pnl - exit_cost
    elif side == "SELL":
        state.cash -= notional + exit_cost
    else:
        state.cash += notional - exit_cost

    session.add(
        PaperOrder(
            symbol=position.symbol,
            side=exit_side,
            instrument_kind=position.instrument_kind,
            lot_size=position.lot_size,
            qty_lots=qty_lots,
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
            "qty_lots": qty_lots,
            "margin_released": float(getattr(position, "margin_reserved", 0.0) or 0.0),
        },
    )
    return exit_cost


def run_paper_step(
    session: Session,
    settings: Settings,
    payload: dict[str, Any],
    store: DataStore | None = None,
) -> dict[str, Any]:
    store = store or DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    state = get_or_create_paper_state(session, settings)
    regime = str(payload.get("regime", "TREND_UP"))
    state_settings = state.settings_json or {}
    base_risk_per_trade = float(state_settings.get("risk_per_trade", settings.risk_per_trade))
    base_max_positions = int(state_settings.get("max_positions", settings.max_positions))
    asof_dt = _asof_datetime(payload)
    policy = _resolve_execution_policy(
        session,
        state,
        settings,
        regime,
        asof_date=asof_dt.date(),
    )
    mark_prices = payload.get("mark_prices", {})

    positions_before = get_positions(session)
    orders_before = get_orders(session)
    positions_before_by_id = {
        int(row.id): row for row in positions_before if row.id is not None
    }
    position_ids_before = {int(row.id) for row in positions_before if row.id is not None}
    order_ids_before = {int(row.id) for row in orders_before if row.id is not None}
    equity_before = float(state.equity)
    cash_before = float(state.cash)
    drawdown_before = float(state.drawdown)
    mtm_before = 0.0
    for position in positions_before:
        mark = float(mark_prices.get(position.symbol, position.avg_price))
        mtm_before += _mark_to_market_component(position, mark)

    if state.kill_switch_active:
        _log(session, "kill_switch", {"active": True})
        run_row = PaperRun(
            bundle_id=None,
            policy_id=policy.get("policy_id"),
            asof_ts=asof_dt,
            regime=regime,
            signals_source="provided",
            generated_signals_count=0,
            selected_signals_count=0,
            skipped_signals_count=0,
            scanned_symbols=0,
            evaluated_candidates=0,
            scan_truncated=False,
            summary_json={
                "status": "kill_switch_active",
                "equity_before": equity_before,
                "equity_after": float(state.equity),
                "cash_before": cash_before,
                "cash_after": float(state.cash),
                "drawdown_before": drawdown_before,
                "drawdown": float(state.drawdown),
                "selected_reason_histogram": {},
                "skipped_reason_histogram": {},
                "positions_before": len(positions_before),
                "positions_after": len(positions_before),
                "positions_opened": 0,
                "positions_closed": 0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "net_pnl": 0.0,
                "gross_pnl": 0.0,
                "trade_count": 0,
                "turnover": 0.0,
                "exposure": float(_positions_notional(positions_before) / max(1e-9, state.equity)),
                "avg_holding_days": _average_holding_days(positions_before, asof_dt),
                "kill_switch_active": True,
            },
            cost_summary_json={
                "entry_cost_total": 0.0,
                "exit_cost_total": 0.0,
                "total_cost": 0.0,
                "entry_slippage_cost": 0.0,
                "exit_slippage_cost": 0.0,
            },
        )
        session.add(run_row)
        session.commit()
        session.refresh(run_row)
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
                "paper_run_id": run_row.id,
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

    if not resolved_timeframes:
        resolved_timeframes = _resolve_timeframes(payload, policy)
    if resolved_bundle_id is None:
        resolved_bundle_id = _resolve_bundle_id(session, payload, policy)
    if resolved_dataset_id is None:
        resolved_dataset_id = _resolve_dataset_id(session, payload, policy, resolved_timeframes)
    primary_timeframe = resolved_timeframes[0] if resolved_timeframes else "1d"

    current_positions = list(positions_before)
    open_symbols = {p.symbol for p in current_positions}
    open_underlyings = {
        str((p.metadata_json or {}).get("underlying_symbol", p.symbol)).upper()
        for p in current_positions
    }
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
    executed_signals: list[dict[str, Any]] = []

    candidates = sorted(
        [dict(item) for item in provided_signals],
        key=lambda item: float(item.get("signal_strength", 0.0)),
        reverse=True,
    )
    candidate_symbols = {
        str(item.get("symbol", "")).upper() for item in candidates if str(item.get("symbol", "")).strip()
    }
    candidate_symbols.update(open_underlyings)
    sectors = {
        row.symbol: (row.sector or "UNKNOWN")
        for row in session.exec(
            select(Symbol).where(
                Symbol.symbol.in_(list(candidate_symbols))
            )
        ).all()
    }
    sector_counts: dict[str, int] = {}
    for pos in current_positions:
        underlying = str((pos.metadata_json or {}).get("underlying_symbol", pos.symbol)).upper()
        sector = sectors.get(underlying, sectors.get(pos.symbol, "UNKNOWN"))
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    selected_symbols = set(open_symbols)
    selected_underlyings = set(open_underlyings)

    for signal in candidates:
        symbol = str(signal.get("symbol", "")).upper()
        side = str(signal.get("side", "BUY")).upper()
        template = str(signal.get("template", "trend_breakout"))
        requested_kind = str(signal.get("instrument_kind", "EQUITY_CASH")).upper()
        underlying_symbol = str(signal.get("underlying_symbol", symbol)).upper()
        allowed_set: set[str] | None = None
        if isinstance(policy_allowed_instruments, dict):
            allowed_for_side = policy_allowed_instruments.get(side)
            if isinstance(allowed_for_side, list) and allowed_for_side:
                allowed_set = {str(item).upper() for item in allowed_for_side}
        instrument_kind = requested_kind
        instrument_choice_reason = "provided"
        base_meta = {
            "symbol": symbol,
            "underlying_symbol": underlying_symbol,
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
        if underlying_symbol in selected_underlyings:
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

        if side == "SELL":
            explicit_future = None
            if _is_futures_kind(instrument_kind):
                explicit_future = store.find_instrument(
                    session,
                    symbol=symbol,
                    instrument_kind=instrument_kind,
                )
            if explicit_future is not None:
                underlying_symbol = str(explicit_future.underlying or underlying_symbol).upper()
                signal["symbol"] = explicit_future.symbol
                signal["underlying_symbol"] = underlying_symbol
                signal["instrument_kind"] = explicit_future.kind
                signal["lot_size"] = max(1, int(explicit_future.lot_size))
                instrument_kind = explicit_future.kind
                instrument_choice_reason = "provided_futures_signal"
                if explicit_future.id is not None:
                    fut_frame = store.get_ohlcv_for_instrument(
                        session,
                        instrument_id=int(explicit_future.id),
                        timeframe=primary_timeframe,
                        asof=asof_dt,
                        window=20,
                    )
                    if not fut_frame.empty:
                        fut_bar = fut_frame.iloc[-1]
                        fut_open = float(fut_bar["open"])
                        fut_close = float(fut_bar["close"])
                        signal["price"] = fut_open if fut_open > 0 else fut_close
                        signal["adv"] = float(
                            np.nan_to_num(
                                (fut_frame["close"] * fut_frame["volume"]).tail(20).mean(),
                                nan=0.0,
                            )
                        )
            if explicit_future is not None:
                pass
            else:
                futures_candidate: Instrument | None = None
                if allowed_set is None or "STOCK_FUT" in allowed_set:
                    futures_candidate = store.find_futures_instrument_for_underlying(
                        session,
                        underlying=underlying_symbol,
                        bundle_id=resolved_bundle_id,
                        timeframe=primary_timeframe,
                    )
                if futures_candidate is not None:
                    instrument_kind = futures_candidate.kind
                    symbol = futures_candidate.symbol
                    signal["symbol"] = symbol
                    signal["underlying_symbol"] = underlying_symbol
                    signal["instrument_kind"] = instrument_kind
                    signal["lot_size"] = max(1, int(futures_candidate.lot_size))
                    instrument_choice_reason = "swing_short_requires_futures"
                    if futures_candidate.id is not None:
                        fut_frame = store.get_ohlcv_for_instrument(
                            session,
                            instrument_id=int(futures_candidate.id),
                            timeframe=primary_timeframe,
                            asof=asof_dt,
                            window=20,
                        )
                        if not fut_frame.empty:
                            fut_bar = fut_frame.iloc[-1]
                            fut_open = float(fut_bar["open"])
                            fut_close = float(fut_bar["close"])
                            signal["price"] = fut_open if fut_open > 0 else fut_close
                            signal["adv"] = float(
                                np.nan_to_num(
                                    (fut_frame["close"] * fut_frame["volume"]).tail(20).mean(),
                                    nan=0.0,
                                )
                            )
                elif allowed_set is None or "EQUITY_CASH" in allowed_set:
                    instrument_kind = "EQUITY_CASH"
                    signal["instrument_kind"] = instrument_kind
                    signal["symbol"] = underlying_symbol
                    signal["underlying_symbol"] = underlying_symbol
                    instrument_choice_reason = "cash_intraday_short_fallback"
                else:
                    skipped_signals.append(
                        {
                            **base_meta,
                            "reason": "no_short_instrument_available",
                        }
                    )
                    continue
        else:
            if allowed_set is not None and instrument_kind not in allowed_set:
                if "EQUITY_CASH" in allowed_set:
                    instrument_kind = "EQUITY_CASH"
                    signal["instrument_kind"] = instrument_kind
                else:
                    skipped_signals.append(
                        {
                            **base_meta,
                            "reason": "instrument_blocked_by_policy",
                        }
                    )
                    continue

        if allowed_set is not None and instrument_kind not in allowed_set:
            skipped_signals.append(
                {
                    **base_meta,
                    "instrument_kind": instrument_kind,
                    "reason": "instrument_blocked_by_policy",
                }
            )
            continue

        signal["instrument_kind"] = instrument_kind
        signal["instrument_choice_reason"] = instrument_choice_reason

        sector = sectors.get(underlying_symbol, sectors.get(symbol, "UNKNOWN"))
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
        selected_underlyings.add(underlying_symbol)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1

    entry_cost_total = 0.0
    exit_cost_total = 0.0
    entry_slippage_cost_total = 0.0
    exit_slippage_cost_total = 0.0
    traded_notional_total = 0.0
    executed_count = 0

    for signal in selected_signals:
        symbol = str(signal.get("symbol", "")).upper()
        underlying_symbol = str(signal.get("underlying_symbol", symbol)).upper()
        side = str(signal.get("side", "BUY")).upper()
        template = str(signal.get("template", "trend_breakout"))
        instrument_kind = str(signal.get("instrument_kind", "EQUITY_CASH")).upper()
        lot_size = max(1, int(signal.get("lot_size", 1)))
        is_futures = _is_futures_kind(instrument_kind)
        price = float(signal.get("price", 0.0))
        stop_distance = float(signal.get("stop_distance", 0.0))
        if is_futures:
            qty_lots = _position_size_lots(
                state.equity,
                float(policy["risk_per_trade"]),
                stop_distance,
                lot_size,
            )
            qty = qty_lots * lot_size
        else:
            qty = _position_size(state.equity, float(policy["risk_per_trade"]), stop_distance)
            qty = _adjust_qty_for_lot(qty, lot_size)
            qty_lots = max(1, int(np.floor(qty / max(1, lot_size)))) if qty > 0 else 0
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
            if is_futures:
                qty_adv_lots = int(np.floor(max_notional / max(fill_price * lot_size, 1e-9)))
                if qty_adv_lots <= 0:
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
                qty_lots = min(qty_lots, qty_adv_lots)
                qty = qty_lots * lot_size
            else:
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
                qty_lots = max(1, int(np.floor(qty / max(1, lot_size)))) if qty > 0 else 0

        if qty <= 0 or qty_lots <= 0:
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
        traded_notional_total += float(notional)
        entry_slippage_cost = (
            (fill_price - price) * qty if side == "BUY" else (price - fill_price) * qty
        )
        entry_slippage_cost_total += max(0.0, float(entry_slippage_cost))
        margin_required = (
            notional * _futures_margin_pct(state_settings, settings) if is_futures else 0.0
        )
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
        if is_futures and (margin_required + entry_cost > state.cash):
            skipped_signals.append(
                {
                    "symbol": symbol,
                    "template": template,
                    "side": side,
                    "instrument_kind": instrument_kind,
                    "reason": "insufficient_margin",
                    "policy_mode": policy.get("mode"),
                    "policy_id": policy.get("policy_id"),
                }
            )
            continue
        if (not is_futures) and side == "BUY" and (notional + entry_cost > state.cash):
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
            qty_lots=qty_lots,
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
            if is_futures:
                state.cash -= margin_required + entry_cost
            else:
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
            if is_futures:
                state.cash -= margin_required + entry_cost
            else:
                state.cash += notional - entry_cost

        position = PaperPosition(
            symbol=symbol,
            side=side,
            instrument_kind=instrument_kind,
            lot_size=lot_size,
            qty_lots=qty_lots,
            margin_reserved=margin_required,
            must_exit_by_eod=must_exit_by_eod,
            qty=qty,
            avg_price=fill_price,
            stop_price=stop_price,
            target_price=target_price,
            metadata_json={
                "template": template,
                "underlying_symbol": underlying_symbol,
                "instrument_choice_reason": signal.get("instrument_choice_reason", "provided"),
            },
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
                "underlying_symbol": underlying_symbol,
                "template": template,
                "side": side,
                "instrument_kind": instrument_kind,
                "instrument_choice_reason": signal.get("instrument_choice_reason", "provided"),
                "qty": qty,
                "qty_lots": qty_lots,
                "margin_reserved": margin_required,
                "fill_price": fill_price,
                "signal_strength": float(signal.get("signal_strength", 0.0)),
                "selection_reason": policy.get("selection_reason"),
                "policy_mode": policy.get("mode"),
                "policy_id": policy.get("policy_id"),
                "must_exit_by_eod": must_exit_by_eod,
            },
        )
        executed_signals.append(
            {
                **dict(signal),
                "symbol": symbol,
                "underlying_symbol": underlying_symbol,
                "instrument_kind": instrument_kind,
                "qty": qty,
                "qty_lots": qty_lots,
                "fill_price": fill_price,
                "entry_cost": entry_cost,
            }
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
        open_underlyings.add(underlying_symbol)

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
    for skip in skipped_signals:
        _log(session, "signal_skipped", skip)
        _log(session, "paper_skip", skip)
    session.commit()
    session.refresh(state)

    live_positions = get_positions(session)
    orders_after = get_orders(session)
    position_ids_after = {int(row.id) for row in live_positions if row.id is not None}
    order_ids_after = {int(row.id) for row in orders_after if row.id is not None}
    new_position_ids = sorted(position_ids_after - position_ids_before)
    closed_position_ids = sorted(position_ids_before - position_ids_after)
    new_order_ids = sorted(order_ids_after - order_ids_before)
    closed_position_symbols = [
        positions_before_by_id[item].symbol
        for item in closed_position_ids
        if item in positions_before_by_id
    ]

    selected_reason_histogram = _selection_reason_histogram(executed_signals)
    skipped_reason_histogram = _reason_histogram(skipped_signals)
    total_cost = float(entry_cost_total + exit_cost_total)
    realized_pnl = float(state.cash - cash_before)
    unrealized_pnl = float(mtm - mtm_before)
    net_pnl = float(state.equity - equity_before)
    gross_pnl = float(net_pnl + total_cost)
    turnover = float(traded_notional_total / max(1e-9, equity_before))
    exposure = float(_positions_notional(live_positions) / max(1e-9, state.equity))
    avg_holding_days = _average_holding_days(live_positions, asof_dt)

    cost_summary = {
        "entry_cost_total": float(entry_cost_total),
        "exit_cost_total": float(exit_cost_total),
        "total_cost": total_cost,
        "entry_slippage_cost": float(entry_slippage_cost_total),
        "exit_slippage_cost": float(exit_slippage_cost_total),
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
    }

    run_summary = {
        "policy_mode": policy.get("mode"),
        "policy_selection_reason": policy.get("selection_reason"),
        "policy_status": policy.get("policy_status"),
        "health_status": policy.get("health_status"),
        "signals_source": signals_source,
        "bundle_id": resolved_bundle_id,
        "dataset_id": resolved_dataset_id,
        "timeframes": resolved_timeframes,
        "scan_truncated": bool(generated_meta.scan_truncated),
        "scanned_symbols": int(generated_meta.scanned_symbols),
        "evaluated_candidates": int(generated_meta.evaluated_candidates),
        "total_symbols": int(generated_meta.total_symbols),
        "generated_signals_count": int(generated_signals_count),
        "selected_signals_count": int(executed_count),
        "skipped_signals_count": int(len(skipped_signals)),
        "positions_before": len(positions_before),
        "positions_after": len(live_positions),
        "positions_opened": len(new_position_ids),
        "positions_closed": len(closed_position_ids),
        "new_position_ids": new_position_ids,
        "closed_position_ids": closed_position_ids,
        "closed_position_symbols": closed_position_symbols,
        "new_order_ids": new_order_ids,
        "selected_signals": [
            {
                "symbol": str(item.get("symbol", "")),
                "side": str(item.get("side", "")),
                "instrument_kind": str(item.get("instrument_kind", "")),
                "selection_reason": str(item.get("instrument_choice_reason", "provided")),
            }
            for item in executed_signals
        ],
        "selected_reason_histogram": selected_reason_histogram,
        "skipped_reason_histogram": skipped_reason_histogram,
        "equity_before": equity_before,
        "equity_after": float(state.equity),
        "cash_before": cash_before,
        "cash_after": float(state.cash),
        "drawdown_before": drawdown_before,
        "drawdown": float(state.drawdown),
        "realized_pnl": realized_pnl,
        "unrealized_pnl": unrealized_pnl,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "trade_count": int(len(new_position_ids) + len(closed_position_ids)),
        "turnover": turnover,
        "exposure": exposure,
        "avg_holding_days": avg_holding_days,
        "kill_switch_active": bool(state.kill_switch_active),
    }
    policy_id_value: int | None = None
    try:
        if policy.get("policy_id") is not None:
            policy_id_value = int(policy["policy_id"])
    except (TypeError, ValueError):
        policy_id_value = None
    run_row = PaperRun(
        bundle_id=resolved_bundle_id,
        policy_id=policy_id_value,
        asof_ts=asof_dt,
        regime=regime,
        signals_source=signals_source,
        generated_signals_count=generated_signals_count,
        selected_signals_count=executed_count,
        skipped_signals_count=len(skipped_signals),
        scanned_symbols=int(generated_meta.scanned_symbols),
        evaluated_candidates=int(generated_meta.evaluated_candidates),
        scan_truncated=bool(generated_meta.scan_truncated),
        summary_json=run_summary,
        cost_summary_json=cost_summary,
    )
    session.add(run_row)
    session.commit()
    session.refresh(run_row)

    health_short_payload: dict[str, Any] | None = None
    health_long_payload: dict[str, Any] | None = None
    health_action_payload: dict[str, Any] | None = None

    active_policy: Policy | None = None
    if policy_id_value is not None:
        active_policy = session.get(Policy, policy_id_value)

    if active_policy is not None and policy.get("mode") == "policy":
        short_window = max(
            5,
            int(
                state_settings.get(
                    "health_window_days_short",
                    settings.health_window_days_short,
                )
            ),
        )
        long_window = max(
            short_window,
            int(
                state_settings.get(
                    "health_window_days_long",
                    settings.health_window_days_long,
                )
            ),
        )
        health_short = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=active_policy,
            window_days=short_window,
            asof_date=asof_dt.date(),
            refresh=True,
            overrides=state_settings,
        )
        health_long = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=active_policy,
            window_days=long_window,
            asof_date=asof_dt.date(),
            refresh=True,
            overrides=state_settings,
        )
        action_source: PolicyHealthSnapshot = (
            health_long if health_long.status == DEGRADED else health_short
        )
        health_action_payload = apply_policy_health_actions(
            session,
            settings=settings,
            policy=active_policy,
            snapshot=action_source,
            overrides=state_settings,
        )
        _log(
            session,
            "policy_health_snapshot",
            {
                "policy_id": active_policy.id,
                "short_window_days": short_window,
                "short_status": health_short.status,
                "short_reasons": health_short.reasons_json,
                "long_window_days": long_window,
                "long_status": health_long.status,
                "long_reasons": health_long.reasons_json,
            },
        )
        _log(
            session,
            "policy_health_action",
            {
                "policy_id": active_policy.id,
                "status": action_source.status,
                "action": health_action_payload.get("action"),
                "risk_scale_override": health_action_payload.get("risk_scale_override"),
                "policy_status": health_action_payload.get("policy_status"),
                "reasons": action_source.reasons_json,
            },
        )

        action_name = str(health_action_payload.get("action", "NONE")).upper()
        action_policy_status = str(
            health_action_payload.get("policy_status", policy_status(active_policy))
        ).upper()
        if action_policy_status in {PAUSED, RETIRED}:
            fallback = select_fallback_policy(
                session,
                current_policy_id=int(active_policy.id),
                regime=regime,
            )
            if fallback is not None:
                merged_state = dict(state.settings_json or {})
                if (
                    str(merged_state.get("paper_mode", "strategy")) == "policy"
                    and int(merged_state.get("active_policy_id") or 0) == int(active_policy.id)
                ):
                    merged_state["active_policy_id"] = int(fallback.id)
                    merged_state["active_policy_name"] = fallback.name
                    state.settings_json = merged_state
                    session.add(state)
                    _log(
                        session,
                        "policy_fallback_selected",
                        {
                            "from_policy_id": active_policy.id,
                            "to_policy_id": fallback.id,
                            "from_status": action_policy_status,
                            "reason": "policy_degraded_or_paused",
                            "regime": regime,
                        },
                    )
                    session.commit()
                    session.refresh(state)
            elif action_name != "NONE":
                _log(
                    session,
                    "policy_fallback_unavailable",
                    {
                        "policy_id": active_policy.id,
                        "status": action_policy_status,
                        "regime": regime,
                        "reason": "no_fallback_policy",
                    },
                )
        session.commit()
        health_short_payload = health_short.model_dump()
        health_long_payload = health_long.model_dump()

    generated_report_id: int | None = None
    auto_report_enabled = bool(
        state_settings.get("reports_auto_generate_daily", settings.reports_auto_generate_daily)
    )
    if auto_report_enabled:
        report_row = generate_daily_report(
            session=session,
            settings=settings,
            report_date=asof_dt.date(),
            bundle_id=resolved_bundle_id,
            policy_id=run_row.policy_id,
            overwrite=True,
        )
        generated_report_id = int(report_row.id) if report_row.id is not None else None

    return jsonable_encoder(
        {
            "status": "ok",
            "regime": regime,
            "policy": policy,
            "policy_mode": policy.get("mode"),
            "policy_selection_reason": policy.get("selection_reason"),
            "policy_status": policy.get("policy_status"),
            "policy_health_status": policy.get("health_status"),
            "policy_health_reasons": policy.get("health_reasons", []),
            "risk_scaled": bool(
                float(policy["risk_per_trade"]) < base_risk_per_trade
                or int(policy["max_positions"]) < base_max_positions
            ),
            "paper_run_id": run_row.id,
            "signals_source": signals_source,
            "generated_signals_count": generated_signals_count,
            "selected_signals_count": executed_count,
            "selected_signals": executed_signals,
            "skipped_signals": skipped_signals,
            "scan_truncated": generated_meta.scan_truncated,
            "scanned_symbols": generated_meta.scanned_symbols,
            "evaluated_candidates": generated_meta.evaluated_candidates,
            "total_symbols": generated_meta.total_symbols,
            "bundle_id": resolved_bundle_id,
            "dataset_id": resolved_dataset_id,
            "timeframes": resolved_timeframes,
            "cost_summary": cost_summary,
            "report_id": generated_report_id,
            "health_short": health_short_payload,
            "health_long": health_long_payload,
            "health_action": health_action_payload,
            "state": _dump_model(state),
            "positions": [_dump_model(p) for p in live_positions],
            "orders": [_dump_model(o) for o in orders_after],
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
    preview_asof = _asof_datetime(payload)
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
        asof_date=preview_asof.date(),
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
            "policy_status": policy.get("policy_status"),
            "health_status": policy.get("health_status"),
            "health_reasons": policy.get("health_reasons", []),
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
        asof=preview_asof,
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
        "policy_status": policy.get("policy_status"),
        "health_status": policy.get("health_status"),
        "health_reasons": policy.get("health_reasons", []),
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
