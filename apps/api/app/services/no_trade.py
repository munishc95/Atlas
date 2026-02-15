from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import NoTradeSnapshot, PaperRun
from app.services.data_store import DataStore
from app.services.trading_calendar import list_trading_days


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(result) or np.isinf(result):
        return default
    return result


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_regimes(value: Any, fallback: list[str]) -> set[str]:
    if isinstance(value, list):
        tokens = [str(item).strip().upper() for item in value if str(item).strip()]
        if tokens:
            return set(tokens)
    return {str(item).strip().upper() for item in fallback if str(item).strip()}


def _run_return(row: PaperRun) -> float:
    summary = row.summary_json if isinstance(row.summary_json, dict) else {}
    equity_before = _safe_float(summary.get("equity_before"), 0.0)
    equity_after = _safe_float(summary.get("equity_after"), 0.0)
    if equity_before > 0 and equity_after > 0:
        return (equity_after / equity_before) - 1.0
    net_pnl = _safe_float(summary.get("net_pnl"), 0.0)
    return net_pnl / equity_before if equity_before > 0 else 0.0


def _realized_vol_annual(
    session: Session,
    *,
    bundle_id: int,
    asof_ts: datetime,
    lookback_rows: int = 30,
) -> float:
    rows = list(
        session.exec(
            select(PaperRun)
            .where(PaperRun.bundle_id == int(bundle_id))
            .where(PaperRun.asof_ts <= asof_ts)
            .order_by(PaperRun.asof_ts.desc(), PaperRun.id.desc())
            .limit(max(5, int(lookback_rows)))
        ).all()
    )
    rows.reverse()
    if len(rows) < 2:
        return 0.0
    values = np.array([_run_return(row) for row in rows], dtype=float)
    if values.size < 2:
        return 0.0
    sigma = float(np.nan_to_num(np.std(values, ddof=1), nan=0.0))
    return max(0.0, sigma * np.sqrt(252.0))


def _breadth_pct(
    session: Session,
    *,
    store: DataStore,
    bundle_id: int,
    asof_ts: datetime,
) -> float:
    symbols = store.get_bundle_symbols(session, bundle_id, timeframe="1d")
    if not symbols:
        return 100.0
    valid = 0
    above = 0
    for symbol in symbols:
        features = store.load_features(symbol=symbol, timeframe="1d", end=asof_ts)
        if features.empty:
            continue
        latest = features.iloc[-1]
        close = _safe_float(latest.get("close"), 0.0)
        ema_50 = _safe_float(latest.get("ema_50"), 0.0)
        if close <= 0 or ema_50 <= 0:
            continue
        valid += 1
        if close > ema_50:
            above += 1
    if valid == 0:
        return 100.0
    return float(100.0 * above / valid)


def _trend_strength(
    session: Session,
    *,
    store: DataStore,
    bundle_id: int,
    asof_ts: datetime,
) -> float:
    symbols = store.get_bundle_symbols(session, bundle_id, timeframe="1d")
    if not symbols:
        return 0.0
    selected = "NIFTY500" if "NIFTY500" in {item.upper() for item in symbols} else symbols[0]
    features = store.load_features(symbol=selected, timeframe="1d", end=asof_ts)
    if features.empty:
        return 0.0
    latest = features.iloc[-1]
    adx_14 = _safe_float(latest.get("adx_14"), 0.0)
    if adx_14 > 0:
        return adx_14
    close = _safe_float(latest.get("close"), 0.0)
    ema_20 = _safe_float(latest.get("ema_20"), 0.0)
    ema_50 = _safe_float(latest.get("ema_50"), 0.0)
    if close <= 0:
        return 0.0
    return abs((ema_20 - ema_50) / close) * 100.0


def _latest_snapshot(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
) -> NoTradeSnapshot | None:
    return session.exec(
        select(NoTradeSnapshot)
        .where(NoTradeSnapshot.bundle_id == int(bundle_id))
        .where(NoTradeSnapshot.timeframe == str(timeframe))
        .order_by(NoTradeSnapshot.ts.desc(), NoTradeSnapshot.id.desc())
    ).first()


def evaluate_no_trade_gate(
    session: Session,
    *,
    settings: Settings,
    store: DataStore,
    bundle_id: int | None,
    timeframe: str,
    asof_ts: datetime,
    regime: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bundle_key = int(bundle_id) if isinstance(bundle_id, int) and bundle_id > 0 else 0
    scope = dict(overrides or {})
    enabled = bool(scope.get("no_trade_enabled", settings.no_trade_enabled))
    configured_regimes = _safe_regimes(
        scope.get("no_trade_regimes"),
        list(settings.no_trade_regimes),
    )
    max_realized_vol = _safe_float(
        scope.get("no_trade_max_realized_vol_annual"),
        settings.no_trade_max_realized_vol_annual,
    )
    min_breadth_pct = _safe_float(
        scope.get("no_trade_min_breadth_pct"),
        settings.no_trade_min_breadth_pct,
    )
    min_trend_strength = _safe_float(
        scope.get("no_trade_min_trend_strength"),
        settings.no_trade_min_trend_strength,
    )
    cooldown_days = max(
        0,
        _safe_int(
            scope.get("no_trade_cooldown_trading_days"),
            settings.no_trade_cooldown_trading_days,
        ),
    )
    if not enabled:
        return {
            "triggered": False,
            "reasons": [],
            "breadth_pct": 100.0,
            "realized_vol": 0.0,
            "trend_strength": 0.0,
            "cooldown_remaining": 0,
            "enabled": False,
        }

    breadth_pct = 100.0
    trend_strength = 0.0
    if bundle_key > 0:
        breadth_pct = _breadth_pct(session, store=store, bundle_id=bundle_key, asof_ts=asof_ts)
        trend_strength = _trend_strength(
            session,
            store=store,
            bundle_id=bundle_key,
            asof_ts=asof_ts,
        )
    realized_vol = _realized_vol_annual(session, bundle_id=bundle_key, asof_ts=asof_ts)
    reasons: list[str] = []
    regime_token = str(regime or "").strip().upper()
    if regime_token in configured_regimes:
        reasons.append("regime_blocked")
    if bundle_key > 0:
        if realized_vol > max_realized_vol:
            reasons.append("realized_vol_above_threshold")
        if breadth_pct < min_breadth_pct:
            reasons.append("breadth_below_threshold")
        if trend_strength < min_trend_strength:
            reasons.append("trend_strength_below_threshold")

    if bundle_key <= 0:
        triggered = bool(reasons)
        cooldown_remaining = int(cooldown_days) if triggered else 0
        return {
            "id": None,
            "enabled": True,
            "triggered": bool(triggered),
            "reasons": list(dict.fromkeys(reasons)),
            "breadth_pct": float(breadth_pct),
            "realized_vol": float(realized_vol),
            "trend_strength": float(trend_strength),
            "cooldown_remaining": cooldown_remaining,
            "regime": regime_token or None,
            "ts": asof_ts.isoformat(),
        }

    latest = _latest_snapshot(session, bundle_id=bundle_key, timeframe=timeframe)
    cooldown_remaining = 0
    if latest is not None and int(latest.cooldown_remaining) > 0:
        days_passed = max(
            0,
            len(
                list_trading_days(
                    start_date=latest.ts.astimezone(timezone.utc).date(),
                    end_date=asof_ts.astimezone(timezone.utc).date(),
                    segment=str(
                        scope.get("trading_calendar_segment", settings.trading_calendar_segment)
                    ),
                    settings=settings,
                )
            )
            - 1,
        )
        cooldown_remaining = max(0, int(latest.cooldown_remaining) - days_passed)

    triggered = bool(reasons)
    if triggered:
        cooldown_remaining = max(cooldown_remaining, cooldown_days)
    elif cooldown_remaining > 0:
        triggered = True
        reasons.append("cooldown_active")

    snapshot = NoTradeSnapshot(
        ts=asof_ts,
        bundle_id=bundle_key,
        timeframe=str(timeframe),
        regime=regime_token or None,
        triggered=bool(triggered),
        reasons_json=list(dict.fromkeys(reasons)),
        breadth_pct=float(breadth_pct),
        realized_vol=float(realized_vol),
        trend_strength=float(trend_strength),
        cooldown_remaining=int(cooldown_remaining),
    )
    session.add(snapshot)
    session.commit()
    session.refresh(snapshot)

    return {
        "id": int(snapshot.id) if snapshot.id is not None else None,
        "enabled": True,
        "triggered": bool(snapshot.triggered),
        "reasons": list(snapshot.reasons_json or []),
        "breadth_pct": float(snapshot.breadth_pct),
        "realized_vol": float(snapshot.realized_vol),
        "trend_strength": float(snapshot.trend_strength),
        "cooldown_remaining": int(snapshot.cooldown_remaining),
        "regime": snapshot.regime,
        "ts": snapshot.ts.isoformat(),
    }
