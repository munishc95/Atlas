from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.engine.indicators import atr, sma
from app.services.corporate_actions import list_symbol_actions
from app.services.data_store import DataStore
from app.services.event_risk import evaluate_event_risk
from app.strategies.templates import (
    generate_signal_sides,
    get_template,
    list_templates,
    signal_strength,
)


SignalMode = Literal["paper", "preview", "audit"]

SIGNAL_LOOKBACK_DAYS = 760

DEFAULT_RANKING_WEIGHTS: dict[str, float] = {
    "signal": 0.50,
    "liquidity": 0.25,
    "stability": 0.10,
    "quality": 0.15,
}


@dataclass
class SignalGenerationResult:
    signals: list[dict[str, Any]]
    scan_truncated: bool
    scanned_symbols: int
    evaluated_candidates: int
    total_symbols: int


def _asof_timestamp(value: datetime | date | None) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return pd.Timestamp(value).tz_localize("UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _filtered_frame(frame: pd.DataFrame, asof: pd.Timestamp | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    clean = frame.copy()
    clean["datetime"] = pd.to_datetime(clean["datetime"], utc=True)
    clean = clean.sort_values("datetime")
    if asof is not None:
        clean = clean[clean["datetime"] <= asof]
    return clean.reset_index(drop=True)


def _query_window(asof: pd.Timestamp | None) -> tuple[datetime | None, datetime | None]:
    if asof is None:
        return None, None
    start = (asof - pd.Timedelta(days=SIGNAL_LOOKBACK_DAYS)).to_pydatetime()
    return start, asof.to_pydatetime()


def _minimal_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    clean = frame.copy().sort_values("datetime").reset_index(drop=True)
    atr_14 = atr(clean, period=14)
    close = pd.to_numeric(clean["close"], errors="coerce")
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(clean["datetime"], utc=True),
            "atr_14": atr_14,
            "atr_pct": atr_14 / close.replace(0, np.nan),
        }
    )


def _load_signal_frame(
    *,
    store: DataStore,
    session: Session,
    symbol: str,
    timeframe: str,
    asof: pd.Timestamp | None,
    start: datetime | None,
    end: datetime | None,
) -> pd.DataFrame:
    frame = _filtered_frame(
        store.load_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            session=session,
        ),
        asof,
    )
    if len(frame) >= 50 or start is None:
        return frame
    return _filtered_frame(
        store.load_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            end=end,
            session=session,
        ),
        asof,
    )


def _trade_plan_prices(side: str, entry_price: float, stop_distance: float) -> dict[str, float]:
    normalized_side = str(side or "BUY").strip().upper()
    if entry_price <= 0 or stop_distance <= 0:
        return {
            "entry_price": float(entry_price),
            "stop_price": 0.0,
            "target_1_price": 0.0,
            "target_2_price": 0.0,
            "risk_per_share": 0.0,
        }
    if normalized_side == "SELL":
        stop_price = entry_price + stop_distance
        target_1 = max(0.0, entry_price - stop_distance)
        target_2 = max(0.0, entry_price - (2.0 * stop_distance))
    else:
        stop_price = max(0.0, entry_price - stop_distance)
        target_1 = entry_price + stop_distance
        target_2 = entry_price + (2.0 * stop_distance)
    return {
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "target_1_price": float(target_1),
        "target_2_price": float(target_2),
        "target_1_r": 1.0,
        "target_2_r": 2.0,
        "risk_per_share": float(stop_distance),
    }


def _normalize_templates(values: list[str] | None) -> list[str]:
    available = {template.key for template in list_templates()}
    selected = [str(value).strip() for value in (values or []) if str(value).strip()]
    if not selected:
        selected = sorted(available)
    deduped: list[str] = []
    for item in selected:
        if item in available and item not in deduped:
            deduped.append(item)
    return deduped


def _merge_params(
    template_key: str,
    params_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    template = get_template(template_key)
    merged: dict[str, Any] = dict(template.default_params)
    if not isinstance(params_overrides, dict):
        return merged

    def _copy_value(value: Any) -> Any:
        if isinstance(value, dict):
            return dict(value)
        return value

    global_params = params_overrides.get("global")
    if isinstance(global_params, dict):
        merged.update(
            {
                str(key): _copy_value(value)
                for key, value in global_params.items()
                if isinstance(value, (int, float, str, dict))
            }
        )
    template_params = params_overrides.get(template_key)
    if isinstance(template_params, dict):
        merged.update(
            {
                str(key): _copy_value(value)
                for key, value in template_params.items()
                if isinstance(value, (int, float, str, dict))
            }
        )

    # Backwards-compatible support for a flat override dictionary.
    for key, value in params_overrides.items():
        if key in {"global", template_key}:
            continue
        if isinstance(value, (int, float, str, dict)):
            merged[str(key)] = _copy_value(value)
    return merged


def _merge_string_override(
    template_key: str,
    params_overrides: dict[str, Any] | None,
    *,
    key: str,
    default: str,
) -> str:
    if not isinstance(params_overrides, dict):
        return default
    global_params = params_overrides.get("global")
    if isinstance(global_params, dict):
        value = global_params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    template_params = params_overrides.get(template_key)
    if isinstance(template_params, dict):
        value = template_params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    value = params_overrides.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip().upper()
    return default


def _corr_map(frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    if len(frames) < 2:
        return {}
    returns: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        if frame.empty:
            continue
        series = frame["close"].pct_change().tail(90)
        if series.notna().sum() >= 10:
            returns[symbol] = series.reset_index(drop=True)
    if len(returns) < 2:
        return {}
    corr = pd.DataFrame(returns).corr(min_periods=10)
    out: dict[str, dict[str, float]] = {}
    for symbol in corr.columns:
        values = {
            other: float(np.nan_to_num(corr.loc[symbol, other], nan=0.0))
            for other in corr.columns
            if other != symbol
        }
        out[str(symbol)] = values
    return out


def _deterministic_tiebreak(
    *,
    symbol: str,
    side: str,
    template: str,
    timeframe: str,
    seed: int,
) -> str:
    return hashlib.sha1(f"{seed}:{symbol}:{side}:{template}:{timeframe}".encode("utf-8")).hexdigest()


def _signal_sides_for_template(
    *,
    template_key: str,
    frame: pd.DataFrame,
    params: dict[str, Any],
) -> dict[str, pd.Series]:
    signals = generate_signal_sides(template_key, frame, params=params)
    return {
        "BUY": signals.get("BUY", pd.Series(False, index=frame.index)).fillna(False).astype(bool),
        "SELL": signals.get("SELL", pd.Series(False, index=frame.index)).fillna(False).astype(bool),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.nan_to_num(float(value), nan=default, posinf=default, neginf=default))
    except (TypeError, ValueError):
        return default


def _status_min(left: str, right: str) -> str:
    rank = {"FAIL": 0, "WARN": 1, "PASS": 2}
    left_norm = str(left or "PASS").upper()
    right_norm = str(right or "PASS").upper()
    return left_norm if rank.get(left_norm, 2) <= rank.get(right_norm, 2) else right_norm


def _market_context_from_frames(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for frame in frames.values():
        if len(frame) < 220:
            continue
        clean = frame.copy().sort_values("datetime").reset_index(drop=True)
        close = pd.to_numeric(clean["close"], errors="coerce")
        sma50 = sma(close, 50)
        sma200 = sma(close, 200)
        if pd.isna(close.iloc[-1]) or pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1]):
            continue
        previous_idx = max(0, len(clean) - 6)
        rows.append(
            {
                "above50": 1.0 if float(close.iloc[-1]) > float(sma50.iloc[-1]) else 0.0,
                "above200": 1.0 if float(close.iloc[-1]) > float(sma200.iloc[-1]) else 0.0,
                "above50_prev": (
                    1.0 if float(close.iloc[previous_idx]) > float(sma50.iloc[previous_idx]) else 0.0
                ),
                "above200_prev": (
                    1.0
                    if float(close.iloc[previous_idx]) > float(sma200.iloc[previous_idx])
                    else 0.0
                ),
                "ret20": (
                    (float(close.iloc[-1]) / float(close.iloc[-21]) - 1.0) * 100.0
                    if len(close) >= 21 and float(close.iloc[-21]) > 0
                    else 0.0
                ),
            }
        )
    if not rows:
        return {
            "status": "PASS",
            "flags": [],
            "symbols": 0,
            "breadth50": 0.0,
            "breadth200": 0.0,
            "breadth50_chg5": 0.0,
            "breadth200_chg5": 0.0,
            "avg_ret20": 0.0,
        }

    context = pd.DataFrame(rows)
    breadth50 = float(context["above50"].mean() * 100.0)
    breadth200 = float(context["above200"].mean() * 100.0)
    breadth50_chg5 = float((context["above50"].mean() - context["above50_prev"].mean()) * 100.0)
    breadth200_chg5 = float((context["above200"].mean() - context["above200_prev"].mean()) * 100.0)
    avg_ret20 = float(context["ret20"].mean())

    flags: list[str] = []
    if breadth50 <= 25.0 or avg_ret20 <= -5.0 or breadth50_chg5 <= -15.0:
        flags.append("market_breadth_breakdown")
    elif breadth50 <= 35.0:
        flags.append("market_breadth_weak")
    if breadth50_chg5 <= -10.0 or breadth200_chg5 <= -8.0:
        flags.append("market_breadth_deteriorating")
    if avg_ret20 <= -3.0:
        flags.append("market_momentum_negative")

    hard_flags = {"market_breadth_breakdown"}
    status = "FAIL" if any(flag in hard_flags for flag in flags) else ("WARN" if flags else "PASS")
    return {
        "status": status,
        "flags": flags,
        "symbols": int(len(rows)),
        "breadth50": breadth50,
        "breadth200": breadth200,
        "breadth50_chg5": breadth50_chg5,
        "breadth200_chg5": breadth200_chg5,
        "avg_ret20": avg_ret20,
    }


def _market_context_quality_for_side(
    market_context: dict[str, Any],
    *,
    side: str,
) -> dict[str, Any]:
    side_norm = str(side or "BUY").upper()
    flags = [str(flag) for flag in market_context.get("flags", [])]
    status = str(market_context.get("status", "PASS")).upper()
    if side_norm == "SELL":
        bearish_flags = {
            "market_breadth_breakdown",
            "market_breadth_weak",
            "market_breadth_deteriorating",
            "market_momentum_negative",
        }
        adverse_flags = [flag for flag in flags if flag not in bearish_flags]
        return {
            "status": status if adverse_flags else "PASS",
            "flags": adverse_flags,
        }
    return {"status": status, "flags": flags}


def _upcoming_action_quality(
    *,
    session: Session,
    symbol: str,
    decision_day: date,
    lookahead_days: int = 7,
) -> dict[str, Any]:
    end_day = decision_day + timedelta(days=max(0, int(lookahead_days)))
    actions = [
        action
        for action in list_symbol_actions(session, symbol=symbol)
        if decision_day <= action.ex_date <= end_day
    ]
    if not actions:
        return {"status": "PASS", "flags": [], "actions": []}
    serialized = [
        {
            "type": str(action.action_type).upper(),
            "ex_date": action.ex_date.isoformat(),
        }
        for action in actions
    ]
    flags = [
        f"upcoming_corporate_action:{item['type'].lower()}:{item['ex_date']}"
        for item in serialized
    ]
    hard = any(item["type"] == "DEMERGER" for item in serialized)
    return {
        "status": "FAIL" if hard else "WARN",
        "flags": flags,
        "actions": serialized,
    }


def _candidate_quality(
    *,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    decision_idx: int,
    side: str,
    template_key: str,
) -> dict[str, Any]:
    """Score signal-bar quality without using bars after the decision close."""

    row = frame.iloc[decision_idx]
    high = _safe_float(row.get("high"))
    low = _safe_float(row.get("low"))
    close = _safe_float(row.get("close"))
    volume = _safe_float(row.get("volume"))
    bar_range = max(0.0, high - low)
    raw_close_location = 0.5 if bar_range <= 0 else (close - low) / bar_range
    close_location = raw_close_location if side == "BUY" else 1.0 - raw_close_location
    close_location = max(0.0, min(1.0, close_location))

    volume_ma = _safe_float(frame["volume"].iloc[: decision_idx + 1].tail(20).mean())
    volume_ratio = volume / volume_ma if volume_ma > 0 else 0.0

    atr_value = 0.0
    if "atr_14" in features.columns and decision_idx < len(features):
        atr_value = _safe_float(features["atr_14"].iloc[decision_idx])
    if atr_value <= 0:
        atr_value = _safe_float(atr(frame, period=14).iloc[decision_idx])
    atr_pct = atr_value / close if close > 0 and atr_value > 0 else 0.0
    range_atr = bar_range / atr_value if atr_value > 0 else 0.0
    close_history = pd.to_numeric(
        frame["close"].iloc[max(0, decision_idx - 260) : decision_idx + 1],
        errors="coerce",
    )
    recent_gap_pct = float(
        np.nan_to_num(close_history.pct_change().abs().max(), nan=0.0, posinf=0.0, neginf=0.0)
    )

    stability_component = 1.0 - min(1.0, max(0.0, atr_pct) / 0.07)
    volume_component = min(1.0, max(0.0, volume_ratio) / 1.5)
    range_component = 1.0 - min(1.0, max(0.0, range_atr - 2.5) / 2.0)
    score = (
        0.45 * close_location
        + 0.20 * volume_component
        + 0.25 * stability_component
        + 0.10 * range_component
    )

    flags: list[str] = []
    if close_location < 0.35:
        flags.append("weak_signal_bar_close")
    if atr_pct > 0.06:
        flags.append("extreme_volatility")
    if range_atr > 2.5 and close_location < 0.60:
        flags.append("blowoff_reversal_bar")
    if template_key in {"trend_breakout", "squeeze_breakout"} and volume_ratio < 0.75:
        flags.append("low_breakout_volume")
    if recent_gap_pct > 0.35:
        flags.append("recent_price_discontinuity")

    hard_flags = {
        "weak_signal_bar_close",
        "extreme_volatility",
        "blowoff_reversal_bar",
        "recent_price_discontinuity",
    }
    status = "PASS"
    if any(flag in hard_flags for flag in flags):
        status = "FAIL"
    elif flags or score < 0.55:
        status = "WARN"

    return {
        "quality_score": float(score),
        "quality_status": status,
        "quality_flags": flags,
        "quality_metrics": {
            "close_location": float(close_location),
            "volume_ratio": float(volume_ratio),
            "atr_pct": float(atr_pct),
            "range_atr": float(range_atr),
            "recent_gap_pct": float(recent_gap_pct),
        },
    }


def _resolve_symbols(
    *,
    session: Session,
    store: DataStore,
    dataset_id: int | None,
    bundle_id: int | None,
    timeframe: str,
    symbol_scope: str,
    max_symbols_scan: int,
    seed: int,
    asof_date: datetime | date | None,
) -> tuple[list[str], int]:
    if bundle_id is not None:
        total_symbols = store.get_bundle_symbols(session, bundle_id, timeframe=timeframe)
        selected = store.sample_bundle_symbols(
            session,
            bundle_id=bundle_id,
            timeframe=timeframe,
            symbol_scope=symbol_scope,
            max_symbols_scan=max_symbols_scan,
            seed=seed,
            asof_date=asof_date,
        )
        return selected, len(total_symbols)
    if dataset_id is not None:
        total_symbols = store.get_dataset_symbols(session, dataset_id, timeframe=timeframe)
        selected = store.sample_dataset_symbols(
            session,
            dataset_id=dataset_id,
            timeframe=timeframe,
            symbol_scope=symbol_scope,
            max_symbols_scan=max_symbols_scan,
            seed=seed,
            asof_date=asof_date,
        )
        return selected, len(total_symbols)
    return [], 0


def generate_signals_for_policy(
    *,
    session: Session,
    store: DataStore,
    dataset_id: int | None = None,
    bundle_id: int | None = None,
    asof: datetime | date | None = None,
    timeframes: list[str] | None = None,
    allowed_templates: list[str] | None = None,
    params_overrides: dict[str, Any] | None = None,
    max_symbols_scan: int = 50,
    seed: int = 7,
    mode: SignalMode = "paper",
    symbol_scope: str = "liquid",
    ranking_weights: dict[str, float] | None = None,
    max_runtime_seconds: int | None = None,
    event_risk_overrides: dict[str, Any] | None = None,
) -> SignalGenerationResult:
    resolved_timeframes = [str(value).strip() for value in (timeframes or []) if str(value).strip()]
    if not resolved_timeframes:
        resolved_timeframes = ["1d"]
    templates = _normalize_templates(allowed_templates)
    if not templates:
        return SignalGenerationResult(
            signals=[],
            scan_truncated=False,
            scanned_symbols=0,
            evaluated_candidates=0,
            total_symbols=0,
        )

    primary_timeframe = resolved_timeframes[0]
    symbols, total_symbols = _resolve_symbols(
        session=session,
        store=store,
        dataset_id=dataset_id,
        bundle_id=bundle_id,
        timeframe=primary_timeframe,
        symbol_scope=symbol_scope,
        max_symbols_scan=max_symbols_scan,
        seed=seed,
        asof_date=asof,
    )
    if not symbols:
        return SignalGenerationResult(
            signals=[],
            scan_truncated=False,
            scanned_symbols=0,
            evaluated_candidates=0,
            total_symbols=total_symbols,
        )

    asof_ts = _asof_timestamp(asof)
    query_start, query_end = _query_window(asof_ts)
    primary_frames: dict[str, pd.DataFrame] = {}
    ranked: list[dict[str, Any]] = []
    scanned_symbols = 0
    evaluated_candidates = 0
    scan_truncated = len(symbols) < total_symbols
    started = time.monotonic()
    runtime_limit = max_runtime_seconds if max_runtime_seconds and max_runtime_seconds > 0 else None

    weights = dict(DEFAULT_RANKING_WEIGHTS)
    if isinstance(ranking_weights, dict):
        for key in ("signal", "liquidity", "stability", "quality"):
            value = ranking_weights.get(key)
            if isinstance(value, (int, float)):
                weights[key] = float(value)

    for symbol in symbols:
        if runtime_limit is not None and (time.monotonic() - started) >= runtime_limit:
            scan_truncated = True
            break
        scanned_symbols += 1
        base = _load_signal_frame(
            store=store,
            session=session,
            symbol=symbol,
            timeframe=primary_timeframe,
            asof=asof_ts,
            start=query_start,
            end=query_end,
        )
        if len(base) < 50:
            continue
        primary_frames[symbol] = base

        for timeframe in resolved_timeframes:
            if runtime_limit is not None and (time.monotonic() - started) >= runtime_limit:
                scan_truncated = True
                break
            frame = (
                base
                if timeframe == primary_timeframe
                else _load_signal_frame(
                    store=store,
                    session=session,
                    symbol=symbol,
                    timeframe=timeframe,
                    asof=asof_ts,
                    start=query_start,
                    end=query_end,
                )
            )
            if len(frame) < 50:
                continue
            features = _minimal_feature_frame(frame)
            if len(features) < 50:
                continue
            min_len = min(len(frame), len(features))
            frame = frame.tail(min_len).reset_index(drop=True)
            features = features.tail(min_len).reset_index(drop=True)
            if len(frame) < 2:
                continue

            for template_key in templates:
                evaluated_candidates += 1
                params = _merge_params(template_key, params_overrides)
                side_series = _signal_sides_for_template(
                    template_key=template_key,
                    frame=frame,
                    params=params,
                )
                instrument = store.find_instrument(session, symbol=symbol)
                default_instrument_kind = instrument.kind if instrument is not None else "EQUITY_CASH"
                instrument_kind = _merge_string_override(
                    template_key,
                    params_overrides,
                    key="instrument_kind",
                    default=default_instrument_kind,
                )
                side_override = _merge_string_override(
                    template_key,
                    params_overrides,
                    key="side",
                    default="AUTO",
                )
                decision_idx = len(frame) - 2
                fill_idx = len(frame) - 1
                candidate_sides = ["BUY", "SELL"]
                if side_override in {"BUY", "SELL"}:
                    candidate_sides = [side_override]

                for side in candidate_sides:
                    signal_on_decision_bar = bool(
                        side_series.get(side, pd.Series(False, index=frame.index)).iloc[decision_idx]
                    )
                    if not signal_on_decision_bar:
                        continue

                    atr_period = int(params.get("atr_period", 14))
                    if atr_period == 14 and "atr_14" in features.columns:
                        atr_value = float(np.nan_to_num(features["atr_14"].iloc[decision_idx], nan=0.0))
                    else:
                        # Fallback for non-cached ATR period.
                        atr_value = float(
                            np.nan_to_num(atr(frame, period=atr_period).iloc[decision_idx], nan=0.0)
                        )
                    execution_symbol = symbol.upper()
                    underlying_symbol = symbol.upper()
                    instrument_choice_reason = "provided"
                    chosen_instrument_kind = instrument_kind
                    chosen_lot_size = max(
                        1,
                        int(
                            params.get(
                                "lot_size",
                                store.get_lot_size(
                                    session,
                                    symbol=symbol,
                                    instrument_kind=instrument_kind,
                                ),
                            )
                        ),
                    )

                    chosen_frame = frame
                    chosen_features = features
                    chosen_decision_idx = decision_idx
                    chosen_fill_idx = fill_idx

                    if side == "SELL" and instrument_kind not in {"STOCK_FUT", "INDEX_FUT"}:
                        futures_instrument = store.find_futures_instrument_for_underlying(
                            session,
                            underlying=underlying_symbol,
                            bundle_id=bundle_id,
                            timeframe=timeframe,
                        )
                        if futures_instrument is not None:
                            fut_frame = _load_signal_frame(
                                store=store,
                                session=session,
                                symbol=futures_instrument.symbol,
                                timeframe=timeframe,
                                asof=asof_ts,
                                start=query_start,
                                end=query_end,
                            )
                            if len(fut_frame) >= 2:
                                fut_features = _minimal_feature_frame(fut_frame)
                                if len(fut_features) >= 2:
                                    fut_min_len = min(len(fut_frame), len(fut_features))
                                    fut_frame = fut_frame.tail(fut_min_len).reset_index(drop=True)
                                    fut_features = fut_features.tail(fut_min_len).reset_index(drop=True)
                                    chosen_features = fut_features
                                chosen_frame = fut_frame.reset_index(drop=True)
                                chosen_decision_idx = len(chosen_frame) - 2
                                chosen_fill_idx = len(chosen_frame) - 1
                                execution_symbol = futures_instrument.symbol.upper()
                                chosen_instrument_kind = futures_instrument.kind
                                chosen_lot_size = max(1, int(futures_instrument.lot_size))
                                instrument_choice_reason = "swing_short_requires_futures"

                    price = float(chosen_frame.iloc[chosen_fill_idx]["open"])
                    if price <= 0:
                        price = float(chosen_frame.iloc[chosen_decision_idx]["close"])
                    if price <= 0 or atr_value <= 0:
                        continue

                    atr_mult = float(params.get("atr_stop_mult", params.get("atr_stop", 2.0)))
                    stop_distance = atr_value * atr_mult
                    if stop_distance <= 0:
                        continue

                    take_profit_r = params.get("take_profit_r")
                    if isinstance(take_profit_r, (int, float)):
                        target_price = (
                            price + float(take_profit_r) * stop_distance
                            if side == "BUY"
                            else max(0.0, price - float(take_profit_r) * stop_distance)
                        )
                    else:
                        target_price = None

                    raw_strength = float(signal_strength(frame, decision_idx))
                    adv = float(
                        np.nan_to_num(
                            (
                                chosen_frame["close"] * chosen_frame["volume"]
                            ).iloc[: chosen_decision_idx + 1].tail(20).mean(),
                            nan=0.0,
                        )
                    )
                    atr_pct = float(
                        np.nan_to_num(
                            chosen_features.get("atr_pct", pd.Series([0.0])).iloc[
                                chosen_decision_idx
                            ],
                            nan=0.0,
                        )
                    )
                    liquidity_component = float(np.tanh(np.log1p(max(0.0, adv)) / 20.0))
                    stability_component = 1.0 - min(1.0, max(0.0, atr_pct) * 15.0)
                    quality = _candidate_quality(
                        frame=frame,
                        features=features,
                        decision_idx=decision_idx,
                        side=side,
                        template_key=template_key,
                    )
                    decision_day = pd.Timestamp(frame.iloc[decision_idx]["datetime"]).date()
                    action_quality = _upcoming_action_quality(
                        session=session,
                        symbol=underlying_symbol,
                        decision_day=decision_day,
                    )
                    event_quality = evaluate_event_risk(
                        asof_date=decision_day,
                        symbol=underlying_symbol,
                        overrides=event_risk_overrides,
                    )
                    quality_flags = (
                        list(quality["quality_flags"])
                        + list(action_quality["flags"])
                        + list(event_quality["flags"])
                    )
                    quality_status = _status_min(
                        _status_min(str(quality["quality_status"]), str(action_quality["status"])),
                        str(event_quality["status"]),
                    )
                    quality_metrics = dict(quality["quality_metrics"])
                    if action_quality["actions"]:
                        quality_metrics["upcoming_corporate_actions"] = action_quality["actions"]
                    if event_quality["events"]:
                        quality_metrics["event_risk_events"] = event_quality["events"]
                    quality_score = float(quality["quality_score"])
                    ranking_score = (
                        weights["signal"] * raw_strength
                        + weights["liquidity"] * liquidity_component
                        + weights["stability"] * stability_component
                        + weights["quality"] * quality_score
                    )
                    trade_plan_prices = _trade_plan_prices(side, price, stop_distance)

                    ranked.append(
                        {
                            "symbol": execution_symbol,
                            "underlying_symbol": underlying_symbol,
                            "side": side,
                            "template": template_key,
                            "timeframe": timeframe,
                            "price": price,
                            **trade_plan_prices,
                            "stop_distance": stop_distance,
                            "target_price": target_price,
                            "signal_strength": float(ranking_score),
                            "raw_signal_strength": raw_strength,
                            "adv": adv,
                            "vol_scale": max(0.0, atr_pct),
                            "quality_score": quality_score,
                            "quality_status": quality_status,
                            "quality_flags": quality_flags,
                            "quality_metrics": quality_metrics,
                            "signal_at": str(frame.iloc[decision_idx]["datetime"]),
                            "fill_at": str(chosen_frame.iloc[chosen_fill_idx]["datetime"]),
                            "source_mode": mode,
                            "instrument_kind": chosen_instrument_kind,
                            "lot_size": chosen_lot_size,
                            "instrument_choice_reason": instrument_choice_reason,
                            "ranking_weights": weights,
                            "explanation": (
                                f"{template_key} {side} signal at close on {timeframe} "
                                f"(rank={ranking_score:.3f}, raw={raw_strength:.3f}, adv={adv:.0f})."
                            ),
                        }
                    )
        else:
            continue
        break

    if not ranked:
        return SignalGenerationResult(
            signals=[],
            scan_truncated=scan_truncated,
            scanned_symbols=scanned_symbols,
            evaluated_candidates=evaluated_candidates,
            total_symbols=total_symbols,
        )

    corr_map = _corr_map(primary_frames)
    market_context = _market_context_from_frames(primary_frames)
    for row in ranked:
        symbol = str(row["symbol"])
        if symbol in corr_map and corr_map[symbol]:
            row["correlations"] = corr_map[symbol]
        row["market_context"] = market_context
        market_quality = _market_context_quality_for_side(
            market_context,
            side=str(row.get("side", "BUY")),
        )
        if market_quality["status"] != "PASS":
            existing_flags = [str(flag) for flag in row.get("quality_flags", [])]
            for flag in market_quality["flags"]:
                if flag not in existing_flags:
                    existing_flags.append(str(flag))
            row["quality_flags"] = existing_flags
            row["quality_status"] = _status_min(
                str(row.get("quality_status", "PASS")),
                str(market_quality["status"]),
            )

    ranked.sort(
        key=lambda row: (
            -float(row.get("signal_strength", 0.0)),
            -float(row.get("adv", 0.0)),
            str(row["symbol"]),
            str(row.get("side", "BUY")),
            str(row.get("template", "")),
            str(row.get("timeframe", "")),
            _deterministic_tiebreak(
                symbol=str(row["symbol"]),
                side=str(row.get("side", "BUY")),
                template=str(row.get("template", "")),
                timeframe=str(row.get("timeframe", "")),
                seed=seed,
            ),
        )
    )
    return SignalGenerationResult(
        signals=ranked,
        scan_truncated=scan_truncated,
        scanned_symbols=scanned_symbols,
        evaluated_candidates=evaluated_candidates,
        total_symbols=total_symbols,
    )
