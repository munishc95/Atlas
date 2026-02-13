from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.engine.indicators import atr
from app.services.data_store import DataStore
from app.strategies.templates import (
    generate_signal_sides,
    get_template,
    list_templates,
    signal_strength,
)


SignalMode = Literal["paper", "preview"]

DEFAULT_RANKING_WEIGHTS: dict[str, float] = {
    "signal": 0.65,
    "liquidity": 0.25,
    "stability": 0.10,
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
    primary_frames: dict[str, pd.DataFrame] = {}
    ranked: list[dict[str, Any]] = []
    evaluated_candidates = 0
    scan_truncated = len(symbols) < total_symbols
    started = time.monotonic()
    runtime_limit = max_runtime_seconds if max_runtime_seconds and max_runtime_seconds > 0 else None

    weights = dict(DEFAULT_RANKING_WEIGHTS)
    if isinstance(ranking_weights, dict):
        for key in ("signal", "liquidity", "stability"):
            value = ranking_weights.get(key)
            if isinstance(value, (int, float)):
                weights[key] = float(value)

    for symbol in symbols:
        if runtime_limit is not None and (time.monotonic() - started) >= runtime_limit:
            scan_truncated = True
            break
        base = _filtered_frame(
            store.load_ohlcv(symbol=symbol, timeframe=primary_timeframe), asof_ts
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
                else _filtered_frame(store.load_ohlcv(symbol=symbol, timeframe=timeframe), asof_ts)
            )
            if len(frame) < 50:
                continue
            features = _filtered_frame(
                store.load_features(symbol=symbol, timeframe=timeframe),
                asof_ts,
            )
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
                            fut_frame = _filtered_frame(
                                store.load_ohlcv(symbol=futures_instrument.symbol, timeframe=timeframe),
                                asof_ts,
                            )
                            if len(fut_frame) >= 2:
                                fut_features = _filtered_frame(
                                    store.load_features(symbol=futures_instrument.symbol, timeframe=timeframe),
                                    asof_ts,
                                )
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
                    ranking_score = (
                        weights["signal"] * raw_strength
                        + weights["liquidity"] * liquidity_component
                        + weights["stability"] * stability_component
                    )

                    ranked.append(
                        {
                            "symbol": execution_symbol,
                            "underlying_symbol": underlying_symbol,
                            "side": side,
                            "template": template_key,
                            "timeframe": timeframe,
                            "price": price,
                            "stop_distance": stop_distance,
                            "target_price": target_price,
                            "signal_strength": float(ranking_score),
                            "raw_signal_strength": raw_strength,
                            "adv": adv,
                            "vol_scale": max(0.0, atr_pct),
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
            scanned_symbols=len(symbols),
            evaluated_candidates=evaluated_candidates,
            total_symbols=total_symbols,
        )

    corr_map = _corr_map(primary_frames)
    for row in ranked:
        symbol = str(row["symbol"])
        if symbol in corr_map and corr_map[symbol]:
            row["correlations"] = corr_map[symbol]

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
        scanned_symbols=len(symbols),
        evaluated_candidates=evaluated_candidates,
        total_symbols=total_symbols,
    )
