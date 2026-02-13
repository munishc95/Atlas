from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import time
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.engine.indicators import atr, ema, rsi
from app.services.data_store import DataStore
from app.strategies.templates import get_template, list_templates, signal_strength


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
) -> dict[str, float | int]:
    template = get_template(template_key)
    merged: dict[str, float | int] = dict(template.default_params)
    if not isinstance(params_overrides, dict):
        return merged

    global_params = params_overrides.get("global")
    if isinstance(global_params, dict):
        merged.update(
            {
                str(key): value
                for key, value in global_params.items()
                if isinstance(value, (int, float))
            }
        )
    template_params = params_overrides.get(template_key)
    if isinstance(template_params, dict):
        merged.update(
            {
                str(key): value
                for key, value in template_params.items()
                if isinstance(value, (int, float))
            }
        )

    # Backwards-compatible support for a flat override dictionary.
    for key, value in params_overrides.items():
        if key in {"global", template_key}:
            continue
        if isinstance(value, (int, float)):
            merged[str(key)] = value
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
    template: str,
    timeframe: str,
    seed: int,
) -> str:
    return hashlib.sha1(f"{seed}:{symbol}:{template}:{timeframe}".encode("utf-8")).hexdigest()


def _resolve_ema(features: pd.DataFrame, frame: pd.DataFrame, period: int) -> pd.Series:
    key = f"ema_{period}"
    if key in features.columns:
        return pd.Series(features[key].to_numpy(), index=frame.index, dtype="float64")
    return ema(frame["close"], period)


def _resolve_rsi(features: pd.DataFrame, frame: pd.DataFrame, period: int) -> pd.Series:
    key = f"rsi_{period}"
    if key in features.columns:
        return pd.Series(features[key].to_numpy(), index=frame.index, dtype="float64")
    return rsi(frame["close"], period)


def _trigger_template(
    *,
    template_key: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    params: dict[str, float | int],
) -> bool:
    if len(frame) < 50 or len(features) < 50:
        return False
    frame = frame.reset_index(drop=True)
    features = features.reset_index(drop=True)

    if template_key == "trend_breakout":
        trend_period = int(params.get("trend_period", 200))
        breakout_lookback = int(params.get("breakout_lookback", 20))
        trend = frame["close"] > _resolve_ema(features, frame, trend_period)
        breakout_level = (
            frame["high"].rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
        )
        if breakout_level.empty:
            return False
        return bool(trend.iloc[-1] and frame["close"].iloc[-1] > breakout_level.iloc[-1])

    if template_key == "pullback_trend":
        ema_fast = _resolve_ema(features, frame, int(params.get("ema_fast", 20)))
        ema_mid = _resolve_ema(features, frame, int(params.get("ema_mid", 50)))
        ema_slow = _resolve_ema(features, frame, int(params.get("ema_slow", 100)))
        rsi_period = int(params.get("rsi_period", 4))
        rsi_oversold = float(params.get("rsi_oversold", 25))
        pullback_band = float(params.get("pullback_band", 0.99))
        trend = (frame["close"] > ema_slow) & (ema_fast > ema_mid)
        pullback = _resolve_rsi(features, frame, rsi_period) < rsi_oversold
        near_band = frame["close"] <= (ema_fast * pullback_band)
        return bool((trend & pullback & near_band).fillna(False).iloc[-1])

    if template_key == "squeeze_breakout":
        breakout_lookback = int(params.get("breakout_lookback", 20))
        squeeze_prev = (features["bb_upper"] < features["kc_upper"]) & (
            features["bb_lower"] > features["kc_lower"]
        )
        breakout_level = (
            frame["high"].rolling(breakout_lookback, min_periods=breakout_lookback).max().shift(1)
        )
        vol_confirm = pd.Series(True, index=frame.index)
        if "volume" in frame.columns:
            vol_ma = frame["volume"].rolling(20, min_periods=20).mean()
            vol_confirm = frame["volume"] >= vol_ma
        shifted = squeeze_prev.shift(1).astype("boolean").fillna(False).astype(bool)
        trigger = shifted & (frame["close"] > breakout_level) & vol_confirm.fillna(False)
        return bool(trigger.fillna(False).iloc[-1])

    return False


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

            for template_key in templates:
                evaluated_candidates += 1
                params = _merge_params(template_key, params_overrides)
                if not _trigger_template(
                    template_key=template_key,
                    frame=frame,
                    features=features,
                    params=params,
                ):
                    continue

                atr_period = int(params.get("atr_period", 14))
                if atr_period == 14 and "atr_14" in features.columns:
                    atr_value = float(np.nan_to_num(features["atr_14"].iloc[-1], nan=0.0))
                else:
                    # Fallback for non-cached ATR period.
                    atr_value = float(
                        np.nan_to_num(atr(frame, period=atr_period).iloc[-1], nan=0.0)
                    )
                price = float(frame.iloc[-1]["close"])
                if price <= 0 or atr_value <= 0:
                    continue

                atr_mult = float(params.get("atr_stop_mult", params.get("atr_stop", 2.0)))
                stop_distance = atr_value * atr_mult
                if stop_distance <= 0:
                    continue

                side = _merge_string_override(
                    template_key,
                    params_overrides,
                    key="side",
                    default="BUY",
                )
                instrument_kind = _merge_string_override(
                    template_key,
                    params_overrides,
                    key="instrument_kind",
                    default="EQUITY_CASH",
                )
                lot_size = max(
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

                take_profit_r = params.get("take_profit_r")
                if isinstance(take_profit_r, (int, float)):
                    target_price = (
                        price + float(take_profit_r) * stop_distance
                        if side == "BUY"
                        else max(0.0, price - float(take_profit_r) * stop_distance)
                    )
                else:
                    target_price = None

                raw_strength = float(signal_strength(frame, len(frame) - 1))
                adv = float(
                    np.nan_to_num((frame["close"] * frame["volume"]).tail(20).mean(), nan=0.0)
                )
                atr_pct = float(
                    np.nan_to_num(features.get("atr_pct", pd.Series([0.0])).iloc[-1], nan=0.0)
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
                        "symbol": symbol.upper(),
                        "side": "SELL" if side == "SELL" else "BUY",
                        "template": template_key,
                        "timeframe": timeframe,
                        "price": price,
                        "stop_distance": stop_distance,
                        "target_price": target_price,
                        "signal_strength": float(ranking_score),
                        "raw_signal_strength": raw_strength,
                        "adv": adv,
                        "vol_scale": max(0.0, atr_pct),
                        "signal_at": str(frame.iloc[-1]["datetime"]),
                        "source_mode": mode,
                        "instrument_kind": instrument_kind,
                        "lot_size": lot_size,
                        "ranking_weights": weights,
                        "explanation": (
                            f"{template_key} triggered on latest {timeframe} bar "
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
            str(row.get("template", "")),
            str(row.get("timeframe", "")),
            _deterministic_tiebreak(
                symbol=str(row["symbol"]),
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
