from __future__ import annotations

from datetime import date, datetime
import hashlib
from typing import Any, Literal

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.engine.indicators import atr
from app.services.data_store import DataStore
from app.strategies.templates import generate_signals, get_template, list_templates, signal_strength


SignalMode = Literal["paper", "preview"]


def _asof_timestamp(value: datetime | date | None) -> pd.Timestamp | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return pd.Timestamp(value).tz_localize("UTC")
    return pd.Timestamp(value, tz="UTC")


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


def generate_signals_for_policy(
    *,
    session: Session,
    store: DataStore,
    dataset_id: int,
    asof: datetime | date | None = None,
    timeframes: list[str] | None = None,
    allowed_templates: list[str] | None = None,
    params_overrides: dict[str, Any] | None = None,
    max_symbols_scan: int = 50,
    seed: int = 7,
    mode: SignalMode = "paper",
    symbol_scope: str = "liquid",
) -> list[dict[str, Any]]:
    dataset = store.get_dataset(session, dataset_id)
    if dataset is None:
        return []

    resolved_timeframes = [str(value).strip() for value in (timeframes or []) if str(value).strip()]
    if not resolved_timeframes:
        resolved_timeframes = [dataset.timeframe or "1d"]
    templates = _normalize_templates(allowed_templates)
    if not templates:
        return []

    symbols = store.sample_dataset_symbols(
        session,
        dataset_id=dataset_id,
        timeframe=resolved_timeframes[0],
        symbol_scope=symbol_scope,
        max_symbols_scan=max_symbols_scan,
        seed=seed,
    )
    if not symbols:
        return []

    asof_ts = _asof_timestamp(asof)
    primary_frames: dict[str, pd.DataFrame] = {}
    ranked: list[dict[str, Any]] = []

    for symbol in symbols:
        base = store.load_ohlcv(symbol=symbol, timeframe=resolved_timeframes[0])
        base = _filtered_frame(base, asof_ts)
        if len(base) < 50:
            continue
        primary_frames[symbol] = base

        for timeframe in resolved_timeframes:
            frame = (
                base
                if timeframe == resolved_timeframes[0]
                else _filtered_frame(
                    store.load_ohlcv(symbol=symbol, timeframe=timeframe),
                    asof_ts,
                )
            )
            if len(frame) < 50:
                continue

            for template_key in templates:
                params = _merge_params(template_key, params_overrides)
                signal_series = generate_signals(template_key, frame, params=params)
                if signal_series.empty or not bool(signal_series.iloc[-1]):
                    continue

                atr_period = int(params.get("atr_period", 14))
                atr_mult = float(params.get("atr_stop_mult", params.get("atr_stop", 2.0)))
                atr_series = atr(frame, period=atr_period)
                atr_value = float(np.nan_to_num(atr_series.iloc[-1], nan=0.0))
                if atr_value <= 0:
                    continue

                price = float(frame.iloc[-1]["close"])
                if price <= 0:
                    continue
                stop_distance = atr_value * atr_mult
                if stop_distance <= 0:
                    continue

                take_profit_r = params.get("take_profit_r")
                target_price = (
                    price + float(take_profit_r) * stop_distance
                    if isinstance(take_profit_r, (int, float))
                    else None
                )
                signal_idx = len(frame) - 1
                strength = float(signal_strength(frame, signal_idx))
                adv = float(
                    np.nan_to_num((frame["close"] * frame["volume"]).tail(20).mean(), nan=0.0)
                )
                vol_scale = float(np.nan_to_num(atr_value / max(price, 1e-9), nan=0.0))

                ranked.append(
                    {
                        "symbol": symbol.upper(),
                        "side": "BUY",
                        "template": template_key,
                        "timeframe": timeframe,
                        "price": price,
                        "stop_distance": stop_distance,
                        "target_price": target_price,
                        "signal_strength": strength,
                        "adv": adv,
                        "vol_scale": vol_scale,
                        "signal_at": str(frame.iloc[-1]["datetime"]),
                        "source_mode": mode,
                        "explanation": (
                            f"{template_key} triggered on latest {timeframe} bar "
                            f"(strength={strength:.3f}, adv={adv:.0f})."
                        ),
                    }
                )

    if not ranked:
        return []

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
    return ranked
