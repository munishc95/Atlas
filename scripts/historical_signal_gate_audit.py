from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np
import pandas as pd
from sqlmodel import Session

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import engine, init_db  # noqa: E402
from app.engine.signal_engine import (  # noqa: E402
    DEFAULT_RANKING_WEIGHTS,
    _candidate_quality,
    _market_context_quality_for_side,
    _merge_params,
    _minimal_feature_frame,
    _signal_sides_for_template,
    _status_min,
    _trade_plan_prices,
)
from app.engine.indicators import sma  # noqa: E402
from app.services.data_store import DataStore  # noqa: E402
from app.services.event_risk import evaluate_event_risk  # noqa: E402
from app.services.paper import _preview_trade_plan, entry_quality_block_reason  # noqa: E402
from app.strategies.templates import list_templates, signal_strength  # noqa: E402


TERMINAL_STATUSES = {"STOP_HIT", "T1_HIT", "T2_HIT", "EXPIRED"}


@dataclass
class PreparedSymbol:
    frame: pd.DataFrame
    features: pd.DataFrame
    date_to_index: dict[date, int]
    signals_by_template: dict[str, dict[str, pd.Series]]


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _settings_store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _market_context_from_prepared(
    prepared: dict[str, PreparedSymbol],
    *,
    fill_date: date,
) -> dict[str, Any]:
    rows: list[dict[str, float]] = []
    for item in prepared.values():
        fill_idx = item.date_to_index.get(fill_date)
        if fill_idx is None or fill_idx < 220:
            continue
        decision_idx = fill_idx - 1
        clean = item.frame.iloc[: decision_idx + 1].reset_index(drop=True)
        close = pd.to_numeric(clean["close"], errors="coerce")
        sma50 = sma(close, 50)
        sma200 = sma(close, 200)
        if pd.isna(close.iloc[-1]) or pd.isna(sma50.iloc[-1]) or pd.isna(sma200.iloc[-1]):
            continue
        prev5 = max(0, len(clean) - 6)
        prev1 = max(0, len(clean) - 2)
        prev2 = max(0, len(clean) - 3)
        rows.append(
            {
                "above50": 1.0 if float(close.iloc[-1]) > float(sma50.iloc[-1]) else 0.0,
                "above200": 1.0 if float(close.iloc[-1]) > float(sma200.iloc[-1]) else 0.0,
                "above50_prev1": (
                    1.0 if float(close.iloc[prev1]) > float(sma50.iloc[prev1]) else 0.0
                ),
                "above50_prev2": (
                    1.0 if float(close.iloc[prev2]) > float(sma50.iloc[prev2]) else 0.0
                ),
                "above200_prev1": (
                    1.0 if float(close.iloc[prev1]) > float(sma200.iloc[prev1]) else 0.0
                ),
                "above50_prev": (
                    1.0 if float(close.iloc[prev5]) > float(sma50.iloc[prev5]) else 0.0
                ),
                "above200_prev": (
                    1.0 if float(close.iloc[prev5]) > float(sma200.iloc[prev5]) else 0.0
                ),
                "ret5": (
                    (float(close.iloc[-1]) / float(close.iloc[-6]) - 1.0) * 100.0
                    if len(close) >= 6 and float(close.iloc[-6]) > 0
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
            "breadth50_chg1": 0.0,
            "breadth50_chg2": 0.0,
            "breadth200_chg1": 0.0,
            "breadth50_chg5": 0.0,
            "breadth200_chg5": 0.0,
            "avg_ret5": 0.0,
            "avg_ret20": 0.0,
        }

    context = pd.DataFrame(rows)
    breadth50 = float(context["above50"].mean() * 100.0)
    breadth200 = float(context["above200"].mean() * 100.0)
    breadth50_chg1 = float(
        (context["above50"].mean() - context["above50_prev1"].mean()) * 100.0
    )
    breadth50_chg2 = float(
        (context["above50"].mean() - context["above50_prev2"].mean()) * 100.0
    )
    breadth200_chg1 = float(
        (context["above200"].mean() - context["above200_prev1"].mean()) * 100.0
    )
    breadth50_chg5 = float(
        (context["above50"].mean() - context["above50_prev"].mean()) * 100.0
    )
    breadth200_chg5 = float(
        (context["above200"].mean() - context["above200_prev"].mean()) * 100.0
    )
    avg_ret5 = float(context["ret5"].mean())
    avg_ret20 = float(context["ret20"].mean())
    flags: list[str] = []
    if breadth50 >= 80.0 and avg_ret20 >= 9.0 and avg_ret5 >= 1.5:
        flags.append("market_breadth_overextended")
    if breadth50 >= 70.0 and (
        breadth50_chg1 <= -5.0 or breadth50_chg2 <= -7.0 or breadth200_chg1 <= -5.0
    ):
        flags.append("market_breadth_short_term_rollover")
    if breadth50 <= 25.0 or avg_ret20 <= -5.0 or breadth50_chg5 <= -15.0:
        flags.append("market_breadth_breakdown")
    elif breadth50 <= 35.0:
        flags.append("market_breadth_weak")
    if breadth50_chg5 <= -10.0 or breadth200_chg5 <= -8.0:
        flags.append("market_breadth_deteriorating")
    if avg_ret20 <= -3.0:
        flags.append("market_momentum_negative")
    status = "FAIL" if "market_breadth_breakdown" in flags else ("WARN" if flags else "PASS")
    return {
        "status": status,
        "flags": flags,
        "symbols": int(len(rows)),
        "breadth50": breadth50,
        "breadth200": breadth200,
        "breadth50_chg1": breadth50_chg1,
        "breadth50_chg2": breadth50_chg2,
        "breadth200_chg1": breadth200_chg1,
        "breadth50_chg5": breadth50_chg5,
        "breadth200_chg5": breadth200_chg5,
        "avg_ret5": avg_ret5,
        "avg_ret20": avg_ret20,
    }


def _prepare_symbols(
    *,
    session: Session,
    store: DataStore,
    symbols: list[str],
    templates: list[str],
    params_overrides: dict[str, Any] | None,
) -> dict[str, PreparedSymbol]:
    prepared: dict[str, PreparedSymbol] = {}
    for offset, symbol in enumerate(symbols, start=1):
        frame = store.load_ohlcv(symbol=symbol, timeframe="1d", session=session)
        if len(frame) < 260:
            continue
        features = _minimal_feature_frame(frame)
        min_len = min(len(frame), len(features))
        frame = frame.tail(min_len).reset_index(drop=True)
        features = features.tail(min_len).reset_index(drop=True)
        if len(frame) < 260:
            continue
        dates = pd.to_datetime(frame["datetime"], utc=True).dt.date
        date_to_index = {day: int(index) for index, day in enumerate(dates)}
        signals_by_template = {
            template: _signal_sides_for_template(
                template_key=template,
                frame=frame,
                params=_merge_params(template, params_overrides),
            )
            for template in templates
        }
        prepared[symbol] = PreparedSymbol(
            frame=frame,
            features=features,
            date_to_index=date_to_index,
            signals_by_template=signals_by_template,
        )
        if offset % 100 == 0:
            print(f"prepared_symbols={offset}/{len(symbols)} usable={len(prepared)}", flush=True)
    return prepared


def _evaluate_outcome(
    signal: dict[str, Any],
    frame: pd.DataFrame,
    *,
    fill_idx: int,
    horizon_bars: int,
) -> dict[str, Any] | None:
    observed = frame.iloc[fill_idx : fill_idx + max(1, horizon_bars)].reset_index(drop=True)
    if observed.empty:
        return None
    side = str(signal.get("side", "BUY")).upper()
    entry = _safe_float(signal.get("entry_price", signal.get("price")))
    stop = _safe_float(signal.get("stop_price"))
    target_1 = _safe_float(signal.get("target_1_price"))
    target_2 = _safe_float(signal.get("target_2_price"))
    if entry <= 0 or stop <= 0 or target_1 <= 0 or target_2 <= 0:
        return None
    status = "OPEN"
    t1_hit = False
    max_favorable = 0.0
    max_adverse = 0.0
    for _, bar in observed.iterrows():
        high = _safe_float(bar.get("high"))
        low = _safe_float(bar.get("low"))
        if side == "BUY":
            max_favorable = max(max_favorable, ((high - entry) / entry) * 100.0)
            max_adverse = min(max_adverse, ((low - entry) / entry) * 100.0)
            hit_stop = low <= stop
            hit_t1 = high >= target_1
            hit_t2 = high >= target_2
        else:
            max_favorable = max(max_favorable, ((entry - low) / entry) * 100.0)
            max_adverse = min(max_adverse, ((entry - high) / entry) * 100.0)
            hit_stop = high >= stop
            hit_t1 = low <= target_1
            hit_t2 = low <= target_2
        if hit_stop and not t1_hit:
            status = "STOP_HIT"
            break
        if hit_t1:
            t1_hit = True
        if hit_t2:
            status = "T2_HIT"
            break
        if hit_stop and t1_hit:
            status = "T1_THEN_STOP"
            break
    else:
        if t1_hit:
            status = "T1_HIT"
        elif len(observed) >= max(1, horizon_bars):
            status = "EXPIRED"
    latest_close = _safe_float(observed.iloc[-1].get("close"))
    close_return = (
        ((latest_close - entry) / entry) * 100.0
        if side == "BUY"
        else ((entry - latest_close) / entry) * 100.0
    )
    return {
        "status": status,
        "bars_observed": int(len(observed)),
        "latest_price": latest_close,
        "close_return_pct": close_return,
        "max_favorable_pct": max_favorable,
        "max_adverse_pct": max_adverse,
    }


def _candidate_for_fill(
    *,
    session: Session,
    item: PreparedSymbol,
    symbol: str,
    template: str,
    fill_date: date,
    market_context: dict[str, Any],
    state_settings: dict[str, Any],
    equity: float,
    risk_per_trade: float,
    ranking_weights: dict[str, float],
    params_overrides: dict[str, Any] | None,
    horizon_bars: int,
) -> dict[str, Any] | None:
    fill_idx = item.date_to_index.get(fill_date)
    if fill_idx is None or fill_idx <= 0:
        return None
    decision_idx = fill_idx - 1
    buy = item.signals_by_template[template].get("BUY", pd.Series(False, index=item.frame.index))
    if not bool(buy.iloc[decision_idx]):
        return None
    params = _merge_params(template, params_overrides)
    atr_period = int(params.get("atr_period", 14))
    if atr_period == 14 and "atr_14" in item.features.columns:
        atr_value = _safe_float(item.features["atr_14"].iloc[decision_idx])
    else:
        return None
    price = _safe_float(item.frame.iloc[fill_idx].get("open"))
    if price <= 0:
        price = _safe_float(item.frame.iloc[decision_idx].get("close"))
    if price <= 0 or atr_value <= 0:
        return None
    stop_distance = atr_value * _safe_float(params.get("atr_stop_mult", params.get("atr_stop", 2.0)))
    if stop_distance <= 0:
        return None

    side = "BUY"
    quality = _candidate_quality(
        frame=item.frame,
        features=item.features,
        decision_idx=decision_idx,
        side=side,
        template_key=template,
    )
    event_quality = evaluate_event_risk(
        asof_date=pd.Timestamp(item.frame.iloc[decision_idx]["datetime"]).date(),
        symbol=symbol,
        overrides=state_settings,
    )
    quality_flags = list(quality["quality_flags"]) + list(event_quality["flags"])
    quality_status = _status_min(str(quality["quality_status"]), str(event_quality["status"]))
    market_quality = _market_context_quality_for_side(market_context, side=side)
    if market_quality["status"] != "PASS":
        for flag in market_quality["flags"]:
            if flag not in quality_flags:
                quality_flags.append(str(flag))
        quality_status = _status_min(quality_status, str(market_quality["status"]))

    metrics = dict(quality["quality_metrics"])
    if event_quality["events"]:
        metrics["event_risk_events"] = event_quality["events"]
    raw_strength = _safe_float(signal_strength(item.frame, decision_idx))
    adv = _safe_float((item.frame["close"] * item.frame["volume"]).iloc[: decision_idx + 1].tail(20).mean())
    atr_pct = _safe_float(item.features.get("atr_pct", pd.Series([0.0])).iloc[decision_idx])
    liquidity_component = float(np.tanh(np.log1p(max(0.0, adv)) / 20.0))
    stability_component = 1.0 - min(1.0, max(0.0, atr_pct) * 15.0)
    quality_score = _safe_float(quality["quality_score"])
    ranking_score = (
        ranking_weights["signal"] * raw_strength
        + ranking_weights["liquidity"] * liquidity_component
        + ranking_weights["stability"] * stability_component
        + ranking_weights["quality"] * quality_score
    )
    row = {
        "symbol": symbol,
        "underlying_symbol": symbol,
        "side": side,
        "template": template,
        "timeframe": "1d",
        "price": price,
        **_trade_plan_prices(side, price, stop_distance),
        "stop_distance": stop_distance,
        "signal_strength": ranking_score,
        "raw_signal_strength": raw_strength,
        "adv": adv,
        "vol_scale": atr_pct,
        "quality_score": quality_score,
        "quality_status": quality_status,
        "quality_flags": quality_flags,
        "quality_metrics": metrics,
        "signal_at": str(item.frame.iloc[decision_idx]["datetime"]),
        "fill_at": str(item.frame.iloc[fill_idx]["datetime"]),
        "instrument_kind": "EQUITY_CASH",
        "lot_size": 1,
        "market_context": market_context,
    }
    row = _preview_trade_plan(row, equity=equity, risk_per_trade=risk_per_trade)
    outcome = _evaluate_outcome(row, item.frame, fill_idx=fill_idx, horizon_bars=horizon_bars)
    if outcome is None:
        return None
    row.update(outcome)
    row["fill_date"] = fill_date.isoformat()
    row["signal_date"] = pd.Timestamp(item.frame.iloc[decision_idx]["datetime"]).date().isoformat()
    row["entry_block_reason"] = entry_quality_block_reason(row)
    return row


def _ranked(
    rows: list[dict[str, Any]],
    top_n: int,
    *,
    mode: str = "production",
) -> list[dict[str, Any]]:
    candidates = list(rows)
    mode_norm = mode.strip().lower() or "production"
    score_key = "signal_strength"
    preference: dict[str, int] = {}
    if mode_norm in {"no_pullback_score", "no-pullback-score"}:
        candidates = [row for row in candidates if str(row.get("template")) != "pullback_trend"]
    elif mode_norm in {"raw_strength", "raw-strength"}:
        score_key = "raw_signal_strength"
    elif mode_norm in {"quality_score", "quality-score"}:
        score_key = "quality_score"
    elif mode_norm in {"no_pullback_alt_score", "no-pullback-alt-score"}:
        candidates = [row for row in candidates if str(row.get("template")) != "pullback_trend"]
        score_key = "_alt_score"
        for row in candidates:
            row["_alt_score"] = (
                0.55 * _safe_float(row.get("raw_signal_strength"))
                + 0.25 * _safe_float(row.get("quality_score"))
                + 0.20 * min(1.0, _safe_float(row.get("vol_scale")) / 0.08)
            )
    elif mode_norm in {"squeeze_then_trend", "squeeze-then-trend"}:
        candidates = [
            row
            for row in candidates
            if str(row.get("template")) in {"squeeze_breakout", "trend_breakout"}
        ]
        preference = {"squeeze_breakout": 0, "trend_breakout": 1}
    elif mode_norm in {"trend_then_squeeze", "trend-then-squeeze"}:
        candidates = [
            row
            for row in candidates
            if str(row.get("template")) in {"trend_breakout", "squeeze_breakout"}
        ]
        preference = {"trend_breakout": 0, "squeeze_breakout": 1}

    selected: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row in sorted(
        candidates,
        key=lambda item: (
            preference.get(str(item.get("template")), 0),
            -_safe_float(item.get(score_key)),
            -_safe_float(item.get("adv")),
            str(item.get("symbol", "")),
            str(item.get("template", "")),
        ),
    ):
        symbol = str(row.get("symbol", "")).upper()
        if symbol in seen_symbols:
            continue
        selected.append(row)
        seen_symbols.add(symbol)
        if top_n > 0 and len(selected) >= top_n:
            break
    return selected


def _selection_mode_summary(
    rows: list[dict[str, Any]],
    *,
    top_n: int,
    modes: list[str],
) -> dict[str, dict[str, Any]]:
    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_day[str(row.get("fill_date", ""))].append(row)

    summaries: dict[str, dict[str, Any]] = {}
    for mode in modes:
        selected: list[dict[str, Any]] = []
        for day in sorted(by_day):
            selected.extend(_ranked(by_day[day], top_n, mode=mode))
        summaries[mode] = _summarize(selected)
    return summaries


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if str(row.get("status")) in TERMINAL_STATUSES]
    winners = [row for row in completed if str(row.get("status")) in {"T1_HIT", "T2_HIT"}]
    returns = [_safe_float(row.get("close_return_pct")) for row in completed]
    return {
        "count": len(rows),
        "completed_count": len(completed),
        "status_counts": dict(Counter(str(row.get("status")) for row in rows)),
        "win_rate_pct": 100.0 * len(winners) / len(completed) if completed else 0.0,
        "avg_return_pct": statistics.fmean(returns) if returns else 0.0,
        "median_return_pct": statistics.median(returns) if returns else 0.0,
        "avg_max_favorable_pct": (
            statistics.fmean(_safe_float(row.get("max_favorable_pct")) for row in completed)
            if completed
            else 0.0
        ),
        "avg_max_adverse_pct": (
            statistics.fmean(_safe_float(row.get("max_adverse_pct")) for row in completed)
            if completed
            else 0.0
        ),
    }


def _month_key(row: dict[str, Any]) -> str:
    return str(row.get("fill_date", ""))[:7] or "UNKNOWN"


def _write_report(
    *,
    output_dir: Path,
    selected_rows: list[dict[str, Any]],
    actionable_rows: list[dict[str, Any]],
    blocked_rows: list[dict[str, Any]],
    daily_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"historical-signal-gate-audit-{stamp}.csv"
    report_path = output_dir / f"historical-signal-gate-audit-{stamp}.txt"
    export_rows: list[dict[str, Any]] = []
    for bucket, rows in (
        ("selected", selected_rows),
        ("actionable", actionable_rows),
        ("blocked", blocked_rows),
    ):
        for row in rows:
            item = dict(row)
            item["bucket"] = bucket
            export_rows.append(item)
    pd.DataFrame(export_rows).to_csv(csv_path, index=False)

    by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_month[_month_key(row)].append(row)
    lines = [
        "Atlas Historical Signal Gate Audit",
        f"bundle_id={args.bundle_id} start={args.start_date} end={args.end_date}",
        (
            f"templates={args.templates} top_n_per_day={args.top_n_per_day} "
            f"selection_mode={args.selection_mode}"
        ),
        "",
        f"selected={_summarize(selected_rows)}",
        f"actionable={_summarize(actionable_rows)}",
        f"blocked={_summarize(blocked_rows)}",
        "",
        "Monthly selected performance:",
    ]
    for month in sorted(by_month):
        lines.append(f"{month}: {_summarize(by_month[month])}")
    if bool(args.compare_selection_modes):
        modes = [
            "production",
            "no_pullback_score",
            "raw_strength",
            "quality_score",
            "squeeze_then_trend",
            "trend_then_squeeze",
            "no_pullback_alt_score",
        ]
        lines.extend(["", "Selection mode comparison:"])
        for mode, summary in _selection_mode_summary(
            actionable_rows,
            top_n=int(args.top_n_per_day),
            modes=modes,
        ).items():
            lines.append(f"{mode}: {summary}")
    lines.extend(["", "Daily counts:"])
    for row in daily_rows:
        lines.append(str(row))
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_report={report_path}")
    return report_path


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    init_db()
    settings = get_settings()
    store = _settings_store()
    template_arg = str(args.templates).strip()
    templates = [item.strip() for item in template_arg.split(",") if item.strip()]
    if template_arg.lower() == "all":
        templates = [template.key for template in list_templates()]
    if not templates:
        templates = ["trend_breakout", "squeeze_breakout"]
    ranking_weights = dict(DEFAULT_RANKING_WEIGHTS)
    with Session(engine) as session:
        symbols = store.get_bundle_symbols(session, int(args.bundle_id), timeframe="1d")
        if int(args.max_symbols) > 0:
            symbols = symbols[: int(args.max_symbols)]
        prepared = _prepare_symbols(
            session=session,
            store=store,
            symbols=symbols,
            templates=templates,
            params_overrides={},
        )
        if not prepared:
            return {}
        all_dates = sorted({day for item in prepared.values() for day in item.date_to_index})
        latest_date = all_dates[-1]
        start_date = _parse_day(args.start_date) or (latest_date - timedelta(days=int(args.lookback_days)))
        end_date = _parse_day(args.end_date) or latest_date
        fill_dates = [day for day in all_dates if start_date <= day <= end_date]
        state_settings = {
            "event_risk_enabled": True,
            "event_risk_sync_before_signals": True,
        }
        equity = float(args.equity)
        risk_per_trade = float(args.risk_per_trade or settings.risk_per_trade)
        selected_rows: list[dict[str, Any]] = []
        actionable_rows: list[dict[str, Any]] = []
        blocked_rows: list[dict[str, Any]] = []
        daily_rows: list[dict[str, Any]] = []
        for day in fill_dates:
            market_context = _market_context_from_prepared(prepared, fill_date=day)
            candidates: list[dict[str, Any]] = []
            for symbol, item in prepared.items():
                if day not in item.date_to_index:
                    continue
                for template in templates:
                    candidate = _candidate_for_fill(
                        session=session,
                        item=item,
                        symbol=symbol,
                        template=template,
                        fill_date=day,
                        market_context=market_context,
                        state_settings=state_settings,
                        equity=equity,
                        risk_per_trade=risk_per_trade,
                        ranking_weights=ranking_weights,
                        params_overrides={},
                        horizon_bars=int(args.horizon_bars),
                    )
                    if candidate is not None:
                        candidates.append(candidate)
            actionable = [
                row
                for row in candidates
                if str(row.get("quality_status", "PASS")).upper() == "PASS"
                and row.get("entry_block_reason") is None
                and int(row.get("planned_qty") or 0) > 0
            ]
            blocked = [row for row in candidates if row not in actionable]
            selected = _ranked(
                actionable,
                int(args.top_n_per_day),
                mode=str(args.selection_mode),
            )
            actionable_rows.extend(actionable)
            blocked_rows.extend(blocked)
            selected_rows.extend(selected)
            daily_rows.append(
                {
                    "fill_date": day.isoformat(),
                    "generated": len(candidates),
                    "actionable": len(actionable),
                    "selected": len(selected),
                    "market_flags": list(market_context.get("flags", [])),
                    "selected_summary": _summarize(selected),
                }
            )
            print(
                f"day={day} generated={len(candidates)} actionable={len(actionable)} "
                f"selected={len(selected)} flags={','.join(market_context.get('flags', [])) or '-'}",
                flush=True,
            )
    report_path = _write_report(
        output_dir=Path(str(args.output_dir)),
        selected_rows=selected_rows,
        actionable_rows=actionable_rows,
        blocked_rows=blocked_rows,
        daily_rows=daily_rows,
        args=args,
    )
    return {
        "selected": _summarize(selected_rows),
        "actionable": _summarize(actionable_rows),
        "blocked": _summarize(blocked_rows),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay Atlas signal gates over historical daily bars.")
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument("--templates", default="trend_breakout,squeeze_breakout")
    parser.add_argument("--top-n-per-day", type=int, default=3)
    parser.add_argument("--selection-mode", default="production")
    parser.add_argument("--compare-selection-modes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--horizon-bars", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--equity", type=float, default=1_000_000.0)
    parser.add_argument("--risk-per-trade", type=float, default=0.005)
    parser.add_argument("--output-dir", default="data/reports/signal-audits")
    args = parser.parse_args()
    summary = run_audit(args)
    print(summary)


if __name__ == "__main__":
    main()
