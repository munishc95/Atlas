from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
import statistics
import sys
from typing import Any

import pandas as pd
import numpy as np
from sqlmodel import Session

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import engine, init_db  # noqa: E402
from app.engine.signal_engine import _candidate_quality  # noqa: E402
from app.services.data_store import DataStore  # noqa: E402
from app.strategies.templates import generate_signal_sides, signal_strength  # noqa: E402


QUALITY_RANK = {"FAIL": 0, "WARN": 1, "PASS": 2}


def _parse_day(value: str | None) -> date | None:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "avg_best_pct": 0.0,
            "avg_worst_pct": 0.0,
        }
    returns = [float(row["return_pct"]) for row in rows]
    return {
        "count": len(rows),
        "win_rate_pct": 100.0 * sum(value > 0 for value in returns) / len(returns),
        "avg_return_pct": statistics.fmean(returns),
        "median_return_pct": statistics.median(returns),
        "avg_best_pct": statistics.fmean(float(row["best_pct"]) for row in rows),
        "avg_worst_pct": statistics.fmean(float(row["worst_pct"]) for row in rows),
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.nan_to_num(float(value), nan=default, posinf=default, neginf=default))
    except (TypeError, ValueError):
        return default


def _print_summary(title: str, rows: list[dict[str, Any]]) -> None:
    summary = _summarize(rows)
    print(
        f"{title:34} "
        f"n={summary['count']:5d} "
        f"win={summary['win_rate_pct']:6.2f}% "
        f"avg={summary['avg_return_pct']:7.2f}% "
        f"med={summary['median_return_pct']:7.2f}% "
        f"best={summary['avg_best_pct']:7.2f}% "
        f"worst={summary['avg_worst_pct']:7.2f}%"
    )


def _ranked(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row in sorted(
        rows,
        key=lambda item: (
            -_safe_float(item.get("ranking_score")),
            -_safe_float(item.get("quality_score")),
            -_safe_float(item.get("adv")),
            str(item.get("symbol", "")),
            str(item.get("template", "")),
        ),
    ):
        symbol = str(row.get("symbol", "")).upper()
        if symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        selected.append(row)
        if top_n > 0 and len(selected) >= top_n:
            break
    return selected


def _select_top_by_day(
    rows: list[dict[str, Any]],
    *,
    top_n_per_day: int,
    min_quality_status: str,
) -> list[dict[str, Any]]:
    if top_n_per_day <= 0:
        return []
    min_rank = QUALITY_RANK[min_quality_status]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if QUALITY_RANK.get(str(row.get("quality_status", "FAIL")), 0) < min_rank:
            continue
        grouped[str(row["signal_date"])].append(row)
    selected: list[dict[str, Any]] = []
    for signal_date in sorted(grouped):
        selected.extend(_ranked(grouped[signal_date], top_n_per_day))
    return selected


def evaluate(
    *,
    bundle_id: int,
    timeframe: str,
    templates: list[str],
    horizon_bars: int,
    max_symbols: int,
    start_date: date | None,
    end_date: date | None,
    max_abs_return_pct: float,
    adjustment_mode: str,
    max_path_gap_pct: float,
) -> list[dict[str, Any]]:
    init_db()
    store = _store()
    rows: list[dict[str, Any]] = []
    with Session(engine) as session:
        symbols = store.get_bundle_symbols(session, bundle_id, timeframe=timeframe)
        if max_symbols > 0:
            symbols = symbols[:max_symbols]

        for index, symbol in enumerate(symbols, start=1):
            frame = store.load_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                session=session,
                adjustment_mode=adjustment_mode,
            )
            if len(frame) < 260:
                continue
            features = store.load_features(
                symbol=symbol,
                timeframe=timeframe,
                session=session,
                adjustment_mode=adjustment_mode,
            )
            min_len = min(len(frame), len(features))
            frame = frame.tail(min_len).reset_index(drop=True)
            features = features.tail(min_len).reset_index(drop=True)
            for template in templates:
                sides = generate_signal_sides(template, frame, params={})
                buy = sides.get("BUY", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
                for decision_idx in frame.index[buy]:
                    decision_idx = int(decision_idx)
                    fill_idx = decision_idx + 1
                    exit_idx = fill_idx + max(1, horizon_bars) - 1
                    if fill_idx >= len(frame) or exit_idx >= len(frame):
                        continue
                    signal_day = pd.Timestamp(frame.iloc[decision_idx]["datetime"]).date()
                    if start_date is not None and signal_day < start_date:
                        continue
                    if end_date is not None and signal_day > end_date:
                        continue
                    entry = float(frame.iloc[fill_idx]["open"])
                    if entry <= 0:
                        continue
                    path = frame.iloc[fill_idx : exit_idx + 1]
                    path_gap_pct = _safe_float(
                        pd.to_numeric(
                            frame["close"].iloc[decision_idx : exit_idx + 1],
                            errors="coerce",
                        )
                        .pct_change()
                        .abs()
                        .max()
                        * 100.0
                    )
                    if max_path_gap_pct > 0 and path_gap_pct > max_path_gap_pct:
                        continue
                    exit_close = float(path.iloc[-1]["close"])
                    best = float(path["high"].max())
                    worst = float(path["low"].min())
                    return_pct = (exit_close / entry - 1.0) * 100.0
                    if max_abs_return_pct > 0 and abs(return_pct) > max_abs_return_pct:
                        continue
                    quality = _candidate_quality(
                        frame=frame,
                        features=features,
                        decision_idx=decision_idx,
                        side="BUY",
                        template_key=template,
                    )
                    metrics = quality.get("quality_metrics", {})
                    raw_strength = _safe_float(signal_strength(frame, decision_idx))
                    adv = _safe_float(
                        (frame["close"] * frame["volume"]).iloc[: decision_idx + 1].tail(20).mean()
                    )
                    atr_pct = _safe_float(metrics.get("atr_pct"))
                    liquidity_component = float(np.tanh(np.log1p(max(0.0, adv)) / 20.0))
                    stability_component = 1.0 - min(1.0, max(0.0, atr_pct) * 15.0)
                    quality_score = _safe_float(quality.get("quality_score"))
                    ranking_score = (
                        0.50 * raw_strength
                        + 0.25 * liquidity_component
                        + 0.10 * stability_component
                        + 0.15 * quality_score
                    )
                    rows.append(
                        {
                            "symbol": symbol,
                            "template": template,
                            "signal_date": signal_day.isoformat(),
                            "fill_date": pd.Timestamp(frame.iloc[fill_idx]["datetime"])
                            .date()
                            .isoformat(),
                            "quality_status": quality["quality_status"],
                            "quality_score": quality_score,
                            "quality_flags": ",".join(str(item) for item in quality["quality_flags"]),
                            "raw_strength": raw_strength,
                            "ranking_score": ranking_score,
                            "adv": adv,
                            "close_location": _safe_float(metrics.get("close_location")),
                            "volume_ratio": _safe_float(metrics.get("volume_ratio")),
                            "atr_pct": atr_pct,
                            "range_atr": _safe_float(metrics.get("range_atr")),
                            "path_gap_pct": path_gap_pct,
                            "return_pct": return_pct,
                            "best_pct": (best / entry - 1.0) * 100.0,
                            "worst_pct": (worst / entry - 1.0) * 100.0,
                        }
                    )
            if index % 100 == 0:
                print(f"processed_symbols={index} collected_signals={len(rows)}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Atlas signal reliability from local OHLCV.")
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--templates", default="trend_breakout,pullback_trend,squeeze_breakout")
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--max-abs-return-pct", type=float, default=30.0)
    parser.add_argument("--max-path-gap-pct", type=float, default=35.0)
    parser.add_argument("--adjustment-mode", choices=["RAW", "ADJUSTED"], default="RAW")
    parser.add_argument("--top-n-per-day", type=int, default=0)
    parser.add_argument("--selection-min-quality", choices=["PASS", "WARN", "FAIL"], default="WARN")
    parser.add_argument("--export-csv", default="")
    args = parser.parse_args()

    templates = [item.strip() for item in str(args.templates).split(",") if item.strip()]
    rows = evaluate(
        bundle_id=int(args.bundle_id),
        timeframe=str(args.timeframe),
        templates=templates,
        horizon_bars=int(args.horizon_bars),
        max_symbols=int(args.max_symbols),
        start_date=_parse_day(args.start_date),
        end_date=_parse_day(args.end_date),
        max_abs_return_pct=float(args.max_abs_return_pct),
        adjustment_mode=str(args.adjustment_mode),
        max_path_gap_pct=float(args.max_path_gap_pct),
    )
    if str(args.export_csv).strip():
        output_path = Path(str(args.export_csv))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)

    print()
    _print_summary("ALL", rows)
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_template_quality: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        template = str(row["template"])
        quality = str(row["quality_status"])
        by_template[template].append(row)
        by_quality[quality].append(row)
        by_template_quality[(template, quality)].append(row)
    for key in sorted(by_template):
        _print_summary(key, by_template[key])
    for key in ("PASS", "WARN", "FAIL"):
        _print_summary(f"quality={key}", by_quality.get(key, []))
    for template, quality in sorted(by_template_quality):
        _print_summary(f"{template} / {quality}", by_template_quality[(template, quality)])

    selected = _select_top_by_day(
        rows,
        top_n_per_day=int(args.top_n_per_day),
        min_quality_status=str(args.selection_min_quality),
    )
    if selected:
        print()
        _print_summary(
            f"selected_top{int(args.top_n_per_day)}/day quality>={args.selection_min_quality}",
            selected,
        )
        selected_by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
        selected_by_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            selected_by_template[str(row["template"])].append(row)
            selected_by_quality[str(row["quality_status"])].append(row)
        for key in sorted(selected_by_template):
            _print_summary(f"selected {key}", selected_by_template[key])
        for key in ("PASS", "WARN", "FAIL"):
            _print_summary(f"selected quality={key}", selected_by_quality.get(key, []))


if __name__ == "__main__":
    main()
