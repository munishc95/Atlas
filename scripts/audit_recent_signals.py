from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
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
from app.engine.signal_engine import _candidate_quality  # noqa: E402
from app.services.data_store import DataStore  # noqa: E402
from app.strategies.templates import generate_signal_sides, signal_strength  # noqa: E402


CURRENT_TEMPLATES = {"trend_breakout", "squeeze_breakout"}
QUALITY_RANK = {"FAIL": 0, "WARN": 1, "PASS": 2}
SWEEP_THRESHOLDS = (0.0, 0.55, 0.60, 0.65, 0.70, 0.75)


@dataclass
class PreparedSymbol:
    frame: pd.DataFrame
    features: pd.DataFrame
    dates: pd.Series
    signals_by_template: dict[str, pd.Series]


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )


def _nearest_trading_date(available_dates: list[date], target: date) -> date | None:
    available = set(available_dates)
    for offset in range(0, 15):
        candidate = target - timedelta(days=offset)
        if candidate in available:
            return candidate
    return None


def _signal_window_dates(
    *,
    available_dates: list[date],
    decision_date: date,
    window_bars: int,
) -> list[date]:
    if decision_date not in available_dates:
        return []
    index = available_dates.index(decision_date)
    bars = max(1, int(window_bars))
    start = max(0, index - bars + 1)
    return available_dates[start : index + 1]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.nan_to_num(float(value), nan=default, posinf=default, neginf=default))
    except (TypeError, ValueError):
        return default


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "count": 0,
            "win_rate_pct": 0.0,
            "avg_pct": 0.0,
            "median_pct": 0.0,
            "best_avg_pct": 0.0,
            "worst_avg_pct": 0.0,
            "avg_quality": 0.0,
        }
    values = [_as_float(row["return_pct"]) for row in rows]
    return {
        "count": len(rows),
        "win_rate_pct": 100.0 * sum(value > 0 for value in values) / len(values),
        "avg_pct": statistics.fmean(values),
        "median_pct": statistics.median(values),
        "best_avg_pct": statistics.fmean(_as_float(row["best_pct"]) for row in rows),
        "worst_avg_pct": statistics.fmean(_as_float(row["worst_pct"]) for row in rows),
        "avg_quality": statistics.fmean(_as_float(row["quality_score"]) for row in rows),
    }


def _print_summary(label: str, rows: list[dict[str, Any]]) -> None:
    summary = _summary(rows)
    print(
        f"{label:26} n={summary['count']:4d} "
        f"win={summary['win_rate_pct']:6.2f}% "
        f"avg={summary['avg_pct']:7.2f}% "
        f"med={summary['median_pct']:7.2f}% "
        f"best={summary['best_avg_pct']:7.2f}% "
        f"worst={summary['worst_avg_pct']:7.2f}% "
        f"q={summary['avg_quality']:5.2f}"
    )


def _quality_at_least(row: dict[str, Any], status: str) -> bool:
    return QUALITY_RANK.get(str(row.get("quality_status", "FAIL")), 0) >= QUALITY_RANK[status]


def _ranked(rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    for row in sorted(
        rows,
        key=lambda row: (
            -_as_float(row.get("ranking_score")),
            -_as_float(row.get("quality_score")),
            -_as_float(row.get("adv")),
            str(row.get("symbol", "")),
            str(row.get("template", "")),
            str(row.get("signal_date", "")),
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


def _evaluate_signal(
    *,
    symbol: str,
    template: str,
    frame: pd.DataFrame,
    features: pd.DataFrame,
    decision_idx: int,
    latest_idx: int,
    max_abs_return_pct: float,
    max_path_gap_pct: float,
) -> dict[str, Any] | None:
    fill_idx = decision_idx + 1
    if fill_idx > latest_idx:
        return None
    entry = _as_float(frame.iloc[fill_idx]["open"])
    if entry <= 0:
        return None
    path = frame.iloc[fill_idx : latest_idx + 1]
    latest_close = _as_float(path.iloc[-1]["close"])
    if latest_close <= 0:
        return None
    path_gap_pct = _as_float(
        pd.to_numeric(
            frame["close"].iloc[decision_idx : latest_idx + 1],
            errors="coerce",
        )
        .pct_change()
        .abs()
        .max()
        * 100.0
    )
    if max_path_gap_pct > 0 and path_gap_pct > max_path_gap_pct:
        return None

    quality = _candidate_quality(
        frame=frame,
        features=features,
        decision_idx=decision_idx,
        side="BUY",
        template_key=template,
    )
    metrics = quality.get("quality_metrics", {})
    raw_strength = _as_float(signal_strength(frame, decision_idx))
    adv = _as_float((frame["close"] * frame["volume"]).iloc[: decision_idx + 1].tail(20).mean())
    atr_pct = _as_float(metrics.get("atr_pct"))
    liquidity_component = float(np.tanh(np.log1p(max(0.0, adv)) / 20.0))
    stability_component = 1.0 - min(1.0, max(0.0, atr_pct) * 15.0)
    quality_score = _as_float(quality.get("quality_score"))
    ranking_score = (
        0.50 * raw_strength
        + 0.25 * liquidity_component
        + 0.10 * stability_component
        + 0.15 * quality_score
    )

    return_pct = (latest_close / entry - 1.0) * 100.0
    if max_abs_return_pct > 0 and abs(return_pct) > max_abs_return_pct:
        return None
    return {
        "symbol": symbol,
        "template": template,
        "quality_status": str(quality["quality_status"]),
        "quality_score": quality_score,
        "quality_flags": list(quality["quality_flags"]),
        "close_location": _as_float(metrics.get("close_location")),
        "volume_ratio": _as_float(metrics.get("volume_ratio")),
        "atr_pct": atr_pct,
        "range_atr": _as_float(metrics.get("range_atr")),
        "path_gap_pct": path_gap_pct,
        "raw_strength": raw_strength,
        "ranking_score": ranking_score,
        "adv": adv,
        "signal_date": pd.Timestamp(frame.iloc[decision_idx]["datetime"]).date().isoformat(),
        "fill_date": pd.Timestamp(frame.iloc[fill_idx]["datetime"]).date().isoformat(),
        "latest_date": pd.Timestamp(path.iloc[-1]["datetime"]).date().isoformat(),
        "entry": entry,
        "latest_close": latest_close,
        "return_pct": return_pct,
        "best_pct": (_as_float(path["high"].max()) / entry - 1.0) * 100.0,
        "worst_pct": (_as_float(path["low"].min()) / entry - 1.0) * 100.0,
        "bars_held": int(len(path)),
    }


def _prepare_symbols(
    *,
    session: Session,
    store: DataStore,
    symbols: list[str],
    templates: list[str],
    adjustment_mode: str,
) -> tuple[dict[str, PreparedSymbol], Counter[date]]:
    prepared: dict[str, PreparedSymbol] = {}
    date_counts: Counter[date] = Counter()
    for index, symbol in enumerate(symbols, start=1):
        frame = store.load_ohlcv(
            symbol=symbol,
            timeframe="1d",
            session=session,
            adjustment_mode=adjustment_mode,
        )
        if len(frame) < 260:
            continue
        features = store.load_features(
            symbol=symbol,
            timeframe="1d",
            session=session,
            adjustment_mode=adjustment_mode,
        )
        min_len = min(len(frame), len(features))
        frame = frame.tail(min_len).reset_index(drop=True)
        features = features.tail(min_len).reset_index(drop=True)
        if len(frame) < 260:
            continue

        dates = pd.to_datetime(frame["datetime"], utc=True).dt.date
        signals_by_template = {
            template: generate_signal_sides(template, frame, params={})
            .get("BUY", pd.Series(False, index=frame.index))
            .fillna(False)
            .astype(bool)
            for template in templates
        }
        for day in set(dates.tolist()):
            date_counts[day] += 1
        prepared[symbol] = PreparedSymbol(
            frame=frame,
            features=features,
            dates=dates,
            signals_by_template=signals_by_template,
        )
        if index % 50 == 0:
            print(f"prepared_symbols={index}/{len(symbols)} usable={len(prepared)}", flush=True)
    return prepared, date_counts


def _candidate_buckets(rows: list[dict[str, Any]], *, strict_min_quality_score: float) -> dict[str, list[dict[str, Any]]]:
    current = [
        row
        for row in rows
        if str(row.get("template")) in CURRENT_TEMPLATES and _quality_at_least(row, "WARN")
    ]
    strict = [
        row
        for row in current
        if _quality_at_least(row, "PASS")
        and _as_float(row.get("quality_score")) >= strict_min_quality_score
    ]
    return {
        "baseline_all": rows,
        "current_no_fail": current,
        "strict_pass": strict,
    }


def audit(
    *,
    bundle_id: int,
    templates: list[str],
    anchors: list[int],
    top_n: int,
    max_symbols: int,
    adjustment_mode: str,
    signal_window_bars: int,
    min_coverage_pct: float,
    strict_min_quality_score: float,
    max_abs_return_pct: float,
    max_path_gap_pct: float,
) -> dict[str, Any]:
    init_db()
    store = _store()
    output: dict[str, Any] = {"anchors": []}
    with Session(engine) as session:
        symbols = store.get_bundle_symbols(session, bundle_id, timeframe="1d")
        if max_symbols > 0:
            symbols = symbols[:max_symbols]
        prepared, date_counts = _prepare_symbols(
            session=session,
            store=store,
            symbols=symbols,
            templates=templates,
            adjustment_mode=adjustment_mode,
        )
        if not prepared:
            return output

        min_count = max(1, int(len(prepared) * max(0.0, min_coverage_pct) / 100.0))
        eligible_dates = sorted(day for day, count in date_counts.items() if count >= min_count)
        if not eligible_dates:
            eligible_dates = sorted(date_counts)
        latest_date = eligible_dates[-1]

        for days_back in anchors:
            target = latest_date - timedelta(days=int(days_back))
            decision_date = _nearest_trading_date(eligible_dates, target)
            if decision_date is None:
                continue
            window_dates = _signal_window_dates(
                available_dates=eligible_dates,
                decision_date=decision_date,
                window_bars=signal_window_bars,
            )
            print(
                f"scanning_anchor={days_back}d target={target} "
                f"window={','.join(day.isoformat() for day in window_dates)}",
                flush=True,
            )
            rows: list[dict[str, Any]] = []
            for symbol, item in prepared.items():
                latest_matches = item.frame.index[item.dates == latest_date].tolist()
                if not latest_matches:
                    continue
                latest_idx = int(latest_matches[-1])
                for signal_day in window_dates:
                    decision_matches = item.frame.index[item.dates == signal_day].tolist()
                    if not decision_matches:
                        continue
                    decision_idx = int(decision_matches[-1])
                    if decision_idx + 1 > latest_idx:
                        continue
                    for template in templates:
                        buy = item.signals_by_template[template]
                        if not bool(buy.iloc[decision_idx]):
                            continue
                        row = _evaluate_signal(
                            symbol=symbol,
                            template=template,
                            frame=item.frame,
                            features=item.features,
                            decision_idx=decision_idx,
                            latest_idx=latest_idx,
                            max_abs_return_pct=max_abs_return_pct,
                            max_path_gap_pct=max_path_gap_pct,
                        )
                        if row is not None:
                            rows.append(row)

            buckets = _candidate_buckets(rows, strict_min_quality_score=strict_min_quality_score)
            output["anchors"].append(
                {
                    "days_back": int(days_back),
                    "target_date": target.isoformat(),
                    "decision_date": decision_date.isoformat(),
                    "signal_window_dates": [day.isoformat() for day in window_dates],
                    "latest_date": latest_date.isoformat(),
                    "coverage_symbols": int(date_counts[latest_date]),
                    "prepared_symbols": len(prepared),
                    "all_candidates": rows,
                    "baseline_top": _ranked(buckets["baseline_all"], top_n),
                    "current_top": _ranked(buckets["current_no_fail"], top_n),
                    "strict_top": _ranked(buckets["strict_pass"], top_n),
                    "buckets": buckets,
                }
            )
    return output


def _print_group_summaries(rows: list[dict[str, Any]]) -> None:
    by_template: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_quality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_template[str(row["template"])].append(row)
        by_quality[str(row["quality_status"])].append(row)
    for template in sorted(by_template):
        _print_summary(f"template={template}", by_template[template])
    for quality in ("PASS", "WARN", "FAIL"):
        _print_summary(f"quality={quality}", by_quality.get(quality, []))


def _print_sweep(rows: list[dict[str, Any]], top_n: int) -> None:
    current = [row for row in rows if str(row.get("template")) in CURRENT_TEMPLATES]
    print("quality_score_sweep_current_templates")
    for threshold in SWEEP_THRESHOLDS:
        filtered = [
            row
            for row in current
            if _quality_at_least(row, "PASS") and _as_float(row.get("quality_score")) >= threshold
        ]
        _print_summary(f"PASS score>={threshold:.2f}", _ranked(filtered, top_n))


def _print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print(title)
    for row in rows:
        flags = ",".join(str(flag) for flag in row.get("quality_flags", [])) or "-"
        print(
            f"  {row['symbol']:12} {row['template']:16} {row['quality_status']:4} "
            f"sig={row['signal_date']} fill={row['fill_date']} "
            f"entry={row['entry']:9.2f} latest={row['latest_close']:9.2f} "
            f"ret={row['return_pct']:7.2f}% best={row['best_pct']:7.2f}% "
            f"worst={row['worst_pct']:7.2f}% q={row['quality_score']:.2f} "
            f"rank={row['ranking_score']:.3f} flags={flags}"
        )


def _export_csv(result: dict[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for anchor in result["anchors"]:
        for row in anchor["all_candidates"]:
            item = dict(row)
            item["anchor_days_back"] = anchor["days_back"]
            item["anchor_target_date"] = anchor["target_date"]
            item["anchor_latest_date"] = anchor["latest_date"]
            item["anchor_signal_window"] = ",".join(anchor["signal_window_dates"])
            rows.append(item)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit recent Atlas signals versus the latest local bar."
    )
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--templates", default="trend_breakout,pullback_trend,squeeze_breakout")
    parser.add_argument("--anchors", default="7,14,30")
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--adjustment-mode", choices=["RAW", "ADJUSTED"], default="ADJUSTED")
    parser.add_argument("--signal-window-bars", type=int, default=3)
    parser.add_argument("--min-coverage-pct", type=float, default=75.0)
    parser.add_argument("--strict-min-quality-score", type=float, default=0.65)
    parser.add_argument("--max-abs-return-pct", type=float, default=0.0)
    parser.add_argument("--max-path-gap-pct", type=float, default=35.0)
    parser.add_argument("--export-csv", default="")
    args = parser.parse_args()

    templates = [item.strip() for item in str(args.templates).split(",") if item.strip()]
    anchors = [int(item.strip()) for item in str(args.anchors).split(",") if item.strip()]
    result = audit(
        bundle_id=int(args.bundle_id),
        templates=templates,
        anchors=anchors,
        top_n=int(args.top_n),
        max_symbols=int(args.max_symbols),
        adjustment_mode=str(args.adjustment_mode),
        signal_window_bars=int(args.signal_window_bars),
        min_coverage_pct=float(args.min_coverage_pct),
        strict_min_quality_score=float(args.strict_min_quality_score),
        max_abs_return_pct=float(args.max_abs_return_pct),
        max_path_gap_pct=float(args.max_path_gap_pct),
    )
    if str(args.export_csv).strip():
        _export_csv(result, Path(str(args.export_csv)))

    for anchor in result["anchors"]:
        print()
        print(
            f"Anchor {anchor['days_back']}d: target={anchor['target_date']} "
            f"signal_window={','.join(anchor['signal_window_dates'])} "
            f"latest={anchor['latest_date']} "
            f"coverage={anchor['coverage_symbols']}/{anchor['prepared_symbols']}"
        )
        buckets = anchor["buckets"]
        _print_summary("baseline_all", buckets["baseline_all"])
        _print_summary("baseline_top", anchor["baseline_top"])
        _print_summary("current_no_fail", buckets["current_no_fail"])
        _print_summary("current_top", anchor["current_top"])
        _print_summary("strict_pass", buckets["strict_pass"])
        _print_summary("strict_top", anchor["strict_top"])
        _print_group_summaries(anchor["all_candidates"])
        _print_sweep(anchor["all_candidates"], int(args.top_n))
        _print_rows("current_top_details", anchor["current_top"])
        _print_rows("strict_top_details", anchor["strict_top"])


if __name__ == "__main__":
    main()
