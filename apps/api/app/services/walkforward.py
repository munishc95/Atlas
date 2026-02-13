from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import WalkForwardFold, WalkForwardRun
from app.engine.backtester import BacktestConfig, run_backtest
from app.engine.optimizer import optimize_template_params, robust_score
from app.services.data_store import DataStore
from app.strategies.templates import get_template

ProgressCallback = Callable[[int, str | None], None]


def _window_defaults(timeframe: str) -> tuple[int, int, int]:
    if timeframe in {"4h_ish", "2h"}:
        return (24, 4, 2)
    return (60, 9, 3)


def _build_folds(
    index: pd.DatetimeIndex,
    train_m: int,
    test_m: int,
    step_m: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    if len(index) == 0:
        return []

    folds: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    cursor = index.min().normalize()
    max_date = index.max().normalize()

    while True:
        train_end = cursor + pd.DateOffset(months=train_m)
        test_end = train_end + pd.DateOffset(months=test_m)
        if test_end > max_date:
            break
        folds.append((cursor, train_end, test_end))
        cursor = cursor + pd.DateOffset(months=step_m)

    return folds


def _with_adaptive_windows(
    frame_index: pd.DatetimeIndex,
    train_m: int,
    test_m: int,
    step_m: int,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]], int, int, int]:
    folds = _build_folds(frame_index, train_m, test_m, step_m)
    if folds:
        return folds, train_m, test_m, step_m

    months_span = max(
        1,
        (frame_index.max().year - frame_index.min().year) * 12
        + (frame_index.max().month - frame_index.min().month),
    )
    adaptive_train = max(12, min(train_m, int(months_span * 0.6)))
    adaptive_test = max(3, min(test_m, int(months_span * 0.25)))
    adaptive_step = max(1, min(step_m, max(1, adaptive_test // 2)))

    folds = _build_folds(frame_index, adaptive_train, adaptive_test, adaptive_step)
    return folds, adaptive_train, adaptive_test, adaptive_step


def _emit(progress_cb: ProgressCallback | None, progress: int, message: str | None = None) -> None:
    if progress_cb is not None:
        progress_cb(progress, message)


def execute_walkforward(
    session: Session,
    store: DataStore,
    settings: Settings,
    payload: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    symbol = str(payload["symbol"]).upper()
    timeframe = str(payload.get("timeframe", "1d"))
    template = get_template(str(payload["strategy_template"]))

    _emit(progress_cb, 12, f"Loading OHLCV for {symbol} ({timeframe})")
    frame = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
    if frame.empty:
        raise APIError(code="missing_data", message="No data available for walk-forward run")

    frame = frame.copy().sort_values("datetime")
    frame = frame.set_index(pd.to_datetime(frame["datetime"], utc=True))

    train_m, test_m, step_m = _window_defaults(timeframe)
    cfg = payload.get("config", {})
    train_m = int(cfg.get("train_months", train_m))
    test_m = int(cfg.get("test_months", test_m))
    step_m = int(cfg.get("step_months", step_m))

    folds, train_m, test_m, step_m = _with_adaptive_windows(frame.index, train_m, test_m, step_m)
    if not folds:
        raise APIError(code="insufficient_data", message="Not enough data for configured walk-forward windows")

    run = WalkForwardRun(config_json=payload, summary_json={})
    session.add(run)
    session.commit()
    session.refresh(run)

    trials = int(cfg.get("trials", settings.optuna_default_trials))
    timeout_seconds = cfg.get("timeout_seconds", settings.optuna_default_timeout_seconds)
    timeout_seconds = int(timeout_seconds) if timeout_seconds is not None else None
    sampler = str(cfg.get("sampler", "tpe"))
    pruner = str(cfg.get("pruner", "median"))
    seed = cfg.get("seed")
    seed = int(seed) if seed is not None else None

    fold_rows: list[dict[str, Any]] = []
    oos_scores: list[float] = []
    stress_failures = 0

    for idx, (train_start, train_end, test_end) in enumerate(folds, start=1):
        fold_progress = 15 + int((idx - 1) / max(1, len(folds)) * 70)
        _emit(
            progress_cb,
            fold_progress,
            f"Fold {idx}/{len(folds)} optimize: {train_start.date()} to {train_end.date()}",
        )

        train_frame = frame[(frame.index >= train_start) & (frame.index < train_end)]
        test_frame = frame[(frame.index >= train_end) & (frame.index < test_end)]
        if len(train_frame) < 120 or len(test_frame) < 30:
            continue

        optimization = optimize_template_params(
            frame=train_frame,
            template=template,
            symbol=symbol,
            settings=settings,
            trials=trials,
            timeout_seconds=timeout_seconds,
            sampler=sampler,
            pruner=pruner,
            seed=seed,
            min_trades=int(cfg.get("min_train_trades", 5)),
        )
        best_params = optimization.params

        signals_train = template.signal_fn(train_frame, best_params)
        bt_train = run_backtest(
            price_df=train_frame,
            entries=signals_train,
            symbol=symbol,
            config=BacktestConfig(
                risk_per_trade=settings.risk_per_trade,
                max_positions=settings.max_positions,
                atr_stop_mult=float(best_params.get("atr_stop_mult", 2.0)),
                atr_trail_mult=float(best_params.get("atr_trail_mult", 2.0)),
                commission_bps=settings.commission_bps,
                slippage_base_bps=settings.slippage_base_bps,
                slippage_vol_factor=settings.slippage_vol_factor,
            ),
        )

        signals_test = template.signal_fn(test_frame, best_params)
        bt_test = run_backtest(
            price_df=test_frame,
            entries=signals_test,
            symbol=symbol,
            config=BacktestConfig(
                risk_per_trade=settings.risk_per_trade,
                max_positions=settings.max_positions,
                atr_stop_mult=float(best_params.get("atr_stop_mult", 2.0)),
                atr_trail_mult=float(best_params.get("atr_trail_mult", 2.0)),
                commission_bps=settings.commission_bps,
                slippage_base_bps=settings.slippage_base_bps,
                slippage_vol_factor=settings.slippage_vol_factor,
            ),
        )

        stress_cfg = BacktestConfig(
            risk_per_trade=settings.risk_per_trade,
            max_positions=settings.max_positions,
            atr_stop_mult=float(best_params.get("atr_stop_mult", 2.0)),
            atr_trail_mult=float(best_params.get("atr_trail_mult", 2.0)),
            commission_bps=settings.commission_bps * 2,
            slippage_base_bps=settings.slippage_base_bps * 2,
            slippage_vol_factor=settings.slippage_vol_factor,
        )
        delayed_entries = pd.Series(signals_test.shift(1), index=test_frame.index, dtype="boolean").fillna(
            False
        )
        bt_stress = run_backtest(
            price_df=test_frame,
            entries=delayed_entries.astype(bool),
            symbol=symbol,
            config=stress_cfg,
        )
        if bt_stress.metrics.get("calmar", 0.0) < bt_test.metrics.get("calmar", 0.0) * 0.25:
            stress_failures += 1

        oos_score = robust_score(bt_test.metrics)
        oos_scores.append(oos_score)

        fold_payload = {
            "fold_index": idx,
            "train_start": train_start.date().isoformat(),
            "train_end": train_end.date().isoformat(),
            "test_end": test_end.date().isoformat(),
            "params": best_params,
            "optimization": {
                "best_score": optimization.score,
                "best_trial": optimization.best_trial,
                "n_trials": optimization.n_trials,
                "elapsed_seconds": optimization.elapsed_seconds,
            },
            "train_metrics": bt_train.metrics,
            "test_metrics": bt_test.metrics,
            "stress_metrics": bt_stress.metrics,
            "oos_score": oos_score,
            "stress_pass": bt_stress.metrics.get("calmar", 0.0)
            >= bt_test.metrics.get("calmar", 0.0) * 0.25,
        }
        fold_rows.append(fold_payload)

        session.add(
            WalkForwardFold(
                run_id=run.id,
                fold_start=train_end.date(),
                fold_end=test_end.date(),
                params_json=best_params,
                metrics_json={
                    "optimization": fold_payload["optimization"],
                    "train": bt_train.metrics,
                    "test": bt_test.metrics,
                    "stress": bt_stress.metrics,
                    "oos_score": oos_score,
                },
            )
        )

    if not fold_rows:
        raise APIError(code="no_valid_folds", message="No fold had enough observations after window split")

    session.commit()

    oos_consistency = float(sum(score > 0 for score in oos_scores) / len(oos_scores))
    oos_max_dd = min(float(f["test_metrics"].get("max_drawdown", 0.0)) for f in fold_rows)
    too_few_trades = any(float(f["test_metrics"].get("turnover", 0.0)) < 0.01 for f in fold_rows)
    fold_profitability_pct = float(sum(float(f["oos_score"]) > 0 for f in fold_rows) / len(fold_rows))
    worst_fold_drawdown = float(min(float(f["test_metrics"].get("max_drawdown", 0.0)) for f in fold_rows))
    stress_pass_rate = float(sum(bool(f["stress_pass"]) for f in fold_rows) / len(fold_rows))

    stability_components: list[float] = []
    numeric_param_keys = {
        key
        for fold in fold_rows
        for key, value in fold["params"].items()
        if isinstance(value, (int, float))
    }
    for key in numeric_param_keys:
        values = np.array([float(f["params"].get(key, 0.0)) for f in fold_rows], dtype=float)
        mean = float(np.mean(np.abs(values)))
        std = float(np.std(values))
        cv = std / max(mean, 1e-9)
        stability_components.append(1.0 / (1.0 + cv))
    parameter_stability_score = float(np.mean(stability_components)) if stability_components else 0.0

    promoted_ok = True
    reasons: list[str] = []
    if abs(oos_max_dd) > float(cfg.get("max_oos_drawdown", 0.2)):
        promoted_ok = False
        reasons.append("oos_drawdown_exceeded")
    if too_few_trades:
        promoted_ok = False
        reasons.append("too_few_trades")
    if stress_failures > max(1, len(fold_rows) // 3):
        promoted_ok = False
        reasons.append("stress_test_collapse")

    _emit(progress_cb, 92, "Compiling walk-forward summary")
    summary = {
        "run_id": run.id,
        "fold_count": len(fold_rows),
        "windows_used": {
            "train_months": train_m,
            "test_months": test_m,
            "step_months": step_m,
        },
        "optimization": {
            "trials": trials,
            "timeout_seconds": timeout_seconds,
            "sampler": sampler,
            "pruner": pruner,
            "seed": seed,
        },
        "oos_consistency": oos_consistency,
        "oos_score_mean": float(np.mean(oos_scores)),
        "oos_score_std": float(np.std(oos_scores)),
        "oos_only": {
            "fold_profitability_pct": fold_profitability_pct,
            "worst_fold_drawdown": worst_fold_drawdown,
            "parameter_stability_score": parameter_stability_score,
            "stress_pass_rate": stress_pass_rate,
        },
        "stress_failures": stress_failures,
        "eligible_for_promotion": promoted_ok,
        "rejection_reasons": reasons,
        "folds": fold_rows,
    }

    run.summary_json = summary
    session.add(run)
    session.commit()
    session.refresh(run)
    _emit(progress_cb, 98, "Walk-forward completed")

    return summary
