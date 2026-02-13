from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import optuna
import pandas as pd

from app.core.config import Settings
from app.engine.backtester import BacktestConfig, BacktestResult, run_backtest
from app.strategies.templates import StrategyTemplate


def robust_score(metrics: dict[str, float], oos_consistency: float = 0.0) -> float:
    calmar = float(metrics.get("calmar", 0.0))
    tail_norm = float(metrics.get("tail_loss_norm", 1.0))
    turnover = float(metrics.get("turnover", 0.0))
    turnover_penalty = min(1.0, turnover / 10.0)
    return (
        0.45 * calmar + 0.25 * (1.0 - tail_norm) + 0.20 * oos_consistency - 0.10 * turnover_penalty
    )


def _sampler(kind: str, seed: int | None) -> optuna.samplers.BaseSampler:
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if kind == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True)


def _pruner(kind: str) -> optuna.pruners.BasePruner:
    if kind == "none":
        return optuna.pruners.NopPruner()
    if kind == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=3)
    return optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2)


def _storage_url(settings: Settings) -> str | None:
    if settings.optuna_storage_url:
        return settings.optuna_storage_url
    if settings.database_url.startswith("postgresql"):
        return settings.database_url
    if settings.database_url.startswith("sqlite"):
        return "sqlite:///./apps/api/.atlas/optuna.db"
    return None


@dataclass
class OptimizationResult:
    params: dict[str, float | int]
    score: float
    best_trial: int
    n_trials: int
    elapsed_seconds: float


def _trial_params(trial: optuna.Trial, template: StrategyTemplate) -> dict[str, float | int]:
    params = dict(template.default_params)
    for key, (low, high) in template.param_ranges.items():
        if isinstance(low, int) and isinstance(high, int):
            params[key] = trial.suggest_int(key, int(low), int(high))
        else:
            params[key] = trial.suggest_float(key, float(low), float(high))
    return params


def _backtest_for_params(
    frame: pd.DataFrame,
    template: StrategyTemplate,
    symbol: str,
    settings: Settings,
    params: dict[str, float | int],
) -> BacktestResult:
    signals = template.signal_fn(frame, params)
    config = BacktestConfig(
        risk_per_trade=settings.risk_per_trade,
        max_positions=settings.max_positions,
        atr_stop_mult=float(params.get("atr_stop_mult", 2.0)),
        atr_trail_mult=float(params.get("atr_trail_mult", 2.0)),
        commission_bps=settings.commission_bps,
        slippage_base_bps=settings.slippage_base_bps,
        slippage_vol_factor=settings.slippage_vol_factor,
    )
    return run_backtest(price_df=frame, entries=signals, symbol=symbol, config=config)


def optimize_template_params(
    frame: pd.DataFrame,
    template: StrategyTemplate,
    symbol: str,
    settings: Settings,
    *,
    trials: int,
    timeout_seconds: int | None,
    sampler: str,
    pruner: str,
    seed: int | None,
    min_trades: int = 5,
) -> OptimizationResult:
    started = datetime.now(tz=UTC)

    study = optuna.create_study(
        direction="maximize",
        sampler=_sampler(sampler, seed),
        pruner=_pruner(pruner),
        study_name=f"atlas-{template.key}-{symbol}-{started.strftime('%Y%m%d%H%M%S%f')}",
        storage=_storage_url(settings),
        load_if_exists=False,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _trial_params(trial, template)
        result = _backtest_for_params(frame, template, symbol, settings, params)
        trade_count = len(result.trades)
        score = robust_score(result.metrics)

        # Hard floor to avoid parameter sets with no signal quality.
        if trade_count < min_trades:
            trial.set_user_attr("rejected", "too_few_trades")
            score -= 2.0

        trial.report(score, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return score

    study.optimize(objective, n_trials=trials, timeout=timeout_seconds)

    best = dict(template.default_params)
    best.update(study.best_params)

    elapsed = (datetime.now(tz=UTC) - started).total_seconds()
    return OptimizationResult(
        params=best,
        score=float(study.best_value),
        best_trial=int(study.best_trial.number),
        n_trials=len(study.trials),
        elapsed_seconds=float(elapsed),
    )
