from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import mean, pstdev
from typing import Any, Callable

import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import Dataset, DatasetBundle, Policy, ResearchCandidate, ResearchRun
from app.services.data_store import DataStore
from app.services.fast_mode import (
    clamp_job_timeout_seconds,
    clamp_optuna_trials,
    clamp_scan_symbols,
    prefer_sample_bundle_id,
    prefer_sample_dataset_id,
    resolve_seed,
)
from app.services.walkforward import execute_walkforward
from app.strategies.templates import list_templates

ProgressCallback = Callable[[int, str | None], None]

REGIME_TEMPLATE_PREFS: dict[str, list[str]] = {
    "TREND_UP": ["trend_breakout", "pullback_trend"],
    "RANGE": ["pullback_trend", "squeeze_breakout"],
    "HIGH_VOL": ["squeeze_breakout", "pullback_trend"],
    "RISK_OFF": [],
}

REGIME_RISK_SCALE: dict[str, float] = {
    "TREND_UP": 1.0,
    "RANGE": 0.8,
    "HIGH_VOL": 0.6,
    "RISK_OFF": 0.0,
}
REGIME_POSITION_SCALE: dict[str, float] = {
    "TREND_UP": 1.0,
    "RANGE": 0.75,
    "HIGH_VOL": 0.6,
    "RISK_OFF": 0.0,
}


def _emit(progress_cb: ProgressCallback | None, progress: int, message: str | None = None) -> None:
    if progress_cb is not None:
        progress_cb(progress, message)


def _clean_timeframes(timeframes: list[str] | None) -> list[str]:
    values = [str(item).strip() for item in (timeframes or ["1d"]) if str(item).strip()]
    if not values:
        return ["1d"]
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _resolve_templates(requested: list[str] | None) -> list[str]:
    available = {template.key for template in list_templates()}
    selected = [str(item).strip() for item in (requested or []) if str(item).strip()]
    if not selected:
        selected = list(available)
    invalid = [item for item in selected if item not in available]
    if invalid:
        raise APIError(
            code="invalid_strategy_templates",
            message=f"Unknown strategy template(s): {', '.join(sorted(invalid))}",
        )
    # Preserve request order for reproducibility.
    deduped: list[str] = []
    for item in selected:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _sample_symbols_by_adv(
    *,
    session: Session,
    store: DataStore,
    primary_timeframe: str,
    max_symbols: int,
) -> list[str]:
    rows = session.exec(
        select(Dataset.symbol)
        .where(Dataset.timeframe == primary_timeframe)
        .order_by(Dataset.created_at.desc())
    ).all()
    symbols = list(dict.fromkeys(rows))
    if not symbols:
        return []

    scored: list[tuple[str, float]] = []
    for symbol in symbols:
        frame = store.load_ohlcv(symbol=symbol, timeframe=primary_timeframe)
        if frame.empty:
            continue
        adv = (frame["close"] * frame["volume"]).tail(20).mean()
        scored.append((symbol, float(np.nan_to_num(adv, nan=0.0))))

    scored.sort(key=lambda item: item[1], reverse=True)
    limit = len(scored) if max_symbols <= 0 else min(max_symbols, len(scored))
    return [symbol for symbol, _ in scored[:limit]]


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(pstdev(values))


@dataclass
class CandidateEvaluation:
    score: float
    accepted: bool
    explanations: list[str]


def evaluate_candidate_robustness(
    *,
    oos_metrics: dict[str, float],
    stress_metrics: dict[str, float],
    param_dispersion: float,
    fold_variance: float,
    stress_pass_rate: float,
    constraints: dict[str, Any],
) -> CandidateEvaluation:
    """Deterministic candidate score with explicit penalties and hard gating."""
    oos_calmar = float(oos_metrics.get("calmar", 0.0))
    oos_cvar = abs(float(oos_metrics.get("cvar_95", 0.0)))
    oos_turnover = float(oos_metrics.get("turnover", 0.0))
    oos_consistency = float(oos_metrics.get("consistency", 0.0))
    avg_trades = float(oos_metrics.get("trade_count", 0.0))
    max_dd = abs(float(oos_metrics.get("max_drawdown", 0.0)))

    max_drawdown_threshold = float(constraints.get("max_drawdown", 0.2))
    min_trades = int(constraints.get("min_trades", 20))
    stress_threshold = float(constraints.get("stress_pass_rate_threshold", 0.6))

    tail_norm = min(1.0, oos_cvar / 0.05)
    turnover_penalty = min(1.0, oos_turnover / 12.0)
    low_trade_penalty = min(1.0, max(0.0, (min_trades - avg_trades) / max(1.0, float(min_trades))))
    instability_penalty = min(1.0, param_dispersion) * 0.6 + min(1.0, fold_variance) * 0.4
    stress_penalty = min(1.0, max(0.0, 1.0 - stress_pass_rate))

    score = (
        0.45 * oos_calmar
        + 0.2 * oos_consistency
        + 0.15 * (1.0 - tail_norm)
        - 0.08 * turnover_penalty
        - 0.06 * low_trade_penalty
        - 0.06 * instability_penalty
        - 0.1 * stress_penalty
    )

    accepted = True
    explanations: list[str] = []

    if max_dd > max_drawdown_threshold:
        accepted = False
        explanations.append(
            f"Rejected: OOS max drawdown {max_dd:.3f} exceeds threshold {max_drawdown_threshold:.3f}."
        )
    if avg_trades < min_trades:
        accepted = False
        explanations.append(
            f"Rejected: trade count {avg_trades:.1f} is below minimum {min_trades}."
        )
    if stress_pass_rate < stress_threshold:
        accepted = False
        explanations.append(
            f"Rejected: stress pass rate {stress_pass_rate:.2f} is below threshold {stress_threshold:.2f}."
        )

    if not accepted:
        score = min(score, -0.5)
    else:
        explanations.append(
            f"Accepted: OOS Calmar {oos_calmar:.3f} with stress pass rate {stress_pass_rate:.2f}."
        )

    if param_dispersion > 0.55:
        explanations.append(
            f"Penalized: parameter instability (dispersion={param_dispersion:.3f}) across folds."
        )
    if fold_variance > 0.35:
        explanations.append(f"Penalized: fold score variance is high ({fold_variance:.3f}).")
    if float(stress_metrics.get("calmar", 0.0)) < float(oos_metrics.get("calmar", 0.0)) * 0.5:
        explanations.append("Penalized: stress Calmar materially below OOS Calmar.")

    return CandidateEvaluation(score=float(score), accepted=accepted, explanations=explanations)


def _flatten_candidate_metrics(
    summary: dict[str, Any],
) -> tuple[dict[str, float], dict[str, float], float, float]:
    folds = list(summary.get("folds", []))
    if not folds:
        return {}, {}, 1.0, 1.0

    test_metrics = [dict(item.get("test_metrics", {})) for item in folds]
    stress_metrics = [dict(item.get("stress_metrics", {})) for item in folds]
    oos_scores = [float(item.get("oos_score", 0.0)) for item in folds]
    test_trades = [float(item.get("test_trade_count", 0.0)) for item in folds]
    consistency = float(summary.get("oos_consistency", 0.0))
    stability_score = float(summary.get("oos_only", {}).get("parameter_stability_score", 0.0))

    aggregated_oos = {
        "calmar": _avg([float(row.get("calmar", 0.0)) for row in test_metrics]),
        "max_drawdown": min(float(row.get("max_drawdown", 0.0)) for row in test_metrics),
        "cvar_95": _avg([float(row.get("cvar_95", 0.0)) for row in test_metrics]),
        "turnover": _avg([float(row.get("turnover", 0.0)) for row in test_metrics]),
        "trade_count": _avg(test_trades),
        "consistency": consistency,
        "oos_score_mean": _avg(oos_scores),
        "oos_score_std": _std(oos_scores),
    }
    aggregated_stress = {
        "calmar": _avg([float(row.get("calmar", 0.0)) for row in stress_metrics]),
        "max_drawdown": min(float(row.get("max_drawdown", 0.0)) for row in stress_metrics),
        "cvar_95": _avg([float(row.get("cvar_95", 0.0)) for row in stress_metrics]),
    }
    param_dispersion = float(max(0.0, 1.0 - stability_score))
    fold_variance = float(_std(oos_scores))
    return aggregated_oos, aggregated_stress, param_dispersion, fold_variance


def _best_params(summary: dict[str, Any]) -> dict[str, Any]:
    folds = list(summary.get("folds", []))
    if not folds:
        return {}
    best = max(folds, key=lambda row: float(row.get("oos_score", -10_000.0)))
    params = best.get("params", {})
    return dict(params) if isinstance(params, dict) else {}


def _policy_from_candidates(run_id: int, candidates: list[ResearchCandidate]) -> dict[str, Any]:
    accepted = [row for row in candidates if row.accepted]
    accepted.sort(key=lambda row: row.score, reverse=True)

    by_template: dict[str, list[ResearchCandidate]] = {}
    for row in accepted:
        by_template.setdefault(row.strategy_key, []).append(row)

    regime_map: dict[str, Any] = {}
    notes: list[str] = []
    for regime, preferred_templates in REGIME_TEMPLATE_PREFS.items():
        chosen: ResearchCandidate | None = None
        for template in preferred_templates:
            template_rows = by_template.get(template, [])
            if template_rows:
                top = template_rows[0]
                if chosen is None or top.score > chosen.score:
                    chosen = top
        if chosen is None:
            regime_map[regime] = {
                "strategy_key": None,
                "params": {},
                "risk_scale": REGIME_RISK_SCALE[regime],
                "max_positions_scale": REGIME_POSITION_SCALE[regime],
                "reason": "No robust accepted candidate for this regime.",
            }
            notes.append(f"{regime}: no accepted candidate met constraints.")
            continue

        regime_map[regime] = {
            "strategy_key": chosen.strategy_key,
            "params": chosen.best_params_json,
            "timeframe": chosen.timeframe,
            "symbol_reference": chosen.symbol,
            "risk_scale": REGIME_RISK_SCALE[regime],
            "max_positions_scale": REGIME_POSITION_SCALE[regime],
            "reason": (
                f"Selected {chosen.strategy_key} from {chosen.symbol}/{chosen.timeframe} "
                f"(score={chosen.score:.3f}, stress_pass={chosen.stress_pass_rate:.2f})."
            ),
        }
        notes.append(
            f"{regime}: selected {chosen.strategy_key} using robust score {chosen.score:.3f}."
        )

    return {
        "version": "v1.4",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_research_run_id": run_id,
        "universe": {
            "bundle_id": None,
            "dataset_id": None,
            "symbol_scope": "liquid",
            "max_symbols_scan": 50,
        },
        "timeframes": ["1d"],
        "ranking": {
            "method": "robust_score",
            "seed": 7,
            "weights": {"signal": 0.65, "liquidity": 0.25, "stability": 0.10},
        },
        "allowed_instruments": {
            "BUY": ["EQUITY_CASH", "STOCK_FUT", "INDEX_FUT"],
            "SELL": ["EQUITY_CASH", "STOCK_FUT", "INDEX_FUT"],
        },
        "cost_model": {"enabled": False, "mode": "delivery"},
        "regime_map": regime_map,
        "notes": notes,
    }


def execute_research_run(
    *,
    session: Session,
    store: DataStore,
    settings: Settings,
    payload: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    timeframes = _clean_timeframes(payload.get("timeframes"))
    templates = _resolve_templates(payload.get("strategy_templates"))
    cfg = dict(payload.get("config", {}))
    primary_timeframe = timeframes[0]
    symbol_scope = str(payload.get("symbol_scope", "liquid")).strip().lower() or "liquid"
    requested_bundle_id = payload.get("bundle_id")

    trials_per_strategy = clamp_optuna_trials(
        settings=settings,
        requested=int(
            cfg.get("trials_per_strategy", max(20, settings.optuna_default_trials // 3))
        ),
    )
    max_symbols = clamp_scan_symbols(
        settings=settings,
        requested=int(cfg.get("max_symbols", 12)),
    )
    max_evaluations = int(cfg.get("max_evaluations", 0))
    timeout_seconds = cfg.get("timeout_seconds", settings.optuna_default_timeout_seconds)
    timeout_seconds = int(timeout_seconds) if timeout_seconds is not None else None
    timeout_seconds = clamp_job_timeout_seconds(settings=settings, requested=timeout_seconds)
    sampler = str(cfg.get("sampler", "tpe"))
    pruner = str(cfg.get("pruner", "median"))
    seed = resolve_seed(settings=settings, value=cfg.get("seed"), default=7)

    constraints = {
        "max_drawdown": float(cfg.get("max_drawdown", 0.2)),
        "min_trades": int(cfg.get("min_trades", 20)),
        "stress_pass_rate_threshold": float(cfg.get("stress_pass_rate_threshold", 0.6)),
    }

    _emit(progress_cb, 8, "Creating research run")
    resolved_bundle_id: int | None = None
    try:
        if requested_bundle_id is not None:
            parsed_bundle = int(requested_bundle_id)
            if parsed_bundle > 0:
                resolved_bundle_id = parsed_bundle
    except (TypeError, ValueError):
        resolved_bundle_id = None
    if resolved_bundle_id is None:
        resolved_bundle_id = prefer_sample_bundle_id(session, settings=settings)
    resolved_dataset_id: int | None = None
    try:
        if payload.get("dataset_id") is not None:
            parsed_dataset = int(payload.get("dataset_id"))
            if parsed_dataset > 0:
                resolved_dataset_id = parsed_dataset
    except (TypeError, ValueError):
        resolved_dataset_id = None
    if resolved_dataset_id is None and resolved_bundle_id is None:
        resolved_dataset_id = prefer_sample_dataset_id(
            session,
            settings=settings,
            timeframe=primary_timeframe,
        )

    run = ResearchRun(
        dataset_id=resolved_dataset_id,
        bundle_id=resolved_bundle_id,
        timeframes_json=timeframes,
        config_json={
            "symbol_scope": symbol_scope,
            "trials_per_strategy": trials_per_strategy,
            "max_symbols": max_symbols,
            "max_evaluations": max_evaluations,
            "timeout_seconds": timeout_seconds,
            "sampler": sampler,
            "pruner": pruner,
            "seed": int(seed),
            "constraints": constraints,
            "objective": str(cfg.get("objective", "oos_robustness")),
        },
        status="RUNNING",
        summary_json={},
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    _emit(progress_cb, 12, f"Selecting symbols by {primary_timeframe} ADV")
    if run.bundle_id is not None:
        bundle = session.get(DatasetBundle, run.bundle_id)
        if bundle is None:
            run.status = "FAILED"
            run.summary_json = {
                "error": {
                    "code": "bundle_not_found",
                    "message": f"Bundle {run.bundle_id} was not found.",
                }
            }
            session.add(run)
            session.commit()
            raise APIError(
                code="bundle_not_found", message=f"Bundle {run.bundle_id} was not found."
            )
        symbols = store.sample_bundle_symbols(
            session=session,
            bundle_id=run.bundle_id,
            timeframe=timeframes[0],
            symbol_scope=symbol_scope,
            max_symbols_scan=max_symbols,
            seed=seed,
        )
    elif run.dataset_id is not None:
        dataset = session.get(Dataset, run.dataset_id)
        if dataset is None:
            run.status = "FAILED"
            run.summary_json = {
                "error": {
                    "code": "dataset_not_found",
                    "message": f"Dataset {run.dataset_id} was not found.",
                }
            }
            session.add(run)
            session.commit()
            raise APIError(
                code="dataset_not_found", message=f"Dataset {run.dataset_id} was not found."
            )
        if dataset.bundle_id is not None:
            run.bundle_id = dataset.bundle_id
            session.add(run)
            session.commit()
            symbols = store.sample_bundle_symbols(
                session=session,
                bundle_id=int(dataset.bundle_id),
                timeframe=timeframes[0],
                symbol_scope=symbol_scope,
                max_symbols_scan=max_symbols,
                seed=seed,
            )
        else:
            if dataset.timeframe not in timeframes:
                timeframes = [dataset.timeframe, *timeframes]
                run.timeframes_json = timeframes
                session.add(run)
                session.commit()
            symbols = store.sample_dataset_symbols(
                session=session,
                dataset_id=run.dataset_id,
                timeframe=timeframes[0],
                symbol_scope=symbol_scope,
                max_symbols_scan=max_symbols,
                seed=seed,
            )
    else:
        symbols = _sample_symbols_by_adv(
            session=session,
            store=store,
            primary_timeframe=primary_timeframe,
            max_symbols=max_symbols,
        )
    if not symbols:
        run.status = "FAILED"
        run.summary_json = {
            "error": {"code": "missing_data", "message": "No datasets available for Auto Research."}
        }
        session.add(run)
        session.commit()
        raise APIError(code="missing_data", message="No datasets available for Auto Research.")

    combos: list[tuple[str, str, str]] = [
        (symbol, timeframe, template)
        for symbol in symbols
        for timeframe in timeframes
        for template in templates
    ]
    if max_evaluations > 0:
        combos = combos[:max_evaluations]

    _emit(progress_cb, 15, f"Research scheduled: {len(combos)} evaluations")
    candidates: list[ResearchCandidate] = []
    failures: list[dict[str, str]] = []

    for idx, (symbol, timeframe, template) in enumerate(combos, start=1):
        progress = 15 + int((idx - 1) / max(1, len(combos)) * 75)
        _emit(
            progress_cb, progress, f"Evaluate {idx}/{len(combos)}: {symbol} {timeframe} {template}"
        )

        walkforward_payload = {
            "symbol": symbol,
            "timeframe": timeframe,
            "strategy_template": template,
            "config": {
                "trials": trials_per_strategy,
                "timeout_seconds": timeout_seconds,
                "sampler": sampler,
                "pruner": pruner,
                "seed": int(seed + idx),
                "max_oos_drawdown": constraints["max_drawdown"],
                "min_train_trades": max(3, constraints["min_trades"] // 4),
            },
        }

        try:
            summary = execute_walkforward(
                session=session,
                store=store,
                settings=settings,
                payload=walkforward_payload,
                progress_cb=None,
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "strategy_key": template,
                    "error": str(exc),
                }
            )
            continue

        oos_metrics, stress_metrics, param_dispersion, fold_variance = _flatten_candidate_metrics(
            summary
        )
        stress_pass_rate = float(summary.get("oos_only", {}).get("stress_pass_rate", 0.0))
        evaluation = evaluate_candidate_robustness(
            oos_metrics=oos_metrics,
            stress_metrics=stress_metrics,
            param_dispersion=param_dispersion,
            fold_variance=fold_variance,
            stress_pass_rate=stress_pass_rate,
            constraints=constraints,
        )

        candidate = ResearchCandidate(
            run_id=run.id,
            symbol=symbol,
            timeframe=timeframe,
            strategy_key=template,
            best_params_json=_best_params(summary),
            oos_metrics_json=oos_metrics,
            stress_metrics_json=stress_metrics,
            param_dispersion=param_dispersion,
            fold_variance=fold_variance,
            stress_pass_rate=stress_pass_rate,
            score=evaluation.score,
            rank=0,
            accepted=evaluation.accepted,
            explanations_json=evaluation.explanations,
        )
        session.add(candidate)
        session.flush()
        candidates.append(candidate)

    if not candidates:
        run.status = "FAILED"
        run.summary_json = {
            "run_id": run.id,
            "evaluations": len(combos),
            "candidate_count": 0,
            "failures": failures,
            "error": "No candidate completed successfully.",
        }
        session.add(run)
        session.commit()
        raise APIError(
            code="research_failed",
            message="Auto Research did not complete any successful candidate.",
        )

    candidates.sort(key=lambda row: (row.accepted, row.score), reverse=True)
    for rank, row in enumerate(candidates, start=1):
        row.rank = rank
        session.add(row)

    policy_preview = _policy_from_candidates(run.id, candidates)
    policy_preview["universe"] = {
        "bundle_id": run.bundle_id,
        "dataset_id": run.dataset_id,
        "symbol_scope": symbol_scope,
        "max_symbols_scan": max_symbols,
    }
    policy_preview["timeframes"] = timeframes
    policy_preview["ranking"] = {
        "method": "robust_score",
        "seed": int(seed),
        "weights": {"signal": 0.65, "liquidity": 0.25, "stability": 0.10},
    }
    policy_preview["cost_model"] = {
        "enabled": bool(settings.cost_model_enabled),
        "mode": settings.cost_mode,
    }
    accepted = [row for row in candidates if row.accepted]
    top = candidates[0]
    summary = {
        "run_id": run.id,
        "status": "SUCCEEDED",
        "evaluations": len(combos),
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(candidates) - len(accepted),
        "symbols": symbols,
        "bundle_id": run.bundle_id,
        "dataset_id": run.dataset_id,
        "timeframes": timeframes,
        "strategy_templates": templates,
        "top_candidate": {
            "id": top.id,
            "symbol": top.symbol,
            "timeframe": top.timeframe,
            "strategy_key": top.strategy_key,
            "score": top.score,
            "accepted": top.accepted,
            "rank": top.rank,
            "explanations": top.explanations_json,
        },
        "failures": failures,
        "policy_preview": policy_preview,
    }
    run.status = "SUCCEEDED"
    run.summary_json = summary
    session.add(run)
    session.commit()
    session.refresh(run)

    _emit(progress_cb, 96, "Auto Research summary compiled")
    return summary


def list_research_runs(
    session: Session, page: int, page_size: int
) -> tuple[list[ResearchRun], int]:
    rows = session.exec(select(ResearchRun).order_by(ResearchRun.created_at.desc())).all()
    total = len(rows)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return rows[start:end], total


def list_research_candidates(
    session: Session,
    *,
    run_id: int,
    page: int,
    page_size: int,
) -> tuple[list[ResearchCandidate], int]:
    rows = session.exec(
        select(ResearchCandidate)
        .where(ResearchCandidate.run_id == run_id)
        .order_by(ResearchCandidate.rank.asc())
    ).all()
    total = len(rows)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return rows[start:end], total


def create_policy_from_research_run(session: Session, *, run_id: int, name: str) -> Policy:
    run = session.get(ResearchRun, run_id)
    if run is None:
        raise APIError(code="not_found", message="Research run not found", status_code=404)

    candidates = session.exec(
        select(ResearchCandidate)
        .where(ResearchCandidate.run_id == run_id)
        .order_by(ResearchCandidate.rank.asc())
    ).all()
    if not candidates:
        raise APIError(
            code="invalid_state", message="Research run has no candidates to build a policy."
        )

    summary = run.summary_json if isinstance(run.summary_json, dict) else {}
    preview = summary.get("policy_preview")
    if isinstance(preview, dict) and preview.get("regime_map"):
        definition = dict(preview)
    else:
        definition = _policy_from_candidates(run_id, candidates)
    policy = Policy(
        name=name,
        definition_json=definition,
        promoted_from_research_run_id=run_id,
    )
    session.add(policy)
    session.commit()
    session.refresh(policy)
    return policy
