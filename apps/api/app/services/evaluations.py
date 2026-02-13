from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import Policy, PolicyEvaluation, PolicyShadowRun
from app.services.data_store import DataStore
from app.services.paper import get_or_create_paper_state
from app.services.policy_simulation import simulate_policy_on_bundle


ProgressCallback = Callable[[int, str], None]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: str | None, *, field_name: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise APIError(
            code="invalid_date",
            message=f"{field_name} must be YYYY-MM-DD.",
        ) from exc


def _evaluation_thresholds(state_settings: dict[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "auto_promote_enabled": bool(
            state_settings.get(
                "evaluations_auto_promote_enabled",
                settings.evaluations_auto_promote_enabled,
            )
        ),
        "min_window_days": max(
            5,
            _safe_int(
                state_settings.get(
                    "evaluations_min_window_days",
                    settings.evaluations_min_window_days,
                ),
                settings.evaluations_min_window_days,
            ),
        ),
        "score_margin": float(
            state_settings.get(
                "evaluations_score_margin",
                settings.evaluations_score_margin,
            )
        ),
        "max_dd_multiplier": float(
            state_settings.get(
                "evaluations_max_dd_multiplier",
                settings.evaluations_max_dd_multiplier,
            )
        ),
    }


def _resolve_window(
    payload: dict[str, Any],
    *,
    thresholds: dict[str, Any],
) -> tuple[date, date, int]:
    start_date = _parse_date(payload.get("start_date"), field_name="start_date")
    end_date = _parse_date(payload.get("end_date"), field_name="end_date")
    min_window = int(thresholds["min_window_days"])
    if end_date is None:
        end_date = datetime.now(timezone.utc).date()

    if start_date is None:
        requested_window_days = _safe_int(payload.get("window_days"), min_window)
        window_days = max(min_window, requested_window_days)
        start_date = end_date - timedelta(days=window_days - 1)
    else:
        window_days = max(1, (end_date - start_date).days + 1)
        if window_days < min_window:
            raise APIError(
                code="invalid_window",
                message=f"Evaluation window must be at least {min_window} days.",
            )
    if start_date > end_date:
        raise APIError(code="invalid_window", message="start_date must be <= end_date.")
    return start_date, end_date, window_days


def _recommendation(
    *,
    champion: dict[str, Any],
    challengers: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> tuple[int | None, str, list[str]]:
    reasons: list[str] = []
    champion_metrics = champion.get("metrics", {})
    champion_score = float(champion_metrics.get("score", -1.0))
    champion_dd = abs(float(champion_metrics.get("max_drawdown", 0.0)))
    score_margin = float(thresholds["score_margin"])
    max_dd_multiplier = float(thresholds["max_dd_multiplier"])

    accepted: list[dict[str, Any]] = []
    for row in challengers:
        metrics = row.get("metrics", {})
        score = float(metrics.get("score", -1.0))
        drawdown = abs(float(metrics.get("max_drawdown", 0.0)))
        policy_name = str(row.get("policy_name", row.get("policy_id")))
        if champion_dd > 0 and drawdown > champion_dd * max_dd_multiplier:
            reasons.append(
                f"Challenger rejected ({policy_name}): higher drawdown than threshold."
            )
            continue
        if score < champion_score + score_margin:
            reasons.append(
                f"Challenger rejected ({policy_name}): score improvement below margin."
            )
            continue
        reasons.append(
            f"Challenger accepted ({policy_name}): higher score with acceptable drawdown."
        )
        accepted.append(row)

    if not accepted:
        return None, "KEEP_CHAMPION", reasons

    accepted.sort(key=lambda item: float(item.get("metrics", {}).get("score", -1.0)), reverse=True)
    winner = accepted[0]
    winner_id = int(winner["policy_id"])
    return winner_id, "PROMOTE_CHALLENGER", reasons


def execute_policy_evaluation(
    *,
    session: Session,
    store: DataStore,
    settings: Settings,
    payload: dict[str, Any],
    progress_cb: ProgressCallback | None = None,
) -> dict[str, Any]:
    bundle_id = _safe_int(payload.get("bundle_id"), 0)
    if bundle_id <= 0:
        raise APIError(code="invalid_payload", message="bundle_id is required.")

    champion_policy_id = _safe_int(payload.get("champion_policy_id"), 0)
    champion = session.get(Policy, champion_policy_id)
    if champion is None:
        raise APIError(code="not_found", message="Champion policy not found.", status_code=404)

    state = get_or_create_paper_state(session, settings)
    state_settings = state.settings_json or {}
    thresholds = _evaluation_thresholds(state_settings, settings)
    start_date, end_date, window_days = _resolve_window(payload, thresholds=thresholds)
    regime = str(payload.get("regime")) if payload.get("regime") else None
    seed = _safe_int(payload.get("seed"), 7)

    raw_challengers = payload.get("challenger_policy_ids")
    challenger_ids: list[int] = []
    if isinstance(raw_challengers, list) and raw_challengers:
        challenger_ids = [_safe_int(item, 0) for item in raw_challengers if _safe_int(item, 0) > 0]
    else:
        challenger_ids = [
            int(row.id)
            for row in session.exec(select(Policy).order_by(Policy.created_at.desc())).all()
            if row.id is not None and int(row.id) != champion_policy_id
        ]

    evaluation = PolicyEvaluation(
        bundle_id=bundle_id,
        regime=regime,
        window_start=start_date,
        window_end=end_date,
        champion_policy_id=champion_policy_id,
        challenger_policy_ids_json=challenger_ids,
        status="RUNNING",
        summary_json={},
    )
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)

    if progress_cb:
        progress_cb(10, "Champion evaluation started")

    champion_summary = simulate_policy_on_bundle(
        session=session,
        store=store,
        settings=settings,
        policy=champion,
        bundle_id=bundle_id,
        start_date=start_date,
        end_date=end_date,
        regime=regime,
        seed=seed,
    )
    session.add(
        PolicyShadowRun(
            evaluation_id=int(evaluation.id),
            policy_id=int(champion.id),
            asof_date=end_date,
            run_summary_json=champion_summary,
        )
    )
    session.commit()

    challengers: list[dict[str, Any]] = []
    total = max(1, len(challenger_ids))
    for idx, challenger_id in enumerate(challenger_ids, start=1):
        challenger = session.get(Policy, challenger_id)
        if challenger is None:
            continue
        summary = simulate_policy_on_bundle(
            session=session,
            store=store,
            settings=settings,
            policy=challenger,
            bundle_id=bundle_id,
            start_date=start_date,
            end_date=end_date,
            regime=regime,
            seed=seed,
        )
        challengers.append(summary)
        session.add(
            PolicyShadowRun(
                evaluation_id=int(evaluation.id),
                policy_id=int(challenger.id),
                asof_date=end_date,
                run_summary_json=summary,
            )
        )
        session.commit()
        if progress_cb:
            progress_cb(10 + int((idx / total) * 70), f"Challenger {idx}/{total} evaluated")

    recommended_id, decision, reasons = _recommendation(
        champion=champion_summary,
        challengers=challengers,
        thresholds=thresholds,
    )
    auto_promoted = False
    if (
        bool(thresholds["auto_promote_enabled"])
        and recommended_id is not None
        and recommended_id != champion_policy_id
    ):
        selected = session.get(Policy, recommended_id)
        if selected is not None:
            merged = dict(state.settings_json or {})
            merged["paper_mode"] = "policy"
            merged["active_policy_id"] = int(selected.id)
            merged["active_policy_name"] = selected.name
            state.settings_json = merged
            session.add(state)
            session.commit()
            auto_promoted = True

    summary_json = {
        "window": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "window_days": window_days,
        },
        "bundle_id": bundle_id,
        "regime": regime,
        "seed": seed,
        "engine_version": champion_summary.get("engine_version"),
        "data_digest": champion_summary.get("data_digest"),
        "thresholds": thresholds,
        "champion": champion_summary,
        "challengers": challengers,
        "decision": {
            "recommendation": decision,
            "recommended_policy_id": recommended_id,
            "auto_promoted": auto_promoted,
            "reasons": reasons,
        },
        "shadow_mode": True,
        "note": "Shadow evaluation uses deterministic simulation and does not mutate live paper state.",
    }

    evaluation.status = "SUCCEEDED"
    evaluation.summary_json = summary_json
    evaluation.notes = "; ".join(reasons[:6]) if reasons else None
    session.add(evaluation)
    session.commit()
    session.refresh(evaluation)

    if progress_cb:
        progress_cb(100, "Evaluation finished")

    return {
        "evaluation_id": int(evaluation.id),
        "status": evaluation.status,
        "summary": summary_json,
    }


def list_policy_evaluations(
    session: Session,
    *,
    page: int,
    page_size: int,
) -> tuple[list[PolicyEvaluation], int]:
    rows = session.exec(select(PolicyEvaluation).order_by(PolicyEvaluation.created_at.desc())).all()
    total = len(rows)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    return rows[start:end], total


def get_policy_evaluation(session: Session, evaluation_id: int) -> PolicyEvaluation:
    row = session.get(PolicyEvaluation, evaluation_id)
    if row is None:
        raise APIError(code="not_found", message="Evaluation not found", status_code=404)
    return row


def get_policy_evaluation_details(
    session: Session,
    *,
    evaluation_id: int,
) -> list[PolicyShadowRun]:
    get_policy_evaluation(session, evaluation_id)
    return list(
        session.exec(
            select(PolicyShadowRun)
            .where(PolicyShadowRun.evaluation_id == evaluation_id)
            .order_by(PolicyShadowRun.policy_id.asc())
        ).all()
    )
