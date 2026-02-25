from __future__ import annotations

from datetime import date as dt_date, datetime
from typing import Any

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import ConfidenceGateSnapshot, DataProvenance
from app.services.operate_events import emit_operate_event
from app.services.trading_calendar import is_trading_day, previous_trading_day


DECISION_PASS = "PASS"
DECISION_SHADOW_ONLY = "SHADOW_ONLY"
DECISION_BLOCK_ENTRIES = "BLOCK_ENTRIES"

PRIMARY_PROVIDER = "UPSTOX"
FALLBACK_PROVIDERS = {"NSE_EOD", "INBOX"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _settings_scope(settings: Settings, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    scope = dict(overrides or {})
    return {
        "enabled": bool(scope.get("confidence_gate_enabled", settings.confidence_gate_enabled)),
        "avg_threshold": _safe_float(
            scope.get("confidence_gate_avg_threshold"),
            settings.confidence_gate_avg_threshold,
        ),
        "low_symbol_threshold": _safe_float(
            scope.get("confidence_gate_low_symbol_threshold"),
            settings.confidence_gate_low_symbol_threshold,
        ),
        "low_pct_threshold": _safe_float(
            scope.get("confidence_gate_low_pct_threshold"),
            settings.confidence_gate_low_pct_threshold,
        ),
        "fallback_pct_threshold": _safe_float(
            scope.get("confidence_gate_fallback_pct_threshold"),
            settings.confidence_gate_fallback_pct_threshold,
        ),
        "hard_floor": _safe_float(
            scope.get("confidence_gate_hard_floor"),
            settings.confidence_gate_hard_floor,
        ),
        "action_on_trigger": str(
            scope.get("confidence_gate_action_on_trigger", settings.confidence_gate_action_on_trigger)
        )
        .strip()
        .upper(),
        "lookback_days": max(
            1,
            _safe_int(
                scope.get("confidence_gate_lookback_days"),
                settings.confidence_gate_lookback_days,
            ),
        ),
        "calendar_segment": str(
            scope.get("trading_calendar_segment", settings.trading_calendar_segment)
        ).strip()
        or "EQUITIES",
    }


def _scaling_scope(settings: Settings, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    scope = dict(overrides or {})
    operate_mode = str(scope.get("operate_mode", settings.operate_mode)).strip().lower()
    explicit_enabled = scope.get("confidence_risk_scaling_enabled")
    if isinstance(explicit_enabled, bool):
        enabled = explicit_enabled
    elif isinstance(explicit_enabled, str):
        enabled = explicit_enabled.strip().lower() in {"true", "1", "yes", "on"}
    else:
        enabled = operate_mode == "live"
    return {
        "enabled": bool(enabled),
        "exponent": max(
            0.1,
            _safe_float(
                scope.get("confidence_risk_scale_exponent"),
                settings.confidence_risk_scale_exponent,
            ),
        ),
        "low_threshold": max(
            0.0,
            min(
                1.0,
                _safe_float(
                    scope.get("confidence_risk_scale_low_threshold"),
                    settings.confidence_risk_scale_low_threshold,
                ),
            ),
        ),
    }


def compute_confidence_risk_scale(
    avg_confidence: float,
    *,
    hard_floor: float,
    avg_threshold: float,
    exponent: float = 1.0,
) -> float:
    floor_value = float(hard_floor)
    threshold_value = float(avg_threshold)
    avg_value = float(avg_confidence)
    if threshold_value <= floor_value:
        raw = 1.0 if avg_value >= threshold_value else 0.0
    elif avg_value <= floor_value:
        raw = 0.0
    elif avg_value >= threshold_value:
        raw = 1.0
    else:
        raw = (avg_value - floor_value) / max(1e-9, threshold_value - floor_value)
    shaped = float(max(0.0, min(1.0, raw))) ** max(0.1, float(exponent))
    return float(max(0.0, min(1.0, shaped)))


def resolve_confidence_risk_scaling(
    *,
    settings: Settings,
    overrides: dict[str, Any] | None,
    avg_confidence: float,
    hard_floor: float,
    avg_threshold: float,
) -> dict[str, Any]:
    scope = _scaling_scope(settings, overrides=overrides)
    if not bool(scope["enabled"]):
        return {
            "enabled": False,
            "confidence_risk_scale": 1.0,
            "exponent": float(scope["exponent"]),
            "low_threshold": float(scope["low_threshold"]),
        }
    scale = compute_confidence_risk_scale(
        float(avg_confidence),
        hard_floor=float(hard_floor),
        avg_threshold=float(avg_threshold),
        exponent=float(scope["exponent"]),
    )
    return {
        "enabled": True,
        "confidence_risk_scale": float(scale),
        "exponent": float(scope["exponent"]),
        "low_threshold": float(scope["low_threshold"]),
    }


def _resolve_effective_trading_date(
    *,
    asof_ts: datetime,
    segment: str,
    settings: Settings,
) -> dt_date:
    asof_date = asof_ts.date()
    if is_trading_day(asof_date, segment=segment, settings=settings):
        return asof_date
    return previous_trading_day(asof_date, segment=segment, settings=settings)


def _rows_for_day(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    trading_date: dt_date,
) -> list[DataProvenance]:
    return list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == str(timeframe))
            .where(DataProvenance.bar_date == trading_date)
        ).all()
    )


def _day_stats(
    rows: list[DataProvenance],
    *,
    low_symbol_threshold: float,
) -> dict[str, Any]:
    eligible_symbols = len(rows)
    if eligible_symbols <= 0:
        return {
            "eligible_symbols": 0,
            "avg_confidence": 0.0,
            "pct_low_confidence": 1.0,
            "provider_counts": {},
            "provider_mix": {},
            "fallback_pct": 1.0,
            "primary_pct": 0.0,
        }
    provider_counts: dict[str, int] = {}
    low_count = 0
    confidence_sum = 0.0
    for row in rows:
        confidence = _safe_float(row.confidence_score, 0.0)
        confidence_sum += confidence
        if confidence < float(low_symbol_threshold):
            low_count += 1
        provider = str(row.source_provider or "INBOX").strip().upper() or "INBOX"
        provider_counts[provider] = int(provider_counts.get(provider, 0)) + 1
    avg_confidence = confidence_sum / max(1, eligible_symbols)
    pct_low = low_count / max(1, eligible_symbols)
    provider_mix = {
        provider: (count / max(1, eligible_symbols))
        for provider, count in sorted(provider_counts.items())
    }
    fallback_count = sum(
        count for provider, count in provider_counts.items() if provider in FALLBACK_PROVIDERS
    )
    primary_count = int(provider_counts.get(PRIMARY_PROVIDER, 0))
    return {
        "eligible_symbols": int(eligible_symbols),
        "avg_confidence": float(avg_confidence),
        "pct_low_confidence": float(pct_low),
        "provider_counts": provider_counts,
        "provider_mix": provider_mix,
        "fallback_pct": float(fallback_count / max(1, eligible_symbols)),
        "primary_pct": float(primary_count / max(1, eligible_symbols)),
    }


def serialize_confidence_gate_snapshot(row: ConfidenceGateSnapshot) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "created_at": row.created_at.isoformat() if row.created_at is not None else None,
        "bundle_id": int(row.bundle_id) if row.bundle_id is not None else None,
        "timeframe": str(row.timeframe),
        "trading_date": row.trading_date.isoformat(),
        "decision": str(row.decision),
        "reasons": list(row.reasons_json or []),
        "avg_confidence": float(row.avg_confidence),
        "pct_low_confidence": float(row.pct_low_confidence),
        "provider_mix": dict(row.provider_mix_json or {}),
        "threshold_used": dict(row.threshold_json or {}),
    }


def latest_confidence_gate_snapshot(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
) -> ConfidenceGateSnapshot | None:
    stmt = select(ConfidenceGateSnapshot)
    if isinstance(bundle_id, int) and bundle_id > 0:
        stmt = stmt.where(ConfidenceGateSnapshot.bundle_id == int(bundle_id))
    if isinstance(timeframe, str) and timeframe.strip():
        stmt = stmt.where(ConfidenceGateSnapshot.timeframe == str(timeframe).strip())
    stmt = stmt.order_by(
        ConfidenceGateSnapshot.trading_date.desc(),
        ConfidenceGateSnapshot.created_at.desc(),
        ConfidenceGateSnapshot.id.desc(),
    ).limit(1)
    return session.exec(stmt).first()


def list_confidence_gate_snapshots(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
    limit: int = 60,
) -> list[ConfidenceGateSnapshot]:
    stmt = select(ConfidenceGateSnapshot)
    if isinstance(bundle_id, int) and bundle_id > 0:
        stmt = stmt.where(ConfidenceGateSnapshot.bundle_id == int(bundle_id))
    if isinstance(timeframe, str) and timeframe.strip():
        stmt = stmt.where(ConfidenceGateSnapshot.timeframe == str(timeframe).strip())
    stmt = stmt.order_by(
        ConfidenceGateSnapshot.trading_date.desc(),
        ConfidenceGateSnapshot.created_at.desc(),
        ConfidenceGateSnapshot.id.desc(),
    ).limit(max(1, min(int(limit), 365)))
    return list(session.exec(stmt).all())


def evaluate_confidence_gate(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int | None,
    timeframe: str,
    asof_ts: datetime,
    operate_mode: str,
    overrides: dict[str, Any] | None = None,
    correlation_id: str | None = None,
    persist: bool = True,
) -> dict[str, Any]:
    mode = str(operate_mode or "").strip().lower()
    scope = _settings_scope(settings, overrides)
    enabled = bool(scope["enabled"]) if mode == "live" else False

    if not isinstance(bundle_id, int) or bundle_id <= 0:
        summary = {
            "trading_date": asof_ts.date().isoformat(),
            "avg_confidence": 0.0,
            "pct_low_confidence": 1.0,
            "provider_mix": {},
            "latest_day_source_counts": {},
            "threshold_used": dict(scope),
            "days_lookback_used": 0,
            "eligible_symbols": 0,
        }
        return {
            "id": None,
            "enabled": enabled,
            "decision": DECISION_SHADOW_ONLY if enabled else DECISION_PASS,
            "reasons": (["no_bundle_scope"] if enabled else []),
            "summary": summary,
        }

    calendar_segment = str(scope["calendar_segment"])
    effective_day = _resolve_effective_trading_date(
        asof_ts=asof_ts,
        segment=calendar_segment,
        settings=settings,
    )
    lookback_days = int(scope["lookback_days"])
    days = [effective_day]
    for _ in range(max(0, lookback_days - 1)):
        days.append(previous_trading_day(days[-1], segment=calendar_segment, settings=settings))
    days = sorted(set(days))

    stats_per_day: list[dict[str, Any]] = []
    for day in days:
        stats_per_day.append(
            _day_stats(
                _rows_for_day(
                    session,
                    bundle_id=int(bundle_id),
                    timeframe=str(timeframe),
                    trading_date=day,
                ),
                low_symbol_threshold=float(scope["low_symbol_threshold"]),
            )
        )

    latest_stats = stats_per_day[-1] if stats_per_day else _day_stats([], low_symbol_threshold=65.0)
    eligible_symbols = int(latest_stats["eligible_symbols"])
    avg_confidence = float(
        sum(float(item["avg_confidence"]) for item in stats_per_day) / max(1, len(stats_per_day))
    )
    pct_low = float(
        sum(float(item["pct_low_confidence"]) for item in stats_per_day) / max(1, len(stats_per_day))
    )
    fallback_pct = float(
        sum(float(item["fallback_pct"]) for item in stats_per_day) / max(1, len(stats_per_day))
    )
    primary_pct = float(
        sum(float(item["primary_pct"]) for item in stats_per_day) / max(1, len(stats_per_day))
    )

    reasons: list[str] = []
    decision = DECISION_PASS
    trigger_action = (
        DECISION_BLOCK_ENTRIES
        if str(scope["action_on_trigger"]).upper() == DECISION_BLOCK_ENTRIES
        else DECISION_SHADOW_ONLY
    )

    if enabled:
        if eligible_symbols <= 0:
            reasons.append("no_eligible_symbols")
        if avg_confidence < float(scope["avg_threshold"]):
            reasons.append("avg_confidence_below_threshold")
        if pct_low > float(scope["low_pct_threshold"]):
            reasons.append("too_many_low_confidence_symbols")
        if fallback_pct > float(scope["fallback_pct_threshold"]):
            reasons.append("fallback_dominance")

        extreme = (
            avg_confidence < float(scope["hard_floor"])
            or (eligible_symbols > 0 and primary_pct <= 0.0)
        )
        if reasons:
            decision = DECISION_BLOCK_ENTRIES if (extreme and trigger_action == DECISION_BLOCK_ENTRIES) else trigger_action
    else:
        reasons = []
        decision = DECISION_PASS

    summary = {
        "trading_date": effective_day.isoformat(),
        "avg_confidence": float(avg_confidence),
        "pct_low_confidence": float(pct_low),
        "provider_mix": dict(latest_stats.get("provider_mix", {})),
        "latest_day_source_counts": dict(latest_stats.get("provider_counts", {})),
        "threshold_used": {
            "avg_threshold": float(scope["avg_threshold"]),
            "low_symbol_threshold": float(scope["low_symbol_threshold"]),
            "low_pct_threshold": float(scope["low_pct_threshold"]),
            "fallback_pct_threshold": float(scope["fallback_pct_threshold"]),
            "hard_floor": float(scope["hard_floor"]),
            "action_on_trigger": trigger_action,
            "enabled": bool(enabled),
        },
        "days_lookback_used": len(stats_per_day),
        "eligible_symbols": eligible_symbols,
    }
    scaling = resolve_confidence_risk_scaling(
        settings=settings,
        overrides=overrides,
        avg_confidence=float(avg_confidence),
        hard_floor=float(scope["hard_floor"]),
        avg_threshold=float(scope["avg_threshold"]),
    )
    summary["confidence_risk_scale"] = float(scaling["confidence_risk_scale"])
    summary["confidence_risk_scaling_enabled"] = bool(scaling["enabled"])
    summary["confidence_risk_scale_exponent"] = float(scaling["exponent"])
    summary["confidence_risk_scale_low_threshold"] = float(scaling["low_threshold"])

    snapshot_id: int | None = None
    if persist:
        row = ConfidenceGateSnapshot(
            bundle_id=int(bundle_id),
            timeframe=str(timeframe),
            trading_date=effective_day,
            decision=decision,
            reasons_json=list(dict.fromkeys(reasons)),
            avg_confidence=float(avg_confidence),
            pct_low_confidence=float(pct_low),
            provider_mix_json=dict(latest_stats.get("provider_mix", {})),
            threshold_json=dict(summary["threshold_used"]),
        )
        session.add(row)
        session.commit()
        session.refresh(row)
        snapshot_id = int(row.id) if row.id is not None else None

    if decision != DECISION_PASS and enabled and persist:
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="confidence_gate_triggered",
            details={
                "snapshot_id": snapshot_id,
                "bundle_id": int(bundle_id),
                "timeframe": str(timeframe),
                "decision": decision,
                "reasons": list(dict.fromkeys(reasons)),
                "summary": summary,
            },
            correlation_id=correlation_id,
            commit=False,
        )

    return {
        "id": snapshot_id,
        "enabled": bool(enabled),
        "decision": decision,
        "reasons": list(dict.fromkeys(reasons)),
        "summary": summary,
    }
