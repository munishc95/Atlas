from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
import hashlib
import json
from typing import Any

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import DailyConfidenceAggregate, DataProvenance
from app.services.confidence_gate import (
    DECISION_PASS,
    evaluate_confidence_gate,
    resolve_confidence_risk_scaling,
)
from app.services.trading_calendar import is_trading_day, previous_trading_day


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


def _normalize_date(value: dt_date | datetime | str | None) -> dt_date | None:
    if value is None:
        return None
    if isinstance(value, dt_date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str) and value.strip():
        try:
            return dt_date.fromisoformat(value.strip())
        except ValueError:
            return None
    return None


def _provider_mix_from_counts(counts: dict[str, int]) -> dict[str, float]:
    total = max(1, int(sum(max(0, int(value)) for value in counts.values())))
    return {
        str(provider): float(max(0, int(value)) / total)
        for provider, value in sorted(counts.items())
    }


def _provider_mix_shift_score(
    current_counts: dict[str, int],
    previous_counts: dict[str, int],
) -> float:
    current_mix = _provider_mix_from_counts(current_counts)
    previous_mix = _provider_mix_from_counts(previous_counts)
    providers = sorted(set(current_mix.keys()) | set(previous_mix.keys()))
    return float(
        sum(
            abs(float(current_mix.get(provider, 0.0)) - float(previous_mix.get(provider, 0.0)))
            for provider in providers
        )
    )


def _effective_trading_date(
    *,
    settings: Settings,
    overrides: dict[str, Any] | None,
    asof: dt_date,
) -> dt_date:
    segment = str((overrides or {}).get("trading_calendar_segment", settings.trading_calendar_segment))
    if is_trading_day(asof, segment=segment, settings=settings):
        return asof
    return previous_trading_day(asof, segment=segment, settings=settings)


def _settings_thresholds(
    *,
    settings: Settings,
    overrides: dict[str, Any] | None,
    operate_mode: str,
) -> dict[str, Any]:
    state = dict(overrides or {})
    scaling_enabled_raw = state.get("confidence_risk_scaling_enabled")
    if isinstance(scaling_enabled_raw, bool):
        scaling_enabled = scaling_enabled_raw
    elif isinstance(scaling_enabled_raw, str):
        scaling_enabled = scaling_enabled_raw.strip().lower() in {"true", "1", "yes", "on"}
    else:
        scaling_enabled = str(operate_mode).strip().lower() == "live"
    thresholds = {
        "confidence_gate_enabled": bool(
            state.get("confidence_gate_enabled", settings.confidence_gate_enabled)
        ),
        "confidence_gate_avg_threshold": _safe_float(
            state.get("confidence_gate_avg_threshold"),
            settings.confidence_gate_avg_threshold,
        ),
        "confidence_gate_low_symbol_threshold": _safe_float(
            state.get("confidence_gate_low_symbol_threshold"),
            settings.confidence_gate_low_symbol_threshold,
        ),
        "confidence_gate_low_pct_threshold": _safe_float(
            state.get("confidence_gate_low_pct_threshold"),
            settings.confidence_gate_low_pct_threshold,
        ),
        "confidence_gate_fallback_pct_threshold": _safe_float(
            state.get("confidence_gate_fallback_pct_threshold"),
            settings.confidence_gate_fallback_pct_threshold,
        ),
        "confidence_gate_hard_floor": _safe_float(
            state.get("confidence_gate_hard_floor"),
            settings.confidence_gate_hard_floor,
        ),
        "confidence_drop_warn_threshold": _safe_float(
            state.get("confidence_drop_warn_threshold"),
            settings.confidence_drop_warn_threshold,
        ),
        "confidence_provider_mix_shift_warn_pct": _safe_float(
            state.get("confidence_provider_mix_shift_warn_pct"),
            settings.confidence_provider_mix_shift_warn_pct,
        ),
        "confidence_gate_action_on_trigger": str(
            state.get("confidence_gate_action_on_trigger", settings.confidence_gate_action_on_trigger)
        )
        .strip()
        .upper(),
        "confidence_gate_lookback_days": max(
            1,
            _safe_int(
                state.get("confidence_gate_lookback_days"),
                settings.confidence_gate_lookback_days,
            ),
        ),
        "confidence_risk_scaling_enabled": bool(scaling_enabled),
        "confidence_risk_scale_exponent": max(
            0.1,
            _safe_float(
                state.get("confidence_risk_scale_exponent"),
                settings.confidence_risk_scale_exponent,
            ),
        ),
        "confidence_risk_scale_low_threshold": max(
            0.0,
            min(
                1.0,
                _safe_float(
                    state.get("confidence_risk_scale_low_threshold"),
                    settings.confidence_risk_scale_low_threshold,
                ),
            ),
        ),
    }
    thresholds["signature"] = hashlib.sha256(
        json.dumps(thresholds, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return thresholds


def serialize_daily_confidence_agg(row: DailyConfidenceAggregate) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "created_at": row.created_at.isoformat() if row.created_at is not None else None,
        "bundle_id": int(row.bundle_id),
        "timeframe": str(row.timeframe),
        "trading_date": row.trading_date.isoformat(),
        "eligible_symbols_count": int(row.eligible_symbols_count),
        "avg_confidence": float(row.avg_confidence),
        "pct_low_confidence": float(row.pct_low_confidence),
        "provider_counts": dict(row.provider_mix_json or {}),
        "low_confidence_symbols_count": int(row.low_confidence_symbols_count),
        "low_confidence_days_count": int(row.low_confidence_days_count),
        "drop_points": float(row.drop_points),
        "mix_shift_score": float(row.mix_shift_score),
        "flags": list(row.flags_json or []),
        "decision": str(row.gate_decision),
        "reasons": list(row.gate_reasons_json or []),
        "confidence_risk_scale": float(row.confidence_risk_scale),
        "threshold_used": dict(row.thresholds_json or {}),
    }


def serialize_agg_as_gate(row: DailyConfidenceAggregate) -> dict[str, Any]:
    payload = serialize_daily_confidence_agg(row)
    return {
        "id": payload["id"],
        "created_at": payload["created_at"],
        "bundle_id": payload["bundle_id"],
        "timeframe": payload["timeframe"],
        "trading_date": payload["trading_date"],
        "decision": payload["decision"],
        "reasons": payload["reasons"],
        "avg_confidence": payload["avg_confidence"],
        "pct_low_confidence": payload["pct_low_confidence"],
        "provider_mix": payload["provider_counts"],
        "threshold_used": payload["threshold_used"],
        "confidence_risk_scale": payload["confidence_risk_scale"],
        "drop_points": payload["drop_points"],
        "mix_shift_score": payload["mix_shift_score"],
        "flags": payload["flags"],
    }


def latest_daily_confidence_agg(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
) -> DailyConfidenceAggregate | None:
    stmt = select(DailyConfidenceAggregate)
    if isinstance(bundle_id, int) and bundle_id > 0:
        stmt = stmt.where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
    if isinstance(timeframe, str) and timeframe.strip():
        stmt = stmt.where(DailyConfidenceAggregate.timeframe == str(timeframe).strip())
    stmt = stmt.order_by(
        DailyConfidenceAggregate.trading_date.desc(),
        DailyConfidenceAggregate.created_at.desc(),
        DailyConfidenceAggregate.id.desc(),
    ).limit(1)
    return session.exec(stmt).first()


def list_daily_confidence_aggs(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str | None,
    limit: int = 60,
) -> list[DailyConfidenceAggregate]:
    stmt = select(DailyConfidenceAggregate)
    if isinstance(bundle_id, int) and bundle_id > 0:
        stmt = stmt.where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
    if isinstance(timeframe, str) and timeframe.strip():
        stmt = stmt.where(DailyConfidenceAggregate.timeframe == str(timeframe).strip())
    stmt = stmt.order_by(
        DailyConfidenceAggregate.trading_date.desc(),
        DailyConfidenceAggregate.created_at.desc(),
        DailyConfidenceAggregate.id.desc(),
    ).limit(max(1, min(int(limit), 366)))
    return list(session.exec(stmt).all())


def compute_daily_confidence_agg(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int,
    timeframe: str,
    trading_date: dt_date,
    operate_mode: str,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tf = str(timeframe or "1d").strip() or "1d"
    mode = str(operate_mode or settings.operate_mode).strip().lower()
    thresholds = _settings_thresholds(settings=settings, overrides=overrides, operate_mode=mode)
    day = _effective_trading_date(
        settings=settings,
        overrides=overrides,
        asof=trading_date,
    )

    rows = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == tf)
            .where(DataProvenance.bar_date == day)
        ).all()
    )
    eligible_symbols_count = int(len(rows))
    provider_counts: dict[str, int] = {}
    avg_confidence = 0.0
    low_confidence_symbols_count = 0
    low_symbol_threshold = float(thresholds["confidence_gate_low_symbol_threshold"])

    if eligible_symbols_count > 0:
        total = 0.0
        for row in rows:
            provider = str(row.source_provider or "INBOX").strip().upper() or "INBOX"
            provider_counts[provider] = int(provider_counts.get(provider, 0)) + 1
            score = _safe_float(row.confidence_score, 0.0)
            total += score
            if score < low_symbol_threshold:
                low_confidence_symbols_count += 1
        avg_confidence = float(total / max(1, eligible_symbols_count))
    pct_low_confidence = (
        float(low_confidence_symbols_count / max(1, eligible_symbols_count))
        if eligible_symbols_count > 0
        else 1.0
    )

    lookback_days = int(thresholds["confidence_gate_lookback_days"])
    lookback_start = day
    for _ in range(max(0, lookback_days - 1)):
        lookback_start = previous_trading_day(
            lookback_start,
            segment=str((overrides or {}).get("trading_calendar_segment", settings.trading_calendar_segment)),
            settings=settings,
        )
    lookback_rows = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == tf)
            .where(DataProvenance.bar_date >= lookback_start)
            .where(DataProvenance.bar_date <= day)
        ).all()
    )
    low_days: set[dt_date] = set()
    for row in lookback_rows:
        if _safe_float(row.confidence_score, 0.0) < low_symbol_threshold:
            low_days.add(row.bar_date)

    gate = evaluate_confidence_gate(
        session,
        settings=settings,
        bundle_id=int(bundle_id),
        timeframe=tf,
        asof_ts=datetime.combine(day, datetime.min.time(), tzinfo=UTC),
        operate_mode=mode,
        overrides=overrides,
        correlation_id=None,
        persist=False,
    )
    gate_decision = str(gate.get("decision", DECISION_PASS)).upper()
    gate_reasons = [str(item) for item in gate.get("reasons", [])]
    scaling = resolve_confidence_risk_scaling(
        settings=settings,
        overrides=overrides,
        avg_confidence=float(avg_confidence),
        hard_floor=float(thresholds["confidence_gate_hard_floor"]),
        avg_threshold=float(thresholds["confidence_gate_avg_threshold"]),
    )
    confidence_risk_scale = float(scaling["confidence_risk_scale"])
    if gate_decision == "BLOCK_ENTRIES":
        confidence_risk_scale = 0.0

    previous_agg = session.exec(
        select(DailyConfidenceAggregate)
        .where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
        .where(DailyConfidenceAggregate.timeframe == tf)
        .where(DailyConfidenceAggregate.trading_date < day)
        .order_by(
            DailyConfidenceAggregate.trading_date.desc(),
            DailyConfidenceAggregate.created_at.desc(),
            DailyConfidenceAggregate.id.desc(),
        )
    ).first()
    previous_provider_counts = (
        {
            str(key): int(value)
            for key, value in dict(previous_agg.provider_mix_json or {}).items()
        }
        if previous_agg is not None
        else {}
    )
    drop_points = (
        float(avg_confidence) - float(previous_agg.avg_confidence)
        if previous_agg is not None
        else 0.0
    )
    mix_shift_score = _provider_mix_shift_score(provider_counts, previous_provider_counts)
    flags: list[str] = []
    if eligible_symbols_count <= 0:
        flags.append("NO_ELIGIBLE_SYMBOLS")
    if avg_confidence < low_symbol_threshold:
        flags.append("LOW_CONF")
    if drop_points <= -float(thresholds["confidence_drop_warn_threshold"]):
        flags.append("CONF_DROP")
    if mix_shift_score >= float(thresholds["confidence_provider_mix_shift_warn_pct"]):
        flags.append("MIX_SHIFT")
    if gate_decision != DECISION_PASS:
        flags.append("GATE_TRIGGERED")

    return {
        "bundle_id": int(bundle_id),
        "timeframe": tf,
        "trading_date": day,
        "eligible_symbols_count": int(eligible_symbols_count),
        "avg_confidence": float(avg_confidence),
        "pct_low_confidence": float(pct_low_confidence),
        "provider_counts": dict(provider_counts),
        "low_confidence_symbols_count": int(low_confidence_symbols_count),
        "low_confidence_days_count": int(len(low_days)),
        "drop_points": float(drop_points),
        "mix_shift_score": float(mix_shift_score),
        "flags": list(dict.fromkeys(flags)),
        "gate_decision": gate_decision,
        "gate_reasons": gate_reasons,
        "confidence_risk_scale": float(max(0.0, min(1.0, confidence_risk_scale))),
        "thresholds": dict(thresholds),
    }


def upsert_daily_confidence_agg(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int,
    timeframe: str,
    trading_date: dt_date | datetime | str | None,
    operate_mode: str,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[DailyConfidenceAggregate, bool]:
    day = _normalize_date(trading_date)
    if day is None:
        day = datetime.now(UTC).date()
    payload = compute_daily_confidence_agg(
        session,
        settings=settings,
        bundle_id=int(bundle_id),
        timeframe=str(timeframe),
        trading_date=day,
        operate_mode=str(operate_mode),
        overrides=overrides,
    )
    signature = str(payload["thresholds"].get("signature", ""))
    existing = session.exec(
        select(DailyConfidenceAggregate)
        .where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
        .where(DailyConfidenceAggregate.timeframe == str(payload["timeframe"]))
        .where(DailyConfidenceAggregate.trading_date == payload["trading_date"])
        .order_by(DailyConfidenceAggregate.id.desc())
    ).first()
    if (
        existing is not None
        and not force
        and str(existing.thresholds_signature or "") == signature
    ):
        return existing, False

    row = existing or DailyConfidenceAggregate(
        bundle_id=int(payload["bundle_id"]),
        timeframe=str(payload["timeframe"]),
        trading_date=payload["trading_date"],
    )
    row.created_at = datetime.now(UTC)
    row.eligible_symbols_count = int(payload["eligible_symbols_count"])
    row.avg_confidence = float(payload["avg_confidence"])
    row.pct_low_confidence = float(payload["pct_low_confidence"])
    row.provider_mix_json = dict(payload["provider_counts"])
    row.low_confidence_symbols_count = int(payload["low_confidence_symbols_count"])
    row.low_confidence_days_count = int(payload["low_confidence_days_count"])
    row.drop_points = float(payload["drop_points"])
    row.mix_shift_score = float(payload["mix_shift_score"])
    row.flags_json = list(payload["flags"])
    row.gate_decision = str(payload["gate_decision"])
    row.gate_reasons_json = list(payload["gate_reasons"])
    row.confidence_risk_scale = float(payload["confidence_risk_scale"])
    row.thresholds_json = dict(payload["thresholds"])
    row.thresholds_signature = signature
    session.add(row)
    session.flush()
    return row, True


def provider_status_trend_from_aggs(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    days: int = 30,
) -> list[dict[str, Any]]:
    rows = list_daily_confidence_aggs(
        session,
        bundle_id=int(bundle_id),
        timeframe=str(timeframe),
        limit=max(1, min(int(days), 365)),
    )
    output: list[dict[str, Any]] = []
    for row in reversed(rows):
        counts = {str(key): int(value) for key, value in dict(row.provider_mix_json or {}).items()}
        total = max(1, int(sum(counts.values())))
        provider_mix = {
            key: round(value / total, 6) for key, value in sorted(counts.items())
        }
        dominant = max(provider_mix.items(), key=lambda item: item[1])[0] if provider_mix else None
        output.append(
            {
                "trading_date": row.trading_date.isoformat(),
                "provider_counts": counts,
                "provider_mix": provider_mix,
                "dominant_provider": dominant,
                "avg_confidence": float(row.avg_confidence),
                "pct_low_confidence": float(row.pct_low_confidence),
                "symbols": int(row.eligible_symbols_count),
                "decision": str(row.gate_decision),
                "confidence_risk_scale": float(row.confidence_risk_scale),
            }
        )
    return output


def recompute_daily_confidence_aggs(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int,
    timeframe: str,
    operate_mode: str,
    overrides: dict[str, Any] | None = None,
    from_date: dt_date | datetime | str | None = None,
    to_date: dt_date | datetime | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    tf = str(timeframe or "1d").strip() or "1d"
    start_day = _normalize_date(from_date)
    end_day = _normalize_date(to_date)
    stmt = (
        select(DataProvenance.bar_date)
        .where(DataProvenance.bundle_id == int(bundle_id))
        .where(DataProvenance.timeframe == tf)
    )
    if start_day is not None:
        stmt = stmt.where(DataProvenance.bar_date >= start_day)
    if end_day is not None:
        stmt = stmt.where(DataProvenance.bar_date <= end_day)
    days = sorted({row for row in session.exec(stmt).all()})
    if not days:
        return {"processed": 0, "updated": 0, "skipped": 0, "days": []}
    if len(days) > 366:
        days = days[-366:]
    updated = 0
    skipped = 0
    for day in days:
        _, changed = upsert_daily_confidence_agg(
            session,
            settings=settings,
            bundle_id=int(bundle_id),
            timeframe=tf,
            trading_date=day,
            operate_mode=operate_mode,
            overrides=overrides,
            force=bool(force),
        )
        if changed:
            updated += 1
        else:
            skipped += 1
    session.commit()
    return {
        "processed": len(days),
        "updated": int(updated),
        "skipped": int(skipped),
        "days": [day.isoformat() for day in days],
    }
