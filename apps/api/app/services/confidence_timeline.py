from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from typing import Any

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import DailyConfidenceAggregate, DataProvenance
from app.services.data_store import DataStore
from app.services.fast_mode import fast_mode_enabled
from app.services.trading_calendar import previous_trading_day


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


def _provider_mix(counts: dict[str, int]) -> dict[str, float]:
    total = max(1, int(sum(max(0, int(value)) for value in counts.values())))
    return {
        str(provider): float(max(0, int(value)) / total)
        for provider, value in sorted(counts.items())
    }


def _aggregate_for_day(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    trading_date: dt_date,
) -> DailyConfidenceAggregate | None:
    return session.exec(
        select(DailyConfidenceAggregate)
        .where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
        .where(DailyConfidenceAggregate.timeframe == str(timeframe))
        .where(DailyConfidenceAggregate.trading_date == trading_date)
        .order_by(DailyConfidenceAggregate.id.desc())
    ).first()


def _previous_aggregate(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    trading_date: dt_date,
) -> DailyConfidenceAggregate | None:
    return session.exec(
        select(DailyConfidenceAggregate)
        .where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
        .where(DailyConfidenceAggregate.timeframe == str(timeframe))
        .where(DailyConfidenceAggregate.trading_date < trading_date)
        .order_by(
            DailyConfidenceAggregate.trading_date.desc(),
            DailyConfidenceAggregate.created_at.desc(),
            DailyConfidenceAggregate.id.desc(),
        )
    ).first()


def _timeline_row(
    row: DailyConfidenceAggregate,
    *,
    confidence_drop_vs_prev: float,
) -> dict[str, Any]:
    provider_counts = {
        str(provider): int(value)
        for provider, value in dict(row.provider_mix_json or {}).items()
    }
    return {
        "id": int(row.id) if row.id is not None else None,
        "trading_date": row.trading_date.isoformat(),
        "avg_confidence": float(row.avg_confidence),
        "confidence_risk_scale": float(row.confidence_risk_scale),
        "gate_decision": str(row.gate_decision),
        "provider_mix": provider_counts,
        "low_confidence_symbols_count": int(row.low_confidence_symbols_count),
        "confidence_drop_vs_prev": float(confidence_drop_vs_prev),
        "drop_points": float(row.drop_points),
        "mix_shift_score": float(row.mix_shift_score),
        "flags": [str(item) for item in (row.flags_json or [])],
    }


def _fast_mode_timeline_stub(
    *,
    settings: Settings,
    bundle_id: int,
    timeframe: str,
    limit: int,
) -> list[dict[str, Any]]:
    cap = max(1, min(int(limit), 60))
    rows: list[dict[str, Any]] = []
    day = datetime.now(UTC).date()
    for _ in range(cap):
        day = previous_trading_day(day, segment=settings.trading_calendar_segment, settings=settings)
        idx = len(rows)
        avg = max(45.0, 74.0 - (idx * 2.2))
        decision = "PASS" if avg >= 70 else "SHADOW_ONLY"
        scale = max(0.0, min(1.0, (avg - 55.0) / 15.0))
        provider = "UPSTOX" if idx % 3 == 0 else ("NSE_EOD" if idx % 3 == 1 else "INBOX")
        provider_mix = {provider: 10}
        flags: list[str] = []
        if avg < 65:
            flags.append("LOW_CONF")
        if idx > 0 and rows[-1]["avg_confidence"] - avg >= 8.0:
            flags.append("CONF_DROP")
        if provider != "UPSTOX":
            flags.append("MIX_SHIFT")
        rows.append(
            {
                "id": None,
                "trading_date": day.isoformat(),
                "avg_confidence": float(avg),
                "confidence_risk_scale": float(scale),
                "gate_decision": decision,
                "provider_mix": provider_mix,
                "low_confidence_symbols_count": int(3 + idx),
                "confidence_drop_vs_prev": float(0.0 if idx == 0 else avg - float(rows[-1]["avg_confidence"])),
                "drop_points": float(0.0 if idx == 0 else avg - float(rows[-1]["avg_confidence"])),
                "mix_shift_score": 1.0 if provider != "UPSTOX" else 0.0,
                "flags": flags,
            }
        )
    return list(reversed(rows))


def confidence_timeline(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int,
    timeframe: str,
    limit: int = 60,
) -> list[dict[str, Any]]:
    rows = list(
        session.exec(
            select(DailyConfidenceAggregate)
            .where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
            .where(DailyConfidenceAggregate.timeframe == str(timeframe))
            .order_by(
                DailyConfidenceAggregate.trading_date.desc(),
                DailyConfidenceAggregate.created_at.desc(),
                DailyConfidenceAggregate.id.desc(),
            )
            .limit(max(1, min(int(limit), 366)))
        ).all()
    )
    if not rows:
        if fast_mode_enabled(settings):
            return _fast_mode_timeline_stub(
                settings=settings,
                bundle_id=bundle_id,
                timeframe=timeframe,
                limit=limit,
            )
        return []

    latest_by_day: dict[dt_date, DailyConfidenceAggregate] = {}
    for row in rows:
        latest_by_day[row.trading_date] = row
    ordered = [latest_by_day[key] for key in sorted(latest_by_day.keys())]
    out: list[dict[str, Any]] = []
    previous_avg: float | None = None
    for row in ordered:
        current_avg = float(row.avg_confidence)
        drop = 0.0 if previous_avg is None else float(current_avg - previous_avg)
        out.append(_timeline_row(row, confidence_drop_vs_prev=drop))
        previous_avg = current_avg
    return out[-max(1, min(int(limit), len(out))) :]


def confidence_drilldown(
    session: Session,
    *,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    trading_date: dt_date,
) -> dict[str, Any]:
    agg_row = _aggregate_for_day(
        session,
        bundle_id=int(bundle_id),
        timeframe=str(timeframe),
        trading_date=trading_date,
    )
    if agg_row is None and fast_mode_enabled(settings):
        return {
            "summary": {
                "trading_date": trading_date.isoformat(),
                "avg_confidence": 62.0,
                "confidence_risk_scale": 0.47,
                "gate_decision": "SHADOW_ONLY",
                "flags": ["LOW_CONF", "MIX_SHIFT"],
                "confidence_drop_vs_prev": -9.2,
            },
            "worst_symbols_by_confidence": [
                {
                    "symbol": "NIFTY500",
                    "confidence": 58.0,
                    "provider": "NSE_EOD",
                    "bars_present": True,
                },
                {
                    "symbol": "RELIANCE",
                    "confidence": 55.0,
                    "provider": "INBOX",
                    "bars_present": True,
                },
            ],
            "missing_symbols": ["INFY", "TCS"],
            "provider_mix_today": {"NSE_EOD": 7, "INBOX": 3},
            "provider_mix_prev": {"UPSTOX": 10},
            "provider_mix_delta": {"UPSTOX": -1.0, "NSE_EOD": 0.7, "INBOX": 0.3},
            "low_confidence_threshold": float(settings.confidence_gate_low_symbol_threshold),
            "notes": ["fast_mode_stub"],
        }
    if agg_row is None:
        return {
            "summary": None,
            "worst_symbols_by_confidence": [],
            "missing_symbols": [],
            "provider_mix_today": {},
            "provider_mix_prev": {},
            "provider_mix_delta": {},
            "low_confidence_threshold": float(settings.confidence_gate_low_symbol_threshold),
            "notes": ["aggregate_not_found"],
        }

    prev_agg = _previous_aggregate(
        session,
        bundle_id=int(bundle_id),
        timeframe=str(timeframe),
        trading_date=trading_date,
    )
    rows_today = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == str(timeframe))
            .where(DataProvenance.bar_date == trading_date)
        ).all()
    )
    worst_rows = sorted(
        rows_today,
        key=lambda row: (_safe_float(row.confidence_score, 0.0), str(row.symbol)),
    )[:20]
    worst_symbols = [
        {
            "symbol": row.symbol,
            "confidence": float(_safe_float(row.confidence_score, 0.0)),
            "provider": str(row.source_provider or "INBOX"),
            "bars_present": True,
        }
        for row in worst_rows
    ]
    expected_symbols = store.get_bundle_symbols(session, int(bundle_id), timeframe=str(timeframe))
    row_symbols = {str(row.symbol).upper() for row in rows_today}
    missing_symbols = [symbol for symbol in expected_symbols if str(symbol).upper() not in row_symbols][:200]

    today_counts = {
        str(key): _safe_int(value, 0)
        for key, value in dict(agg_row.provider_mix_json or {}).items()
    }
    prev_counts = (
        {
            str(key): _safe_int(value, 0)
            for key, value in dict(prev_agg.provider_mix_json or {}).items()
        }
        if prev_agg is not None
        else {}
    )
    today_mix = _provider_mix(today_counts)
    prev_mix = _provider_mix(prev_counts)
    provider_keys = sorted(set(today_mix.keys()) | set(prev_mix.keys()))
    mix_delta = {
        provider: float(today_mix.get(provider, 0.0) - prev_mix.get(provider, 0.0))
        for provider in provider_keys
    }
    low_threshold = float(
        (agg_row.thresholds_json or {}).get(
            "confidence_gate_low_symbol_threshold",
            settings.confidence_gate_low_symbol_threshold,
        )
    )
    return {
        "summary": {
            "id": int(agg_row.id) if agg_row.id is not None else None,
            "trading_date": agg_row.trading_date.isoformat(),
            "avg_confidence": float(agg_row.avg_confidence),
            "pct_low_confidence": float(agg_row.pct_low_confidence),
            "confidence_risk_scale": float(agg_row.confidence_risk_scale),
            "gate_decision": str(agg_row.gate_decision),
            "gate_reasons": [str(item) for item in (agg_row.gate_reasons_json or [])],
            "confidence_drop_vs_prev": float(
                float(agg_row.avg_confidence) - float(prev_agg.avg_confidence)
                if prev_agg is not None
                else 0.0
            ),
            "drop_points": float(agg_row.drop_points),
            "mix_shift_score": float(agg_row.mix_shift_score),
            "flags": [str(item) for item in (agg_row.flags_json or [])],
        },
        "worst_symbols_by_confidence": worst_symbols,
        "missing_symbols": missing_symbols,
        "provider_mix_today": today_counts,
        "provider_mix_prev": prev_counts,
        "provider_mix_delta": mix_delta,
        "low_confidence_threshold": low_threshold,
        "notes": [],
    }


def confidence_drilldown_symbols(
    session: Session,
    *,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    trading_date: dt_date,
    only: str = "all",
    limit: int = 200,
) -> dict[str, Any]:
    cap = max(1, min(int(limit), 500))
    mode = str(only or "all").strip().lower()
    agg_row = _aggregate_for_day(
        session,
        bundle_id=int(bundle_id),
        timeframe=str(timeframe),
        trading_date=trading_date,
    )
    low_threshold = float(
        (agg_row.thresholds_json or {}).get(
            "confidence_gate_low_symbol_threshold",
            settings.confidence_gate_low_symbol_threshold,
        )
        if agg_row is not None
        else settings.confidence_gate_low_symbol_threshold
    )
    rows_today = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == str(timeframe))
            .where(DataProvenance.bar_date == trading_date)
        ).all()
    )
    by_symbol = {str(row.symbol).upper(): row for row in rows_today}
    expected_symbols = store.get_bundle_symbols(session, int(bundle_id), timeframe=str(timeframe))

    output: list[dict[str, Any]] = []
    if mode in {"low", "all"}:
        low_rows = [
            row
            for row in rows_today
            if _safe_float(row.confidence_score, 0.0) < low_threshold
        ]
        low_rows.sort(key=lambda row: (_safe_float(row.confidence_score, 0.0), str(row.symbol)))
        for row in low_rows:
            output.append(
                {
                    "symbol": str(row.symbol),
                    "confidence_score": float(_safe_float(row.confidence_score, 0.0)),
                    "provider": str(row.source_provider or "INBOX"),
                    "reason": "low_confidence",
                    "last_bar_ts": None,
                    "provenance_confidence": float(_safe_float(row.confidence_score, 0.0)),
                    "source_kind": str(row.source_run_kind or "provider_updates").upper(),
                }
            )
            if len(output) >= cap:
                break
    if len(output) < cap and mode in {"missing", "all"}:
        for symbol in expected_symbols:
            token = str(symbol).upper()
            if token in by_symbol:
                continue
            output.append(
                {
                    "symbol": token,
                    "confidence_score": None,
                    "provider": None,
                    "reason": "missing_provenance",
                    "last_bar_ts": None,
                    "provenance_confidence": None,
                    "source_kind": None,
                }
            )
            if len(output) >= cap:
                break

    return {
        "bundle_id": int(bundle_id),
        "timeframe": str(timeframe),
        "trading_date": trading_date.isoformat(),
        "only": mode,
        "limit": cap,
        "rows": output[:cap],
        "low_confidence_threshold": low_threshold,
    }

