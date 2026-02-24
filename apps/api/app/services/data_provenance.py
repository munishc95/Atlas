from __future__ import annotations

from datetime import UTC, date as dt_date, datetime, timedelta
from typing import Any

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import DataProvenance, DataUpdateRun, ProviderUpdateRun
from app.services.upstox_auth import token_status as upstox_token_status


def _provider_token(value: str | None) -> str:
    return str(value or "INBOX").strip().upper() or "INBOX"


def confidence_for_provider(
    *,
    provider: str,
    settings: Settings,
    overrides: dict[str, Any] | None = None,
) -> float:
    state = overrides or {}
    token = _provider_token(provider)
    if token == "UPSTOX":
        return float(state.get("data_provenance_confidence_upstox", settings.data_provenance_confidence_upstox))
    if token == "NSE_EOD":
        return float(state.get("data_provenance_confidence_nse_eod", settings.data_provenance_confidence_nse_eod))
    if token == "INBOX":
        return float(state.get("data_provenance_confidence_inbox", settings.data_provenance_confidence_inbox))
    return 60.0


def upsert_provenance_rows(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    symbol: str,
    bar_dates: list[dt_date],
    source_provider: str,
    source_run_kind: str,
    source_run_id: str | None,
    confidence_score: float,
    reason: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    token = _provider_token(source_provider)
    symbol_up = str(symbol).strip().upper()
    tf = str(timeframe or "1d").strip()
    for bar_day in sorted(set(bar_dates)):
        existing = session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == tf)
            .where(DataProvenance.symbol == symbol_up)
            .where(DataProvenance.bar_date == bar_day)
            .order_by(DataProvenance.id.desc())
        ).first()
        if existing is None:
            existing = DataProvenance(
                bundle_id=int(bundle_id),
                timeframe=tf,
                symbol=symbol_up,
                bar_date=bar_day,
            )
        existing.source_provider = token
        existing.source_run_kind = str(source_run_kind or "provider_updates").strip().lower()
        existing.source_run_id = str(source_run_id) if source_run_id is not None else None
        existing.confidence_score = float(confidence_score)
        existing.reason = str(reason).strip() if isinstance(reason, str) and reason.strip() else None
        existing.metadata_json = dict(metadata or {})
        existing.updated_at = datetime.now(UTC)
        session.add(existing)


def list_provenance(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    symbol: str | None = None,
    from_date: dt_date | None = None,
    to_date: dt_date | None = None,
    limit: int = 2000,
) -> list[DataProvenance]:
    stmt = (
        select(DataProvenance)
        .where(DataProvenance.bundle_id == int(bundle_id))
        .where(DataProvenance.timeframe == str(timeframe))
    )
    if symbol:
        stmt = stmt.where(DataProvenance.symbol == str(symbol).upper())
    if from_date is not None:
        stmt = stmt.where(DataProvenance.bar_date >= from_date)
    if to_date is not None:
        stmt = stmt.where(DataProvenance.bar_date <= to_date)
    stmt = stmt.order_by(DataProvenance.bar_date.desc(), DataProvenance.symbol.asc()).limit(
        max(1, min(int(limit), 10_000))
    )
    return list(session.exec(stmt).all())


def provenance_summary(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    latest_day: dt_date | None = None,
    low_conf_threshold: float = 65.0,
    lookback_days: int = 20,
) -> dict[str, Any]:
    tf = str(timeframe or "1d").strip()
    latest = latest_day
    if latest is None:
        row = session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == tf)
            .order_by(DataProvenance.bar_date.desc())
        ).first()
        latest = row.bar_date if row is not None else None
    if latest is None:
        return {
            "latest_day": None,
            "coverage_by_source_provider": {},
            "low_confidence_days_count": 0,
            "low_confidence_symbols_count": 0,
            "latest_day_all_low_confidence": False,
        }

    latest_rows = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == tf)
            .where(DataProvenance.bar_date == latest)
        ).all()
    )
    by_provider: dict[str, int] = {}
    low_conf_symbols = 0
    for row in latest_rows:
        provider = _provider_token(row.source_provider)
        by_provider[provider] = by_provider.get(provider, 0) + 1
        if float(row.confidence_score or 0.0) < float(low_conf_threshold):
            low_conf_symbols += 1
    total_latest = max(1, len(latest_rows))
    by_provider_pct = {
        key: round((value / total_latest) * 100.0, 3) for key, value in sorted(by_provider.items())
    }
    latest_all_low = len(latest_rows) > 0 and low_conf_symbols == len(latest_rows)

    lookback_start = latest - timedelta(days=max(1, int(lookback_days * 3)))
    lookback_rows = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == tf)
            .where(DataProvenance.bar_date >= lookback_start)
            .where(DataProvenance.bar_date <= latest)
        ).all()
    )
    days_low: set[dt_date] = set()
    for row in lookback_rows:
        if float(row.confidence_score or 0.0) < float(low_conf_threshold):
            days_low.add(row.bar_date)
    return {
        "latest_day": latest.isoformat(),
        "coverage_by_source_provider": by_provider_pct,
        "low_confidence_days_count": len(days_low),
        "low_confidence_symbols_count": int(low_conf_symbols),
        "latest_day_all_low_confidence": latest_all_low,
    }


def provider_status_payload(
    session: Session,
    *,
    settings: Settings,
) -> dict[str, Any]:
    providers = ["UPSTOX", "NSE_EOD", "INBOX"]
    latest_runs = list(
        session.exec(select(ProviderUpdateRun).order_by(ProviderUpdateRun.created_at.desc())).all()
    )
    latest_data_update = session.exec(
        select(DataUpdateRun).order_by(DataUpdateRun.created_at.desc(), DataUpdateRun.id.desc())
    ).first()
    rows: dict[str, dict[str, Any]] = {}
    for provider in providers:
        rows[provider] = {
            "provider": provider,
            "last_run_at": None,
            "last_status": "NOT_RUN",
            "notes": [],
        }
    for row in latest_runs:
        token = _provider_token(row.provider_kind)
        if token not in rows:
            rows[token] = {
                "provider": token,
                "last_run_at": None,
                "last_status": "NOT_RUN",
                "notes": [],
            }
        if rows[token]["last_run_at"] is None:
            rows[token]["last_run_at"] = row.created_at.isoformat()
            rows[token]["last_status"] = str(row.status)
            rows[token]["run_id"] = int(row.id or 0)
            rows[token]["timeframe"] = row.timeframe
            rows[token]["bundle_id"] = row.bundle_id
            rows[token]["bars_added"] = int(row.bars_added or 0)
    if latest_data_update is not None:
        rows["INBOX"]["last_run_at"] = latest_data_update.created_at.isoformat()
        rows["INBOX"]["last_status"] = str(latest_data_update.status)
        rows["INBOX"]["run_id"] = int(latest_data_update.id or 0)
        rows["INBOX"]["timeframe"] = latest_data_update.timeframe
        rows["INBOX"]["bundle_id"] = latest_data_update.bundle_id
        rows["INBOX"]["rows_ingested"] = int(latest_data_update.rows_ingested or 0)

    upstox_meta = upstox_token_status(session, settings=settings, allow_env_fallback=True)
    rows["UPSTOX"]["token"] = {
        "connected": bool(upstox_meta.get("connected")),
        "is_expired": bool(upstox_meta.get("is_expired")),
        "expires_at": upstox_meta.get("expires_at"),
        "last_verified_at": upstox_meta.get("last_verified_at"),
    }
    rows["NSE_EOD"]["enabled"] = bool(settings.data_updates_provider_nse_eod_enabled)
    return {
        "providers": [rows[key] for key in sorted(rows.keys())],
        "upstox_token_status": rows["UPSTOX"].get("token", {}),
    }


def provider_status_trend(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    days: int = 30,
) -> list[dict[str, Any]]:
    tf = str(timeframe or "1d").strip()
    latest_row = session.exec(
        select(DataProvenance)
        .where(DataProvenance.bundle_id == int(bundle_id))
        .where(DataProvenance.timeframe == tf)
        .order_by(DataProvenance.bar_date.desc())
        .limit(1)
    ).first()
    if latest_row is None:
        return []
    latest_day = latest_row.bar_date
    lookback_days = max(1, min(int(days), 365))
    rows = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == tf)
            .where(DataProvenance.bar_date <= latest_day)
            .order_by(DataProvenance.bar_date.desc())
        ).all()
    )
    per_day: dict[dt_date, list[DataProvenance]] = {}
    for row in rows:
        per_day.setdefault(row.bar_date, []).append(row)
    selected_days = sorted(per_day.keys())[-lookback_days:]
    output: list[dict[str, Any]] = []
    for day in selected_days:
        day_rows = per_day.get(day, [])
        total = max(1, len(day_rows))
        counts: dict[str, int] = {}
        conf_sum = 0.0
        low_conf_count = 0
        for row in day_rows:
            provider = _provider_token(row.source_provider)
            counts[provider] = int(counts.get(provider, 0)) + 1
            score = float(row.confidence_score or 0.0)
            conf_sum += score
            if score < 65.0:
                low_conf_count += 1
        mix = {
            provider: round(count / total, 6)
            for provider, count in sorted(counts.items())
        }
        dominant = max(mix.items(), key=lambda item: item[1])[0] if mix else None
        output.append(
            {
                "trading_date": day.isoformat(),
                "provider_counts": counts,
                "provider_mix": mix,
                "dominant_provider": dominant,
                "avg_confidence": round(conf_sum / total, 6),
                "pct_low_confidence": round(low_conf_count / total, 6),
                "symbols": int(len(day_rows)),
            }
        )
    return output
