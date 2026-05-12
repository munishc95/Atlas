from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date as dt_date
from datetime import datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.db.models import DataQualityException


EXCEPTION_NO_TRADE_OR_SUSPENSION = "no_trade_or_suspension"
STATUS_ACTIVE = "ACTIVE"

AUTO_EXCEPTION_CLASSIFICATIONS = {
    "exchange_no_row_active_member",
    "exchange_no_row_membership_unknown",
}


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _parse_day(value: Any) -> dt_date | None:
    if isinstance(value, dt_date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str) and value.strip():
        try:
            return dt_date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def _clean_symbol(value: Any) -> str:
    return str(value or "").strip().upper()


def active_no_trade_exception_dates(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    symbols: list[str] | None = None,
) -> dict[str, set[dt_date]]:
    stmt = (
        select(DataQualityException)
        .where(DataQualityException.bundle_id == int(bundle_id))
        .where(DataQualityException.timeframe == str(timeframe))
        .where(DataQualityException.kind == EXCEPTION_NO_TRADE_OR_SUSPENSION)
        .where(DataQualityException.status == STATUS_ACTIVE)
    )
    symbol_tokens = [_clean_symbol(item) for item in (symbols or []) if _clean_symbol(item)]
    if symbol_tokens:
        stmt = stmt.where(DataQualityException.symbol.in_(symbol_tokens))
    rows = session.exec(stmt).all()
    by_symbol: dict[str, set[dt_date]] = defaultdict(set)
    for row in rows:
        by_symbol[str(row.symbol).upper()].add(row.trading_date)
    return dict(by_symbol)


def list_data_quality_exceptions(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str = "1d",
    symbol: str | None = None,
    kind: str | None = None,
    status: str | None = STATUS_ACTIVE,
    limit: int = 500,
) -> list[DataQualityException]:
    stmt = (
        select(DataQualityException)
        .where(DataQualityException.bundle_id == int(bundle_id))
        .where(DataQualityException.timeframe == str(timeframe))
    )
    if symbol:
        stmt = stmt.where(DataQualityException.symbol == _clean_symbol(symbol))
    if kind:
        stmt = stmt.where(DataQualityException.kind == str(kind))
    if status:
        stmt = stmt.where(DataQualityException.status == str(status).upper())
    return list(
        session.exec(
            stmt.order_by(
                DataQualityException.trading_date.desc(),
                DataQualityException.symbol.asc(),
            ).limit(max(1, min(5000, int(limit))))
        ).all()
    )


def serialize_data_quality_exception(row: DataQualityException) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "bundle_id": int(row.bundle_id),
        "timeframe": str(row.timeframe),
        "symbol": str(row.symbol),
        "trading_date": row.trading_date.isoformat(),
        "kind": str(row.kind),
        "status": str(row.status),
        "source": str(row.source),
        "source_report_id": int(row.source_report_id) if row.source_report_id is not None else None,
        "reason": str(row.reason or ""),
        "metadata": row.metadata_json if isinstance(row.metadata_json, dict) else {},
        "created_at": row.created_at.astimezone(timezone.utc).isoformat(),
        "updated_at": row.updated_at.astimezone(timezone.utc).isoformat(),
    }


def upsert_no_trade_exceptions_from_gap_audit(
    session: Session,
    *,
    payload: dict[str, Any],
    dry_run: bool = False,
    classifications: set[str] | None = None,
) -> dict[str, Any]:
    report = payload.get("report", {}) if isinstance(payload.get("report"), dict) else {}
    bundle_id = int(report.get("bundle_id") or 0)
    timeframe = str(report.get("timeframe") or "1d")
    report_id = int(report.get("id")) if report.get("id") is not None else None
    accepted = classifications or set(AUTO_EXCEPTION_CLASSIFICATIONS)
    rows = payload.get("symbols", []) if isinstance(payload.get("symbols"), list) else []

    candidates: dict[tuple[int, str, str, dt_date, str], dict[str, Any]] = {}
    skipped_counts: Counter[str] = Counter()
    for symbol_row in rows:
        if not isinstance(symbol_row, dict):
            continue
        symbol = _clean_symbol(symbol_row.get("symbol"))
        if not symbol:
            continue
        gaps = symbol_row.get("gaps", []) if isinstance(symbol_row.get("gaps"), list) else []
        for gap in gaps:
            if not isinstance(gap, dict):
                continue
            classified_dates = gap.get("classified_dates")
            if not isinstance(classified_dates, list):
                classified_dates = gap.get("sample_dates", [])
            if not isinstance(classified_dates, list):
                continue
            for item in classified_dates:
                if not isinstance(item, dict):
                    continue
                classification = str(item.get("classification", "")).strip()
                if classification not in accepted:
                    if classification:
                        skipped_counts[classification] += 1
                    continue
                day = _parse_day(item.get("date"))
                if day is None:
                    skipped_counts["invalid_date"] += 1
                    continue
                key = (bundle_id, timeframe, symbol, day, EXCEPTION_NO_TRADE_OR_SUSPENSION)
                candidates[key] = {
                    "classification": classification,
                    "reason": str(item.get("reason") or "Exchange source had no row for this date."),
                    "gap": {
                        "previous_bar_date": gap.get("previous_bar_date"),
                        "next_bar_date": gap.get("next_bar_date"),
                    },
                }

    existing: dict[tuple[int, str, str, dt_date, str], DataQualityException] = {}
    if candidates:
        symbols = sorted({key[2] for key in candidates})
        dates = sorted({key[3] for key in candidates})
        stmt = (
            select(DataQualityException)
            .where(DataQualityException.bundle_id == bundle_id)
            .where(DataQualityException.timeframe == timeframe)
            .where(DataQualityException.kind == EXCEPTION_NO_TRADE_OR_SUSPENSION)
            .where(DataQualityException.symbol.in_(symbols))
            .where(DataQualityException.trading_date.in_(dates))
        )
        for row in session.exec(stmt).all():
            existing[
                (
                    int(row.bundle_id),
                    str(row.timeframe),
                    str(row.symbol).upper(),
                    row.trading_date,
                    str(row.kind),
                )
            ] = row

    inserted = 0
    updated = 0
    now = _utc_now()
    samples: list[dict[str, Any]] = []
    for key, meta in sorted(candidates.items(), key=lambda item: (item[0][2], item[0][3])):
        bundle_key, timeframe_key, symbol, day, kind = key
        samples.append(
            {
                "symbol": symbol,
                "trading_date": day.isoformat(),
                "classification": meta["classification"],
            }
        )
        if dry_run:
            continue
        existing_row = existing.get(key)
        metadata = {
            "classification": meta["classification"],
            "gap": meta["gap"],
        }
        if existing_row is None:
            session.add(
                DataQualityException(
                    bundle_id=bundle_key,
                    timeframe=timeframe_key,
                    symbol=symbol,
                    trading_date=day,
                    kind=kind,
                    status=STATUS_ACTIVE,
                    source="gap_audit",
                    source_report_id=report_id,
                    reason=meta["reason"],
                    metadata_json=metadata,
                    created_at=now,
                    updated_at=now,
                )
            )
            inserted += 1
        else:
            existing_row.status = STATUS_ACTIVE
            existing_row.source = "gap_audit"
            existing_row.source_report_id = report_id
            existing_row.reason = meta["reason"]
            existing_row.metadata_json = metadata
            existing_row.updated_at = now
            session.add(existing_row)
            updated += 1
    if not dry_run:
        session.commit()

    return {
        "bundle_id": bundle_id,
        "timeframe": timeframe,
        "source_report_id": report_id,
        "dry_run": bool(dry_run),
        "candidate_count": len(candidates),
        "inserted_count": inserted,
        "updated_count": updated,
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "sample": samples[:20],
    }
