from __future__ import annotations

from datetime import UTC, date as dt_date, datetime, time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import (
    DataProvenance,
    DatasetBundle,
    HistoricalBackfillRun,
    ProviderUpdateItem,
    ProviderUpdateRun,
)
from app.services.data_store import DataStore
from app.services.operate_events import emit_operate_event
from app.services.provider_updates import run_provider_updates
from app.services.trading_calendar import list_trading_days


IST_ZONE = ZoneInfo("Asia/Kolkata")


def _parse_date_token(value: dt_date | str, *, field: str) -> dt_date:
    if isinstance(value, dt_date):
        return value
    token = str(value or "").strip()
    if not token:
        raise APIError(code="invalid_payload", message=f"{field} is required.")
    try:
        return dt_date.fromisoformat(token)
    except ValueError as exc:
        raise APIError(
            code="invalid_date",
            message=f"{field} must be in YYYY-MM-DD format.",
            details={field: token},
        ) from exc


def _to_utc_bounds(*, start_day: dt_date, end_day: dt_date) -> tuple[datetime, datetime]:
    start_ts = datetime.combine(start_day, dt_time(0, 0), tzinfo=IST_ZONE).astimezone(UTC)
    end_ts = datetime.combine(end_day, dt_time(23, 59, 59), tzinfo=IST_ZONE).astimezone(UTC)
    return start_ts, end_ts


def _normalize_mode(value: str | None) -> str:
    token = str(value or "SINGLE").strip().upper()
    return "FALLBACK" if token == "FALLBACK" else "SINGLE"


def _normalize_provider(value: str | None) -> str:
    token = str(value or "NSE_BHAVCOPY").strip().upper()
    if not token:
        return "NSE_BHAVCOPY"
    return token


def _serialize_issues(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    return [dict(item) for item in (items or []) if isinstance(item, dict)]


def serialize_historical_backfill_run(row: HistoricalBackfillRun) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "bundle_id": row.bundle_id,
        "timeframe": row.timeframe,
        "provider_kind": row.provider_kind,
        "mode": row.mode,
        "dry_run": bool(row.dry_run),
        "start_date": row.start_date.isoformat(),
        "end_date": row.end_date.isoformat(),
        "status": row.status,
        "trading_days_planned": int(row.trading_days_planned or 0),
        "trading_days_completed": int(row.trading_days_completed or 0),
        "symbols_targeted": int(row.symbols_targeted or 0),
        "bars_updated_total": int(row.bars_updated_total or 0),
        "bars_added_total": int(row.bars_added_total or 0),
        "warnings_json": _serialize_issues(row.warnings_json),
        "errors_json": _serialize_issues(row.errors_json),
        "started_at": row.started_at.isoformat() if row.started_at is not None else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at is not None else None,
        "created_at": row.created_at.isoformat(),
    }


def run_historical_backfill(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    provider_kind: str,
    mode: str,
    start_date: dt_date | str,
    end_date: dt_date | str,
    dry_run: bool = False,
    overrides: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> HistoricalBackfillRun:
    bundle = session.get(DatasetBundle, int(bundle_id))
    if bundle is None:
        raise APIError(
            code="bundle_not_found",
            message=f"Bundle {bundle_id} not found.",
            status_code=404,
        )
    tf = str(timeframe or "1d").strip().lower()
    if tf != "1d":
        raise APIError(
            code="unsupported_backfill_timeframe",
            message="Historical backfill currently supports 1d timeframe only.",
            details={"timeframe": tf},
        )
    provider = _normalize_provider(provider_kind)
    provider_mode = _normalize_mode(mode)
    start_day = _parse_date_token(start_date, field="start_date")
    end_day = _parse_date_token(end_date, field="end_date")
    if end_day < start_day:
        raise APIError(
            code="invalid_date_range",
            message="end_date must be on or after start_date.",
            details={"start_date": start_day.isoformat(), "end_date": end_day.isoformat()},
        )

    segment = str(settings.trading_calendar_segment or "EQUITIES")
    trading_days = list_trading_days(
        start_date=start_day,
        end_date=end_day,
        segment=segment,
        settings=settings,
    )
    max_days = max(1, int(settings.historical_backfill_max_trading_days_per_run))
    if len(trading_days) > max_days:
        raise APIError(
            code="backfill_range_too_large",
            message=(
                f"Requested range has {len(trading_days)} trading days; "
                f"max allowed per run is {max_days}."
            ),
            details={
                "max_trading_days_per_run": max_days,
                "requested_trading_days": len(trading_days),
            },
        )

    started_at = datetime.now(UTC)
    row = HistoricalBackfillRun(
        bundle_id=int(bundle_id),
        timeframe=tf,
        provider_kind=provider,
        mode=provider_mode,
        dry_run=bool(dry_run),
        start_date=start_day,
        end_date=end_day,
        status="RUNNING",
        trading_days_planned=int(len(trading_days)),
        started_at=started_at,
    )
    session.add(row)
    session.commit()
    session.refresh(row)

    emit_operate_event(
        session,
        severity="INFO",
        category="DATA",
        message="historical_backfill_started",
        details={
            "run_id": row.id,
            "bundle_id": int(bundle_id),
            "timeframe": tf,
            "provider_kind": provider,
            "mode": provider_mode,
            "start_date": start_day.isoformat(),
            "end_date": end_day.isoformat(),
            "trading_days_planned": int(len(trading_days)),
            "dry_run": bool(dry_run),
        },
        correlation_id=correlation_id,
    )
    session.commit()

    try:
        symbols = store.get_bundle_symbols(session, int(bundle_id), timeframe=tf)
        row.symbols_targeted = int(len(symbols))

        if dry_run:
            row.status = "SUCCEEDED"
            row.trading_days_completed = 0
            row.finished_at = datetime.now(UTC)
            session.add(row)
            session.commit()
            session.refresh(row)
            emit_operate_event(
                session,
                severity="INFO",
                category="DATA",
                message="historical_backfill_dry_run_completed",
                details={
                    "run_id": row.id,
                    "bundle_id": int(bundle_id),
                    "timeframe": tf,
                    "provider_kind": provider,
                    "mode": provider_mode,
                    "trading_days_planned": int(row.trading_days_planned),
                    "symbols_targeted": int(row.symbols_targeted),
                },
                correlation_id=correlation_id,
                commit=True,
            )
            return row

        start_ts, end_ts = _to_utc_bounds(start_day=start_day, end_day=end_day)
        local_overrides = dict(overrides or {})
        local_overrides["data_updates_provider_mode"] = provider_mode
        if provider_mode == "FALLBACK":
            local_overrides.setdefault(
                "data_updates_provider_priority_order",
                [provider, "UPSTOX", "NSE_BHAVCOPY", "NSE_EOD", "INBOX"],
            )
        if provider == "NSE_BHAVCOPY":
            local_overrides["data_updates_provider_nse_bhavcopy_enabled"] = True
        if provider == "NSE_EOD":
            local_overrides["data_updates_provider_nse_eod_enabled"] = True

        provider_row = run_provider_updates(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle_id),
            timeframe=tf,
            overrides=local_overrides,
            provider_kind=provider,
            start=start_ts,
            end=end_ts,
            correlation_id=correlation_id,
        )

        bars_updated_total = 0
        if provider_row.id is not None:
            items = session.exec(
                select(ProviderUpdateItem).where(
                    ProviderUpdateItem.run_id == int(provider_row.id)
                )
            ).all()
            bars_updated_total = int(sum(int(item.bars_updated or 0) for item in items))
            prov_rows = session.exec(
                select(DataProvenance)
                .where(DataProvenance.bundle_id == int(bundle_id))
                .where(DataProvenance.timeframe == tf)
                .where(DataProvenance.source_run_kind == "provider_updates")
                .where(DataProvenance.source_run_id == str(provider_row.id))
            ).all()
            completed_days = {
                item.bar_date
                for item in prov_rows
                if start_day <= item.bar_date <= end_day
            }
            row.trading_days_completed = int(len(completed_days))

        row.status = "FAILED" if str(provider_row.status).upper() == "FAILED" else "SUCCEEDED"
        row.bars_added_total = int(provider_row.bars_added or 0)
        row.bars_updated_total = int(bars_updated_total)
        row.symbols_targeted = int(provider_row.symbols_attempted or row.symbols_targeted or 0)
        row.warnings_json = _serialize_issues(provider_row.warnings_json)
        row.errors_json = _serialize_issues(provider_row.errors_json)
        row.finished_at = datetime.now(UTC)
        session.add(row)
        session.commit()
        session.refresh(row)

        emit_operate_event(
            session,
            severity=("ERROR" if row.status == "FAILED" else ("WARN" if row.warnings_json else "INFO")),
            category="DATA",
            message=(
                "historical_backfill_failed"
                if row.status == "FAILED"
                else (
                    "historical_backfill_completed_with_warnings"
                    if row.warnings_json
                    else "historical_backfill_completed"
                )
            ),
            details={
                "run_id": row.id,
                "bundle_id": int(bundle_id),
                "timeframe": tf,
                "provider_kind": provider,
                "mode": provider_mode,
                "status": row.status,
                "trading_days_planned": int(row.trading_days_planned),
                "trading_days_completed": int(row.trading_days_completed),
                "symbols_targeted": int(row.symbols_targeted),
                "bars_added_total": int(row.bars_added_total),
                "bars_updated_total": int(row.bars_updated_total),
                "warnings_count": len(row.warnings_json or []),
                "errors_count": len(row.errors_json or []),
            },
            correlation_id=correlation_id,
            commit=True,
        )
        return row
    except Exception:
        row.status = "FAILED"
        row.finished_at = datetime.now(UTC)
        session.add(row)
        session.commit()
        session.refresh(row)
        raise


def get_latest_historical_backfill_run(
    session: Session,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = None,
) -> HistoricalBackfillRun | None:
    stmt = select(HistoricalBackfillRun).order_by(
        HistoricalBackfillRun.created_at.desc(),
        HistoricalBackfillRun.id.desc(),
    )
    if bundle_id is not None:
        stmt = stmt.where(HistoricalBackfillRun.bundle_id == int(bundle_id))
    if timeframe:
        stmt = stmt.where(HistoricalBackfillRun.timeframe == str(timeframe))
    return session.exec(stmt).first()


def list_historical_backfill_runs(
    session: Session,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = None,
    limit: int = 20,
) -> list[HistoricalBackfillRun]:
    stmt = select(HistoricalBackfillRun)
    if bundle_id is not None:
        stmt = stmt.where(HistoricalBackfillRun.bundle_id == int(bundle_id))
    if timeframe:
        stmt = stmt.where(HistoricalBackfillRun.timeframe == str(timeframe))
    stmt = stmt.order_by(HistoricalBackfillRun.created_at.desc(), HistoricalBackfillRun.id.desc())
    stmt = stmt.limit(max(1, min(int(limit), 500)))
    return list(session.exec(stmt).all())
