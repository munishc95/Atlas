from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import DailyConfidenceAggregate, DataProvenance
from app.services.data_store import DataStore
from app.services.fast_mode import fast_mode_enabled
from app.services.trading_calendar import (
    get_session as calendar_get_session,
    is_trading_day,
    previous_trading_day,
)


IST_ZONE = ZoneInfo("Asia/Kolkata")
_DEFAULT_TRADING_SEGMENT = "EQUITIES"


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _effective_trading_date(*, asof_ts: datetime, segment: str, settings: Settings) -> dt_date:
    day = asof_ts.astimezone(IST_ZONE).date()
    if is_trading_day(day, segment=segment, settings=settings):
        return day
    return previous_trading_day(day, segment=segment, settings=settings)


def _session_iso(day: dt_date, hhmm: str | None) -> str | None:
    if not isinstance(hhmm, str) or not hhmm.strip():
        return None
    parts = hhmm.split(":", maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        hour = max(0, min(23, int(parts[0])))
        minute = max(0, min(59, int(parts[1])))
    except ValueError:
        return None
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=IST_ZONE).isoformat()


def _latest_agg_for_day(
    session: Session,
    *,
    bundle_id: int | None,
    timeframe: str,
    trading_date: dt_date,
) -> DailyConfidenceAggregate | None:
    if not isinstance(bundle_id, int) or bundle_id <= 0:
        return None
    return session.exec(
        select(DailyConfidenceAggregate)
        .where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
        .where(DailyConfidenceAggregate.timeframe == str(timeframe))
        .where(DailyConfidenceAggregate.trading_date == trading_date)
        .order_by(DailyConfidenceAggregate.id.desc())
    ).first()


def _estimate_data_asof_ist(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int | None,
    timeframe: str,
    trading_date: dt_date,
    session_close_ist: str | None,
    store: DataStore | None = None,
) -> str | None:
    if not isinstance(bundle_id, int) or bundle_id <= 0:
        return _session_iso(trading_date, session_close_ist)

    timeframe_token = str(timeframe or "1d").strip().lower()
    if fast_mode_enabled(settings):
        return _session_iso(trading_date, session_close_ist)

    # Quick guard: if no provenance rows for the day, avoid expensive scans.
    row_count = int(
        session.exec(
            select(DataProvenance.id)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == str(timeframe))
            .where(DataProvenance.bar_date == trading_date)
            .limit(1)
        ).first()
        is not None
    )
    if row_count == 0:
        return None

    # Best-effort parquet inspection (bounded).
    if store is not None:
        try:
            symbols = store.get_bundle_symbols(session, int(bundle_id), timeframe=str(timeframe))
            day_start_utc = datetime(
                trading_date.year, trading_date.month, trading_date.day, 0, 0, tzinfo=UTC
            )
            day_end_utc = datetime(
                trading_date.year, trading_date.month, trading_date.day, 23, 59, 59, tzinfo=UTC
            )
            max_ts: datetime | None = None
            for symbol in symbols[:20]:
                frame = store.load_ohlcv(
                    symbol=symbol,
                    timeframe=str(timeframe),
                    start=day_start_utc,
                    end=day_end_utc,
                )
                if frame.empty:
                    continue
                ts = frame["datetime"].max()
                if isinstance(ts, datetime):
                    ts_utc = _to_utc(ts)
                else:
                    ts_utc = _to_utc(datetime.fromisoformat(str(ts)))
                if max_ts is None or ts_utc > max_ts:
                    max_ts = ts_utc
            if max_ts is not None:
                return max_ts.astimezone(IST_ZONE).isoformat()
        except Exception:  # noqa: BLE001
            pass

    close_iso = _session_iso(trading_date, session_close_ist)
    if timeframe_token in {"1d", "1day", "daily"}:
        return close_iso
    if timeframe_token in {"4h_ish", "4h_ish_resampled"}:
        # NSE "4h-ish" windows: 09:15-13:15 and 13:15-15:30.
        first_window_end = datetime(
            trading_date.year, trading_date.month, trading_date.day, 13, 15, tzinfo=IST_ZONE
        )
        close_dt = (
            datetime.fromisoformat(close_iso) if isinstance(close_iso, str) and close_iso else None
        )
        now_ist = datetime.now(IST_ZONE)
        if now_ist.date() == trading_date:
            if close_dt is not None and now_ist >= close_dt:
                return close_dt.isoformat()
            if now_ist >= first_window_end:
                return first_window_end.isoformat()
            return None
        return close_iso
    return close_iso


def build_effective_trading_context(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int | None,
    timeframe: str,
    asof_ts: datetime | None = None,
    segment: str | None = None,
    provider_stage_status: str | None = None,
    confidence_gate_decision: str | None = None,
    confidence_risk_scale: float | None = None,
    agg_row: DailyConfidenceAggregate | None = None,
    data_digest: str | None = None,
    engine_version: str | None = None,
    seed: int | None = None,
    notes: list[str] | None = None,
    store: DataStore | None = None,
) -> dict[str, Any]:
    timeframe_token = str(timeframe or "1d").strip() or "1d"
    segment_token = str(segment or settings.trading_calendar_segment or _DEFAULT_TRADING_SEGMENT)
    asof = _to_utc(asof_ts or datetime.now(UTC))
    trading_date = _effective_trading_date(
        asof_ts=asof,
        segment=segment_token,
        settings=settings,
    )
    session_info = calendar_get_session(
        trading_date,
        segment=segment_token,
        settings=settings,
    )
    session_open_ist = _session_iso(trading_date, session_info.get("open_time"))
    session_close_ist = _session_iso(trading_date, session_info.get("close_time"))

    resolved_agg = agg_row or _latest_agg_for_day(
        session,
        bundle_id=bundle_id,
        timeframe=timeframe_token,
        trading_date=trading_date,
    )
    decision = (
        str(confidence_gate_decision).upper()
        if confidence_gate_decision is not None
        else (
            str(resolved_agg.gate_decision).upper()
            if resolved_agg is not None
            else None
        )
    )
    risk_scale = (
        float(confidence_risk_scale)
        if confidence_risk_scale is not None
        else (
            float(resolved_agg.confidence_risk_scale)
            if resolved_agg is not None
            else None
        )
    )
    data_asof_ist = _estimate_data_asof_ist(
        session,
        settings=settings,
        bundle_id=bundle_id,
        timeframe=timeframe_token,
        trading_date=trading_date,
        session_close_ist=session_info.get("close_time"),
        store=store,
    )
    out_notes = list(notes or [])
    if data_asof_ist is None:
        out_notes.append("data_asof_unavailable")
    if resolved_agg is None:
        out_notes.append("confidence_aggregate_unavailable")

    return {
        "bundle_id": int(bundle_id) if isinstance(bundle_id, int) and bundle_id > 0 else None,
        "timeframe": timeframe_token,
        "trading_date": trading_date.isoformat(),
        "segment": segment_token,
        "session_open_ist": session_open_ist,
        "session_close_ist": session_close_ist,
        "now_ist": datetime.now(IST_ZONE).isoformat(),
        "data_asof_ist": data_asof_ist,
        "provider_stage_status": provider_stage_status,
        "confidence_gate_decision": decision,
        "confidence_risk_scale": risk_scale,
        "agg_id": int(resolved_agg.id) if resolved_agg is not None and resolved_agg.id is not None else None,
        "data_digest": data_digest,
        "engine_version": engine_version,
        "seed": int(seed) if isinstance(seed, int) else None,
        "notes": list(dict.fromkeys(str(item) for item in out_notes if str(item).strip())),
    }

