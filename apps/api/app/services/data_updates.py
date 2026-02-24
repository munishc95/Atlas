from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DataUpdateFile, DataUpdateRun, DatasetBundle
from app.services.data_provenance import confidence_for_provider, upsert_provenance_rows
from app.services.data_store import DataStore
from app.services.importer import _validate_numeric
from app.services.operate_events import emit_operate_event
from app.services.trading_calendar import (
    get_session as calendar_get_session,
    is_trading_day,
    list_trading_days,
    parse_time_hhmm,
    previous_trading_day,
)


IST_ZONE = ZoneInfo("Asia/Kolkata")
STATUS_OK = "OK"
STATUS_WARN = "WARN"
STATUS_FAIL = "FAIL"

_SYMBOL_ALIASES = ("symbol", "ticker", "tradingsymbol", "scrip", "security")
_DATETIME_ALIASES = ("datetime", "date", "timestamp", "time", "trade_date")
_OPEN_ALIASES = ("open", "o")
_HIGH_ALIASES = ("high", "h")
_LOW_ALIASES = ("low", "l")
_CLOSE_ALIASES = ("close", "c")
_VOLUME_ALIASES = ("volume", "vol")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _segment(settings: Settings, overrides: dict[str, Any] | None = None) -> str:
    state = overrides or {}
    token = str(state.get("trading_calendar_segment", settings.trading_calendar_segment)).strip().upper()
    return token or "EQUITIES"


def _hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _column_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return {str(col).strip().lower(): str(col) for col in frame.columns}


def _pick_column(lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def _read_inbox_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise APIError(code="unsupported_file_type", message=f"Unsupported inbox file extension: {suffix}")


def _normalize_update_payload(
    *,
    frame: pd.DataFrame,
    symbol_hint: str | None,
) -> dict[str, pd.DataFrame]:
    if frame.empty:
        raise APIError(code="empty_data", message="Inbox file has no rows.")

    lookup = _column_lookup(frame)
    symbol_col = _pick_column(lookup, _SYMBOL_ALIASES)
    datetime_col = _pick_column(lookup, _DATETIME_ALIASES)
    open_col = _pick_column(lookup, _OPEN_ALIASES)
    high_col = _pick_column(lookup, _HIGH_ALIASES)
    low_col = _pick_column(lookup, _LOW_ALIASES)
    close_col = _pick_column(lookup, _CLOSE_ALIASES)
    volume_col = _pick_column(lookup, _VOLUME_ALIASES)

    missing: list[str] = []
    if datetime_col is None:
        missing.append("datetime")
    if open_col is None:
        missing.append("open")
    if high_col is None:
        missing.append("high")
    if low_col is None:
        missing.append("low")
    if close_col is None:
        missing.append("close")
    if volume_col is None:
        missing.append("volume")
    if missing:
        raise APIError(
            code="invalid_schema",
            message="Inbox file missing required OHLCV columns.",
            details={"missing": missing},
        )

    if symbol_col is None and not symbol_hint:
        raise APIError(
            code="missing_symbol_column",
            message="Inbox file must include symbol column when bundle has multiple symbols.",
        )

    normalized = pd.DataFrame(
        {
            "symbol": (
                frame[symbol_col]
                if symbol_col is not None
                else pd.Series([str(symbol_hint)] * len(frame), index=frame.index)
            ),
            "datetime": frame[datetime_col],
            "open": frame[open_col],
            "high": frame[high_col],
            "low": frame[low_col],
            "close": frame[close_col],
            "volume": frame[volume_col],
        }
    )
    normalized["symbol"] = normalized["symbol"].astype(str).str.strip().str.upper()
    normalized = normalized[normalized["symbol"] != ""]
    if normalized.empty:
        raise APIError(code="empty_data", message="No symbols found after normalization.")

    per_symbol: dict[str, pd.DataFrame] = {}
    for symbol, group in normalized.groupby("symbol", sort=True):
        clean = _validate_numeric(group[["datetime", "open", "high", "low", "close", "volume"]])
        if clean.empty:
            continue
        clean = clean.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        per_symbol[str(symbol).upper()] = clean.reset_index(drop=True)

    if not per_symbol:
        raise APIError(code="empty_data", message="No valid symbol rows after validation.")
    return per_symbol


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    base = existing.copy()
    if not base.empty:
        base["datetime"] = pd.to_datetime(base["datetime"], utc=True, errors="coerce")
        base = base.dropna(subset=["datetime"])
        base = base.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    base_count = int(len(base))

    inc = incoming.copy()
    inc["datetime"] = pd.to_datetime(inc["datetime"], utc=True, errors="coerce")
    inc = inc.dropna(subset=["datetime"])
    inc = inc.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    if inc.empty:
        return base.reset_index(drop=True), 0

    merged = (
        pd.concat([base, inc], ignore_index=True)
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )
    added = max(0, int(len(merged) - base_count))
    return merged, added


def _expected_latest_trading_day(
    *,
    settings: Settings,
    overrides: dict[str, Any] | None,
    timeframe: str,
    reference_ts: datetime | None = None,
) -> datetime.date:
    now = reference_ts if reference_ts is not None else datetime.now(timezone.utc)
    now_ist = now.astimezone(IST_ZONE)
    today_ist = now_ist.date()
    segment = _segment(settings, overrides=overrides)
    if not is_trading_day(today_ist, segment=segment, settings=settings):
        return previous_trading_day(today_ist, segment=segment, settings=settings)
    session_info = calendar_get_session(today_ist, segment=segment, settings=settings)
    close_cutoff = parse_time_hhmm(
        str(session_info.get("close_time") or settings.nse_equities_close_time_ist),
        default=settings.nse_equities_close_time_ist,
    )
    if str(timeframe).lower() == "1d":
        if now_ist.time() >= close_cutoff:
            return today_ist
        return previous_trading_day(today_ist, segment=segment, settings=settings)
    if now_ist.time() < close_cutoff:
        return previous_trading_day(today_ist, segment=segment, settings=settings)
    return today_ist


def compute_data_coverage(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    overrides: dict[str, Any] | None = None,
    reference_ts: datetime | None = None,
    top_n: int = 50,
) -> dict[str, Any]:
    tf = str(timeframe or "1d").strip()
    symbols = store.get_bundle_symbols(session, bundle_id, timeframe=tf)
    expected_day = _expected_latest_trading_day(
        settings=settings,
        overrides=overrides,
        timeframe=tf,
        reference_ts=reference_ts,
    )
    state = overrides or {}
    warn_pct = max(
        0.0,
        min(
            100.0,
            _safe_float(
                state.get("coverage_missing_latest_warn_pct", settings.coverage_missing_latest_warn_pct),
                settings.coverage_missing_latest_warn_pct,
            ),
        ),
    )
    fail_pct = max(
        warn_pct,
        min(
            100.0,
            _safe_float(
                state.get("coverage_missing_latest_fail_pct", settings.coverage_missing_latest_fail_pct),
                settings.coverage_missing_latest_fail_pct,
            ),
        ),
    )
    inactive_after_days = max(
        1,
        _safe_int(
            state.get(
                "coverage_inactive_after_missing_days",
                settings.coverage_inactive_after_missing_days,
            ),
            settings.coverage_inactive_after_missing_days,
        ),
    )
    segment = _segment(settings, overrides=overrides)
    operate_mode = str(state.get("operate_mode", settings.operate_mode)).strip().lower()

    stale_symbols: list[dict[str, Any]] = []
    last_bar_by_symbol: list[dict[str, Any]] = []
    missing_symbols: list[str] = []
    inactive_symbols: list[str] = []
    latest_bar_ts: datetime | None = None

    for symbol in symbols:
        frame = store.load_ohlcv(symbol=symbol, timeframe=tf)
        if frame.empty:
            missing_symbols.append(symbol)
            inactive_symbols.append(symbol)
            stale_symbols.append(
                {
                    "symbol": symbol,
                    "missing_trading_days": inactive_after_days + 1,
                    "last_bar_day_ist": None,
                }
            )
            last_bar_by_symbol.append(
                {
                    "symbol": symbol,
                    "last_bar_ts": None,
                    "missing_trading_days": inactive_after_days + 1,
                }
            )
            continue

        dt = pd.to_datetime(frame["datetime"], utc=True)
        symbol_last_ts = dt.max().to_pydatetime()
        if latest_bar_ts is None or symbol_last_ts > latest_bar_ts:
            latest_bar_ts = symbol_last_ts
        last_day_ist = symbol_last_ts.astimezone(IST_ZONE).date()
        if last_day_ist < expected_day:
            missing_days = len(
                list_trading_days(
                    start_date=last_day_ist + timedelta(days=1),
                    end_date=expected_day,
                    segment=segment,
                    settings=settings,
                )
            )
        else:
            missing_days = 0

        if missing_days > 0:
            missing_symbols.append(symbol)
            stale_symbols.append(
                {
                    "symbol": symbol,
                    "missing_trading_days": int(missing_days),
                    "last_bar_day_ist": last_day_ist.isoformat(),
                }
            )
        if missing_days >= inactive_after_days:
            inactive_symbols.append(symbol)

        last_bar_by_symbol.append(
            {
                "symbol": symbol,
                "last_bar_ts": symbol_last_ts.isoformat(),
                "missing_trading_days": int(missing_days),
            }
        )

    total = len(symbols)
    missing_count = len(missing_symbols)
    coverage_pct = 0.0 if total == 0 else float(((total - missing_count) / total) * 100.0)
    missing_pct = 0.0 if total == 0 else float((missing_count / total) * 100.0)
    status = STATUS_OK
    if missing_pct >= fail_pct:
        status = STATUS_FAIL
    elif missing_pct >= warn_pct:
        status = STATUS_WARN
    if operate_mode != "live" and status == STATUS_FAIL:
        # Offline/local operation should remain permissive; surface drift as WARN.
        status = STATUS_WARN

    stale_symbols.sort(
        key=lambda item: (-int(item.get("missing_trading_days", 0)), str(item.get("symbol", "")))
    )
    last_bar_by_symbol.sort(
        key=lambda item: (
            -int(item.get("missing_trading_days", 0)),
            str(item.get("symbol", "")),
        )
    )

    return {
        "bundle_id": int(bundle_id),
        "timeframe": tf,
        "coverage_pct": coverage_pct,
        "missing_pct": missing_pct,
        "status": status,
        "expected_latest_trading_day": expected_day.isoformat(),
        "missing_symbols": sorted(missing_symbols),
        "stale_symbols": stale_symbols,
        "inactive_symbols": sorted(set(inactive_symbols)),
        "last_bar_by_symbol": last_bar_by_symbol[: max(1, int(top_n))],
        "checked_symbols": total,
        "total_symbols": total,
        "last_bar_ts": latest_bar_ts.isoformat() if latest_bar_ts is not None else None,
        "thresholds": {
            "warn_pct": warn_pct,
            "fail_pct": fail_pct,
            "inactive_after_missing_days": inactive_after_days,
        },
        "operate_mode": operate_mode,
    }


def inactive_symbols_for_selection(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    overrides: dict[str, Any] | None = None,
    reference_ts: datetime | None = None,
) -> set[str]:
    summary = compute_data_coverage(
        session=session,
        settings=settings,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
        overrides=overrides,
        reference_ts=reference_ts,
        top_n=2000,
    )
    return {str(symbol).upper() for symbol in summary.get("inactive_symbols", [])}


def run_data_updates(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    overrides: dict[str, Any] | None = None,
    max_files_per_run: int | None = None,
    correlation_id: str | None = None,
) -> DataUpdateRun:
    bundle = session.get(DatasetBundle, int(bundle_id))
    if bundle is None:
        raise APIError(code="bundle_not_found", message=f"Bundle {bundle_id} not found.", status_code=404)

    tf = str(timeframe or "1d").strip()
    state = overrides or {}
    inbox_enabled = bool(state.get("data_updates_inbox_enabled", settings.data_updates_inbox_enabled))
    max_files = max(
        1,
        int(
            max_files_per_run
            if max_files_per_run is not None
            else _safe_int(
                state.get("data_updates_max_files_per_run", settings.data_updates_max_files_per_run),
                settings.data_updates_max_files_per_run,
            )
        ),
    )
    inbox_dir = Path(settings.data_inbox_root) / str(bundle.name) / tf
    run = DataUpdateRun(
        bundle_id=int(bundle.id) if bundle.id is not None else None,
        timeframe=tf,
        status="RUNNING",
        inbox_path=str(inbox_dir),
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    symbols_affected: set[str] = set()
    rows_ingested = 0
    processed_files = 0
    skipped_files = 0

    if not inbox_enabled:
        warnings.append(
            {
                "code": "inbox_disabled",
                "message": "Inbox updates are disabled by runtime settings.",
            }
        )
        run.status = "SUCCEEDED"
        run.warnings_json = warnings
        run.ended_at = datetime.now(timezone.utc)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run

    if not inbox_dir.exists():
        warnings.append(
            {
                "code": "inbox_missing",
                "message": f"Inbox folder does not exist: {inbox_dir}",
            }
        )
        run.status = "SUCCEEDED"
        run.warnings_json = warnings
        run.ended_at = datetime.now(timezone.utc)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run

    files = sorted(
        [
            path
            for path in inbox_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}
        ],
        key=lambda path: (path.name.lower(), path.stat().st_mtime_ns),
    )
    run.scanned_files = len(files)
    selected_files = files[:max_files]
    if len(files) > len(selected_files):
        warnings.append(
            {
                "code": "max_files_cap",
                "message": f"Processing limited to first {len(selected_files)} files.",
                "details": {"max_files_per_run": max_files, "scanned_files": len(files)},
            }
        )

    bundle_symbols = set(store.get_bundle_symbols(session, int(bundle_id), timeframe=tf))
    symbol_hint = None
    if len(bundle_symbols) == 1:
        symbol_hint = next(iter(bundle_symbols))

    for path in selected_files:
        file_hash = _hash_path(path)
        duplicate = session.exec(
            select(DataUpdateFile)
            .where(DataUpdateFile.bundle_id == int(bundle_id))
            .where(DataUpdateFile.timeframe == tf)
            .where(DataUpdateFile.file_hash == file_hash)
            .where(DataUpdateFile.status == "INGESTED")
            .order_by(DataUpdateFile.id.desc())
        ).first()
        if duplicate is not None:
            skipped_files += 1
            session.add(
                DataUpdateFile(
                    run_id=int(run.id or 0),
                    bundle_id=int(bundle_id),
                    timeframe=tf,
                    file_path=str(path),
                    file_name=path.name,
                    file_hash=file_hash,
                    status="SKIPPED",
                    reason="duplicate_file_hash",
                )
            )
            continue

        file_rows_ingested = 0
        file_symbols: set[str] = set()
        file_warnings: list[dict[str, Any]] = []
        file_errors: list[dict[str, Any]] = []
        file_status = "INGESTED"
        file_reason: str | None = None
        try:
            raw_frame = _read_inbox_frame(path)
            parsed = _normalize_update_payload(frame=raw_frame, symbol_hint=symbol_hint)
            allowed_parsed: dict[str, pd.DataFrame] = {}
            for symbol, frame in parsed.items():
                if bundle_symbols and symbol not in bundle_symbols:
                    file_warnings.append(
                        {
                            "code": "symbol_not_in_bundle",
                            "message": f"Skipped {symbol} because it is not in bundle {bundle.name}.",
                        }
                    )
                    continue
                allowed_parsed[symbol] = frame
            if not allowed_parsed:
                file_status = "SKIPPED"
                file_reason = "no_bundle_symbols_in_file"
            else:
                for symbol, incoming in allowed_parsed.items():
                    existing = store.load_ohlcv(symbol=symbol, timeframe=tf)
                    merged, added = _merge_frames(existing, incoming)
                    if added <= 0:
                        continue
                    instrument = store.find_instrument(session, symbol=symbol)
                    instrument_kind = instrument.kind if instrument is not None else "EQUITY_CASH"
                    store.save_ohlcv(
                        session=session,
                        symbol=symbol,
                        timeframe=tf,
                        frame=merged,
                        provider=f"{bundle.provider}-inbox",
                        checksum=file_hash,
                        instrument_kind=instrument_kind,
                        underlying=instrument.underlying if instrument is not None else symbol,
                        lot_size=instrument.lot_size if instrument is not None else 1,
                        tick_size=float(instrument.tick_size if instrument is not None else 0.05),
                        bundle_id=int(bundle_id),
                    )
                    incoming_dates = (
                        pd.to_datetime(incoming["datetime"], utc=True, errors="coerce")
                        .dt.tz_convert(IST_ZONE)
                        .dt.date
                    )
                    bar_dates = [
                        day
                        for day in sorted(set(incoming_dates.tolist()))
                        if day is not None
                    ]
                    if bar_dates:
                        inbox_confidence = confidence_for_provider(
                            provider="INBOX",
                            settings=settings,
                            overrides=state,
                        )
                        upsert_provenance_rows(
                            session,
                            bundle_id=int(bundle_id),
                            timeframe=tf,
                            symbol=symbol,
                            bar_dates=bar_dates,
                            source_provider="INBOX",
                            source_run_kind="data_updates",
                            source_run_id=str(run.id) if run.id is not None else None,
                            confidence_score=float(inbox_confidence),
                            reason="manual_inbox_import",
                            metadata={
                                "file_name": path.name,
                                "file_hash": file_hash,
                            },
                        )
                    file_rows_ingested += int(added)
                    file_symbols.add(symbol)
                if file_rows_ingested <= 0:
                    file_status = "SKIPPED"
                    file_reason = "no_new_rows"
        except Exception as exc:  # noqa: BLE001
            file_status = "FAILED"
            file_reason = "parse_or_ingest_error"
            file_errors.append({"code": "file_process_failed", "message": str(exc)})

        if file_status == "INGESTED":
            processed_files += 1
            rows_ingested += file_rows_ingested
            symbols_affected.update(file_symbols)
        else:
            skipped_files += 1
            if file_status == "FAILED":
                errors.extend(file_errors)

        warnings.extend(file_warnings)
        session.add(
            DataUpdateFile(
                run_id=int(run.id or 0),
                bundle_id=int(bundle_id),
                timeframe=tf,
                file_path=str(path),
                file_name=path.name,
                file_hash=file_hash,
                status=file_status,
                reason=file_reason,
                rows_ingested=file_rows_ingested,
                symbols_json=sorted(file_symbols),
                warnings_json=file_warnings,
                errors_json=file_errors,
            )
        )

    run.status = "FAILED" if errors else "SUCCEEDED"
    run.processed_files = int(processed_files)
    run.skipped_files = int(skipped_files)
    run.rows_ingested = int(rows_ingested)
    run.symbols_affected_json = sorted(symbols_affected)
    run.warnings_json = warnings
    run.errors_json = errors
    run.ended_at = datetime.now(timezone.utc)
    session.add(run)

    if errors:
        emit_operate_event(
            session,
            severity="ERROR",
            category="DATA",
            message="Data update run completed with errors.",
            details={
                "run_id": run.id,
                "bundle_id": bundle_id,
                "timeframe": tf,
                "processed_files": processed_files,
                "skipped_files": skipped_files,
                "errors": errors[:5],
            },
            correlation_id=correlation_id,
        )
    elif warnings:
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="Data update run completed with warnings.",
            details={
                "run_id": run.id,
                "bundle_id": bundle_id,
                "timeframe": tf,
                "processed_files": processed_files,
                "skipped_files": skipped_files,
                "warnings": warnings[:5],
            },
            correlation_id=correlation_id,
        )
    else:
        emit_operate_event(
            session,
            severity="INFO",
            category="DATA",
            message="Data update run completed successfully.",
            details={
                "run_id": run.id,
                "bundle_id": bundle_id,
                "timeframe": tf,
                "processed_files": processed_files,
                "rows_ingested": rows_ingested,
                "symbols_affected": sorted(symbols_affected),
            },
            correlation_id=correlation_id,
        )

    session.commit()
    session.refresh(run)
    return run


def get_latest_data_update_run(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
) -> DataUpdateRun | None:
    return session.exec(
        select(DataUpdateRun)
        .where(DataUpdateRun.bundle_id == int(bundle_id))
        .where(DataUpdateRun.timeframe == str(timeframe or "1d"))
        .order_by(DataUpdateRun.created_at.desc(), DataUpdateRun.id.desc())
    ).first()


def list_data_update_history(
    session: Session,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = None,
    days: int = 7,
) -> list[DataUpdateRun]:
    since = datetime.now(timezone.utc) - timedelta(days=max(1, int(days)))
    stmt = select(DataUpdateRun).where(DataUpdateRun.created_at >= since)
    if bundle_id is not None:
        stmt = stmt.where(DataUpdateRun.bundle_id == int(bundle_id))
    if timeframe:
        stmt = stmt.where(DataUpdateRun.timeframe == str(timeframe))
    stmt = stmt.order_by(DataUpdateRun.created_at.desc(), DataUpdateRun.id.desc())
    return list(session.exec(stmt).all())
