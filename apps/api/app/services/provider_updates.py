from __future__ import annotations

from datetime import UTC, date, datetime, time as dt_time, timedelta
import hashlib
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DatasetBundle, ProviderUpdateItem, ProviderUpdateRun
from app.providers.base import BaseProvider
from app.providers.factory import build_provider
from app.services.data_store import DataStore
from app.services.fast_mode import clamp_scan_symbols, fast_mode_enabled
from app.services.operate_events import emit_operate_event
from app.services.trading_calendar import is_trading_day, list_trading_days, previous_trading_day

IST_ZONE = ZoneInfo("Asia/Kolkata")


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"1", "true", "yes", "on"}:
            return True
        if token in {"0", "false", "no", "off"}:
            return False
    if value is None:
        return bool(default)
    return bool(value)


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def _infer_instrument_kind(symbol: str) -> str:
    return "STOCK_FUT" if str(symbol).upper().endswith("_FUT") else "EQUITY_CASH"


def _provider_timeframes_from_settings(
    *,
    settings: Settings,
    overrides: dict[str, Any],
) -> set[str]:
    raw = overrides.get("data_updates_provider_timeframes")
    if isinstance(raw, list):
        values = {str(item).strip().lower() for item in raw if str(item).strip()}
        if values:
            return values
    token = overrides.get("data_updates_provider_timeframe_enabled")
    if token is not None:
        values = {
            str(item).strip().lower()
            for item in str(token).split(",")
            if str(item).strip()
        }
        if values:
            return values
    return {
        str(item).strip().lower()
        for item in settings.data_updates_provider_timeframes
        if str(item).strip()
    }


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    base = frame.copy()
    if base.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    for column in ("datetime", "open", "high", "low", "close", "volume"):
        if column not in base.columns:
            raise APIError(
                code="invalid_provider_schema",
                message=f"Provider frame missing column '{column}'.",
            )
    base = base[["datetime", "open", "high", "low", "close", "volume"]]
    base["datetime"] = pd.to_datetime(base["datetime"], utc=True, errors="coerce")
    base = base.dropna(subset=["datetime"])
    if base.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    for column in ("open", "high", "low", "close", "volume"):
        base[column] = pd.to_numeric(base[column], errors="coerce")
    base = base.dropna(subset=["open", "high", "low", "close", "volume"])
    base = base.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
    return base.reset_index(drop=True)


def _merge_ohlcv(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int]:
    old = _normalize_frame(existing)
    new = _normalize_frame(incoming)
    if new.empty:
        return old, 0, 0
    if old.empty:
        merged = (
            new.sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .reset_index(drop=True)
        )
        return merged, int(len(merged)), 0

    old_indexed = old.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    new_indexed = new.set_index("datetime")[["open", "high", "low", "close", "volume"]]
    common_idx = old_indexed.index.intersection(new_indexed.index)
    if len(common_idx) > 0:
        old_common = old_indexed.loc[common_idx].round(10)
        new_common = new_indexed.loc[common_idx].round(10)
        bars_updated = int((old_common.ne(new_common).any(axis=1)).sum())
    else:
        bars_updated = 0
    bars_added = int(len(new_indexed.index.difference(old_indexed.index)))

    merged = (
        pd.concat([old, new], ignore_index=True)
        .sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )
    return merged, bars_added, bars_updated


def _resolve_symbols(
    *,
    provider: BaseProvider,
    store: DataStore,
    session: Session,
    bundle_id: int,
    timeframe: str,
    max_symbols: int,
    seed: int,
) -> tuple[list[str], bool]:
    symbols = provider.list_symbols(bundle_id)
    if not symbols:
        symbols = store.get_bundle_symbols(session, int(bundle_id), timeframe=timeframe)
    deduped = sorted({str(item).upper().strip() for item in symbols if str(item).strip()})
    if not deduped:
        return [], False
    shuffled = sorted(
        deduped,
        key=lambda value: hashlib.sha1(f"{int(seed)}::{value}".encode("utf-8")).hexdigest(),
    )
    truncated = len(shuffled) > max_symbols
    return shuffled[:max_symbols], truncated


def _expected_end_day(*, end_ts: datetime, settings: Settings) -> date:
    day = end_ts.astimezone(IST_ZONE).date()
    segment = str(settings.trading_calendar_segment or "EQUITIES")
    if is_trading_day(day, segment=segment, settings=settings):
        return day
    return previous_trading_day(day, segment=segment, settings=settings)


def _trading_days_tail(
    *,
    end_day: date,
    count: int,
    settings: Settings,
) -> list[date]:
    if count <= 0:
        return []
    segment = str(settings.trading_calendar_segment or "EQUITIES")
    cursor = end_day
    days: list[date] = []
    for _ in range(count * 6):
        if is_trading_day(cursor, segment=segment, settings=settings):
            days.append(cursor)
            if len(days) >= count:
                break
        cursor -= timedelta(days=1)
    return sorted(set(days))


def _compute_fetch_plan(
    *,
    last_bar_before: datetime | None,
    start_ts: datetime,
    end_ts: datetime,
    settings: Settings,
    repair_last_n_days: int,
    backfill_max_days: int,
) -> dict[str, Any]:
    segment = str(settings.trading_calendar_segment or "EQUITIES")
    start_day = start_ts.astimezone(IST_ZONE).date()
    end_day = _expected_end_day(end_ts=end_ts, settings=settings)

    missing_days: list[date]
    if last_bar_before is None:
        missing_days = list_trading_days(
            start_date=start_day,
            end_date=end_day,
            segment=segment,
            settings=settings,
        )
    else:
        last_day = last_bar_before.astimezone(IST_ZONE).date()
        missing_days = list_trading_days(
            start_date=max(start_day, last_day + timedelta(days=1)),
            end_date=end_day,
            segment=segment,
            settings=settings,
        )
    repair_days = _trading_days_tail(
        end_day=end_day,
        count=max(0, repair_last_n_days),
        settings=settings,
    )
    selected_days = sorted(set(missing_days + repair_days))
    truncated = False
    max_days = max(1, int(backfill_max_days))
    if len(selected_days) > max_days:
        selected_days = selected_days[-max_days:]
        truncated = True

    if not selected_days:
        return {
            "missing_days_detected": int(len(missing_days)),
            "repaired_days_used": 0,
            "backfill_truncated": False,
            "selected_days": [],
            "fetch_start": None,
            "fetch_end": None,
        }
    fetch_start = datetime.combine(selected_days[0], dt_time(0, 0), tzinfo=IST_ZONE).astimezone(UTC)
    fetch_end = datetime.combine(selected_days[-1], dt_time(23, 59, 59), tzinfo=IST_ZONE).astimezone(
        UTC
    )
    repaired_days_used = len([day for day in repair_days if day in set(selected_days)])
    return {
        "missing_days_detected": int(len(missing_days)),
        "repaired_days_used": int(repaired_days_used),
        "backfill_truncated": bool(truncated),
        "selected_days": [day.isoformat() for day in selected_days],
        "fetch_start": fetch_start,
        "fetch_end": fetch_end,
    }


def _enforce_4h_guardrails(
    frame: pd.DataFrame,
    *,
    allow_partial: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    normalized = _normalize_frame(frame)
    if normalized.empty:
        return normalized, []
    by_day = normalized.copy()
    by_day["_ist_day"] = pd.to_datetime(by_day["datetime"], utc=True).dt.tz_convert(IST_ZONE).dt.date
    warnings: list[dict[str, Any]] = []
    keep_days: set[date] = set()
    for day, group in by_day.groupby("_ist_day"):
        bars = int(len(group))
        if bars == 2:
            keep_days.add(day)
            continue
        if bars > 2:
            warnings.append(
                {
                    "code": "too_many_4h_bars",
                    "message": f"{day.isoformat()} had {bars} bars; trimming to two.",
                }
            )
            keep_days.add(day)
            continue
        if allow_partial and bars > 0:
            warnings.append(
                {
                    "code": "incomplete_intraday_day_partial_allowed",
                    "message": f"{day.isoformat()} produced {bars} partial 4h_ish bar(s).",
                }
            )
            keep_days.add(day)
        else:
            warnings.append(
                {
                    "code": "incomplete_intraday_day_skipped",
                    "message": f"{day.isoformat()} intraday data incomplete; skipped.",
                }
            )
    filtered = by_day[by_day["_ist_day"].isin(keep_days)].drop(columns=["_ist_day"])
    if filtered.empty:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"]), warnings

    output_rows: list[pd.DataFrame] = []
    filtered["_ist_day"] = pd.to_datetime(filtered["datetime"], utc=True).dt.tz_convert(IST_ZONE).dt.date
    for _, group in filtered.groupby("_ist_day", sort=True):
        ordered = group.sort_values("datetime")
        if len(ordered) > 2:
            ordered = ordered.tail(2)
        output_rows.append(ordered.drop(columns=["_ist_day"]))
    out = pd.concat(output_rows, ignore_index=True).sort_values("datetime").reset_index(drop=True)
    return _normalize_frame(out), warnings


def run_provider_updates(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    bundle_id: int,
    timeframe: str,
    overrides: dict[str, Any] | None = None,
    provider_kind: str | None = None,
    max_symbols_per_run: int | None = None,
    max_calls_per_run: int | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    correlation_id: str | None = None,
    provider: BaseProvider | None = None,
) -> ProviderUpdateRun:
    bundle = session.get(DatasetBundle, int(bundle_id))
    if bundle is None:
        raise APIError(
            code="bundle_not_found",
            message=f"Bundle {bundle_id} not found.",
            status_code=404,
        )

    tf = str(timeframe or "1d").strip().lower()
    state = overrides or {}
    provider_enabled = bool(
        state.get("data_updates_provider_enabled", settings.data_updates_provider_enabled)
    )
    kind = (
        str(
            provider_kind
            or state.get("data_updates_provider_kind", settings.data_updates_provider_kind)
            or "UPSTOX"
        )
        .strip()
        .upper()
    )
    allowed_timeframes = _provider_timeframes_from_settings(settings=settings, overrides=state)
    run = ProviderUpdateRun(
        bundle_id=int(bundle.id) if bundle.id is not None else None,
        timeframe=tf,
        provider_kind=kind,
        status="RUNNING",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not provider_enabled:
        warnings.append(
            {"code": "provider_updates_disabled", "message": "Provider updates are disabled."}
        )
        run.status = "SUCCEEDED"
        run.warnings_json = warnings
        run.ended_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    if tf not in allowed_timeframes:
        warnings.append(
            {
                "code": "provider_timeframe_disabled",
                "message": f"Provider updates disabled for timeframe '{tf}'.",
                "details": {"allowed_timeframes": sorted(allowed_timeframes)},
            }
        )
        run.status = "SUCCEEDED"
        run.warnings_json = warnings
        run.ended_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run

    selected_provider = provider or build_provider(
        kind=kind,
        session=session,
        settings=settings,
        store=store,
    )
    supported = {item.lower() for item in selected_provider.supports_timeframes()}
    if tf not in supported:
        raise APIError(
            code="unsupported_provider_timeframe",
            message=f"{kind} provider does not support timeframe '{tf}'.",
            details={"supported_timeframes": sorted(selected_provider.supports_timeframes())},
        )

    requested_symbols = (
        int(max_symbols_per_run)
        if max_symbols_per_run is not None
        else _safe_int(
            state.get(
                "data_updates_provider_max_symbols_per_run",
                settings.data_updates_provider_max_symbols_per_run,
            ),
            settings.data_updates_provider_max_symbols_per_run,
        )
    )
    symbol_cap = clamp_scan_symbols(settings=settings, requested=max(1, requested_symbols))
    if fast_mode_enabled(settings) and tf == "4h_ish":
        symbol_cap = min(symbol_cap, max(1, int(settings.fast_mode_provider_intraday_max_symbols)))

    call_cap = max(
        1,
        int(
            max_calls_per_run
            if max_calls_per_run is not None
            else _safe_int(
                state.get(
                    "data_updates_provider_max_calls_per_run",
                    settings.data_updates_provider_max_calls_per_run,
                ),
                settings.data_updates_provider_max_calls_per_run,
            )
        ),
    )
    repair_last_n = max(
        0,
        _safe_int(
            state.get(
                "data_updates_provider_repair_last_n_trading_days",
                settings.data_updates_provider_repair_last_n_trading_days,
            ),
            settings.data_updates_provider_repair_last_n_trading_days,
        ),
    )
    backfill_max_days = max(
        1,
        _safe_int(
            state.get(
                "data_updates_provider_backfill_max_days",
                settings.data_updates_provider_backfill_max_days,
            ),
            settings.data_updates_provider_backfill_max_days,
        ),
    )
    allow_partial_4h = _safe_bool(
        state.get(
            "data_updates_provider_allow_partial_4h_ish",
            settings.data_updates_provider_allow_partial_4h_ish,
        ),
        settings.data_updates_provider_allow_partial_4h_ish,
    )
    if fast_mode_enabled(settings) and tf == "4h_ish":
        repair_last_n = min(repair_last_n, max(0, int(settings.fast_mode_provider_intraday_max_days)))
        backfill_max_days = min(
            backfill_max_days,
            max(1, int(settings.fast_mode_provider_intraday_max_days)),
        )

    seed_default = settings.fast_mode_seed if fast_mode_enabled(settings) else 7
    seed = _safe_int(state.get("seed"), seed_default)
    symbols, scan_truncated = _resolve_symbols(
        provider=selected_provider,
        store=store,
        session=session,
        bundle_id=int(bundle_id),
        timeframe=tf,
        max_symbols=symbol_cap,
        seed=seed,
    )
    if scan_truncated:
        warnings.append(
            {
                "code": "provider_symbol_scan_truncated",
                "message": "Provider updates symbol scan was truncated by cap.",
                "details": {
                    "max_symbols_per_run": symbol_cap,
                    "requested_symbols": requested_symbols,
                },
            }
        )
    if not symbols:
        warnings.append(
            {
                "code": "no_symbols_for_provider_update",
                "message": f"No symbols available for bundle {bundle_id}.",
            }
        )
        run.status = "SUCCEEDED"
        run.symbols_attempted = 0
        run.warnings_json = warnings
        run.ended_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run

    start_ts = _parse_iso_datetime(start) if isinstance(start, str) else start
    end_ts = _parse_iso_datetime(end) if isinstance(end, str) else end
    if end_ts is None:
        end_ts = datetime.now(UTC)
    if start_ts is None:
        start_ts = end_ts - timedelta(days=max(30, backfill_max_days + 5))
    start_ts = start_ts.astimezone(UTC) if start_ts.tzinfo else start_ts.replace(tzinfo=UTC)
    end_ts = end_ts.astimezone(UTC) if end_ts.tzinfo else end_ts.replace(tzinfo=UTC)

    selected_provider.reset_counters()
    symbols_succeeded = 0
    symbols_failed = 0
    bars_added_total = 0
    bars_updated_total = 0
    api_calls_consumed = 0
    missing_days_total = 0
    repaired_days_total = 0
    backfill_truncated_any = False
    started_at = datetime.now(UTC)

    missing_map_symbols = selected_provider.missing_mapped_symbols(symbols)
    if missing_map_symbols:
        warnings.append(
            {
                "code": "missing_instrument_map",
                "message": f"Missing provider mapping for {len(missing_map_symbols)} symbol(s).",
                "details": {"symbols": sorted(missing_map_symbols)[:50]},
            }
        )
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="provider_updates_missing_instrument_map",
            details={
                "run_id": run.id,
                "provider_kind": kind,
                "bundle_id": bundle_id,
                "timeframe": tf,
                "missing_map_count": len(missing_map_symbols),
                "symbols": sorted(missing_map_symbols)[:50],
            },
            correlation_id=correlation_id,
        )
        for symbol in sorted(missing_map_symbols):
            session.add(
                ProviderUpdateItem(
                    run_id=int(run.id or 0),
                    bundle_id=int(bundle_id),
                    timeframe=tf,
                    provider_kind=kind,
                    symbol=symbol,
                    status="SKIPPED",
                    warnings_json=[
                        {
                            "code": "missing_instrument_map",
                            "message": "Symbol skipped due to missing instrument mapping.",
                        }
                    ],
                )
            )

    for symbol in symbols:
        symbol_up = str(symbol).upper()
        if symbol_up in missing_map_symbols:
            continue
        if api_calls_consumed >= call_cap:
            warnings.append(
                {
                    "code": "provider_call_cap_reached",
                    "message": "Provider update call cap reached; remaining symbols skipped.",
                    "details": {"max_calls_per_run": call_cap},
                }
            )
            break

        existing = store.load_ohlcv(symbol=symbol_up, timeframe=tf)
        last_bar_before: datetime | None = None
        if not existing.empty:
            ts_before = pd.to_datetime(existing["datetime"], utc=True, errors="coerce").max()
            if pd.notna(ts_before):
                last_bar_before = ts_before.to_pydatetime()
        plan = _compute_fetch_plan(
            last_bar_before=last_bar_before,
            start_ts=start_ts,
            end_ts=end_ts,
            settings=settings,
            repair_last_n_days=repair_last_n,
            backfill_max_days=backfill_max_days,
        )
        missing_days_total += int(plan["missing_days_detected"])
        repaired_days_total += int(plan["repaired_days_used"])
        backfill_truncated_any = backfill_truncated_any or bool(plan["backfill_truncated"])

        symbol_errors: list[dict[str, Any]] = []
        symbol_warnings: list[dict[str, Any]] = []
        status = "SUCCEEDED"
        bars_added = 0
        bars_updated = 0
        last_bar_after = last_bar_before
        item_start = datetime.now(UTC)
        if plan["fetch_start"] is None or plan["fetch_end"] is None:
            status = "SKIPPED"
            symbol_warnings.append(
                {"code": "no_fetch_range_required", "message": "No missing/repair days for symbol."}
            )
        elif bool(plan["backfill_truncated"]):
            symbol_warnings.append(
                {
                    "code": "backfill_truncated",
                    "message": "Backfill window truncated by max_backfill_days.",
                    "details": {"selected_days": plan.get("selected_days", [])[:20]},
                }
            )
        calls_before = selected_provider.api_calls_made
        if status != "SKIPPED":
            try:
                fetched = selected_provider.fetch_ohlc(
                    [symbol_up],
                    timeframe=tf,
                    start=plan["fetch_start"],
                    end=plan["fetch_end"],
                )
                incoming = fetched.get(symbol_up, pd.DataFrame())
                if tf == "4h_ish":
                    incoming, resample_warnings = _enforce_4h_guardrails(
                        incoming,
                        allow_partial=allow_partial_4h,
                    )
                    symbol_warnings.extend(resample_warnings)
                merged, bars_added, bars_updated = _merge_ohlcv(existing, incoming)
                if bars_added > 0 or bars_updated > 0:
                    instrument = store.find_instrument(session, symbol=symbol_up)
                    kind_for_save = (
                        instrument.kind if instrument is not None else _infer_instrument_kind(symbol_up)
                    )
                    lot_size = instrument.lot_size if instrument is not None else 1
                    underlying = instrument.underlying if instrument is not None else None
                    tick_size = instrument.tick_size if instrument is not None else 0.05
                    store.save_ohlcv(
                        session=session,
                        symbol=symbol_up,
                        timeframe=tf,
                        frame=merged,
                        provider=f"{bundle.provider}-provider-{kind.lower()}",
                        checksum=None,
                        instrument_kind=kind_for_save,
                        underlying=underlying,
                        lot_size=lot_size,
                        tick_size=float(tick_size),
                        bundle_id=int(bundle_id),
                    )
                    after = store.load_ohlcv(symbol=symbol_up, timeframe=tf)
                    if not after.empty:
                        ts_after = pd.to_datetime(after["datetime"], utc=True, errors="coerce").max()
                        if pd.notna(ts_after):
                            last_bar_after = ts_after.to_pydatetime()
                else:
                    status = "SKIPPED"
                    symbol_warnings.append(
                        {
                            "code": "no_new_rows",
                            "message": "Provider fetch returned no new or corrected bars.",
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                status = "FAILED"
                symbol_errors.append({"code": "provider_fetch_failed", "message": str(exc)})
                errors.extend(symbol_errors)

        calls_delta = max(0, selected_provider.api_calls_made - calls_before)
        api_calls_consumed += calls_delta
        api_calls_consumed = min(api_calls_consumed, call_cap)
        bars_added_total += int(max(0, bars_added))
        bars_updated_total += int(max(0, bars_updated))
        if status == "FAILED":
            symbols_failed += 1
        else:
            symbols_succeeded += 1
        session.add(
            ProviderUpdateItem(
                run_id=int(run.id or 0),
                bundle_id=int(bundle_id),
                timeframe=tf,
                provider_kind=kind,
                symbol=symbol_up,
                status=status,
                bars_added=int(max(0, bars_added)),
                bars_updated=int(max(0, bars_updated)),
                api_calls=int(calls_delta),
                start_ts=item_start,
                end_ts=datetime.now(UTC),
                last_bar_before=last_bar_before,
                last_bar_after=last_bar_after,
                warnings_json=symbol_warnings,
                errors_json=symbol_errors,
            )
        )
        if symbol_warnings:
            warnings.extend(symbol_warnings)

    run.status = "FAILED" if (symbols_succeeded == 0 and symbols_failed > 0) else "SUCCEEDED"
    run.symbols_attempted = int(len(symbols))
    run.symbols_succeeded = int(symbols_succeeded)
    run.symbols_failed = int(symbols_failed)
    run.bars_added = int(bars_added_total)
    run.repaired_days_used = int(repaired_days_total)
    run.missing_days_detected = int(missing_days_total)
    run.backfill_truncated = bool(backfill_truncated_any)
    run.api_calls = int(api_calls_consumed)
    run.duration_seconds = round((datetime.now(UTC) - started_at).total_seconds(), 3)
    run.warnings_json = warnings
    run.errors_json = errors
    run.ended_at = datetime.now(UTC)
    session.add(run)

    details = {
        "run_id": run.id,
        "provider_kind": kind,
        "bundle_id": bundle_id,
        "timeframe": tf,
        "symbols_attempted": run.symbols_attempted,
        "symbols_succeeded": run.symbols_succeeded,
        "symbols_failed": run.symbols_failed,
        "bars_added": run.bars_added,
        "bars_updated": bars_updated_total,
        "missing_days_detected": run.missing_days_detected,
        "repaired_days_used": run.repaired_days_used,
        "backfill_truncated": run.backfill_truncated,
        "api_calls": run.api_calls,
    }
    if run.status == "FAILED":
        emit_operate_event(
            session,
            severity="ERROR",
            category="DATA",
            message="Provider update run failed.",
            details=details,
            correlation_id=correlation_id,
        )
    elif warnings:
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="Provider update run completed with warnings.",
            details={**details, "warnings_count": len(warnings)},
            correlation_id=correlation_id,
        )
    else:
        emit_operate_event(
            session,
            severity="INFO",
            category="DATA",
            message="Provider update run completed successfully.",
            details=details,
            correlation_id=correlation_id,
        )

    session.commit()
    session.refresh(run)
    return run


def get_latest_provider_update_run(
    session: Session,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = None,
) -> ProviderUpdateRun | None:
    stmt = select(ProviderUpdateRun).order_by(
        ProviderUpdateRun.created_at.desc(),
        ProviderUpdateRun.id.desc(),
    )
    if bundle_id is not None:
        stmt = stmt.where(ProviderUpdateRun.bundle_id == int(bundle_id))
    if timeframe:
        stmt = stmt.where(ProviderUpdateRun.timeframe == str(timeframe))
    return session.exec(stmt).first()


def list_provider_update_history(
    session: Session,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = None,
    days: int = 7,
) -> list[ProviderUpdateRun]:
    since = datetime.now(UTC) - timedelta(days=max(1, int(days)))
    stmt = select(ProviderUpdateRun).where(ProviderUpdateRun.created_at >= since)
    if bundle_id is not None:
        stmt = stmt.where(ProviderUpdateRun.bundle_id == int(bundle_id))
    if timeframe:
        stmt = stmt.where(ProviderUpdateRun.timeframe == str(timeframe))
    stmt = stmt.order_by(ProviderUpdateRun.created_at.desc(), ProviderUpdateRun.id.desc())
    return list(session.exec(stmt).all())
