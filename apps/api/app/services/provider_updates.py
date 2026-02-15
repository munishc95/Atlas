from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
from typing import Any

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


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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


def _provider_timeframe_allowed(*, timeframe: str, allow_token: str) -> bool:
    allowed = {
        str(item).strip().lower()
        for item in str(allow_token or "1d").split(",")
        if str(item).strip()
    }
    if not allowed:
        return False
    return str(timeframe).strip().lower() in allowed


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


def _merge_ohlcv(existing: pd.DataFrame, incoming: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    old = _normalize_frame(existing)
    old_count = int(len(old))
    new = _normalize_frame(incoming)
    if new.empty:
        return old, 0
    if old.empty:
        merged = new.sort_values("datetime").drop_duplicates(subset=["datetime"], keep="last")
        merged = merged.reset_index(drop=True)
    else:
        merged = (
            pd.concat([old, new], ignore_index=True)
            .sort_values("datetime")
            .drop_duplicates(subset=["datetime"], keep="last")
            .reset_index(drop=True)
        )
    added = max(0, int(len(merged) - old_count))
    return merged, added


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
            code="bundle_not_found", message=f"Bundle {bundle_id} not found.", status_code=404
        )

    tf = str(timeframe or "1d").strip()
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
    timeframe_allow = str(
        state.get(
            "data_updates_provider_timeframe_enabled",
            settings.data_updates_provider_timeframe_enabled,
        )
    )
    now_utc = datetime.now(UTC)

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
        run.ended_at = now_utc
        session.add(run)
        session.commit()
        session.refresh(run)
        return run

    if not _provider_timeframe_allowed(timeframe=tf, allow_token=timeframe_allow):
        warnings.append(
            {
                "code": "provider_timeframe_disabled",
                "message": f"Provider updates disabled for timeframe '{tf}'.",
                "details": {"allowed": timeframe_allow},
            }
        )
        run.status = "SUCCEEDED"
        run.warnings_json = warnings
        run.ended_at = now_utc
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
    if tf.lower() not in {value.lower() for value in selected_provider.supports_timeframes()}:
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
    seed = _safe_int(
        state.get("seed"), settings.fast_mode_seed if fast_mode_enabled(settings) else 7
    )
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
        start_ts = end_ts - timedelta(days=30)
    if start_ts.tzinfo is None:
        start_ts = start_ts.replace(tzinfo=UTC)
    else:
        start_ts = start_ts.astimezone(UTC)
    if end_ts.tzinfo is None:
        end_ts = end_ts.replace(tzinfo=UTC)
    else:
        end_ts = end_ts.astimezone(UTC)

    selected_provider.reset_counters()
    symbols_succeeded = 0
    symbols_failed = 0
    bars_added_total = 0
    api_calls_consumed = 0
    started_at = datetime.now(UTC)

    for symbol in symbols:
        if api_calls_consumed >= call_cap:
            warnings.append(
                {
                    "code": "provider_call_cap_reached",
                    "message": "Provider update call cap reached; remaining symbols skipped.",
                    "details": {"max_calls_per_run": call_cap},
                }
            )
            break

        existing = store.load_ohlcv(symbol=symbol, timeframe=tf)
        last_bar_before = None
        if not existing.empty:
            ts_before = pd.to_datetime(existing["datetime"], utc=True, errors="coerce").max()
            if pd.notna(ts_before):
                last_bar_before = ts_before.to_pydatetime()
        symbol_start = start_ts
        if last_bar_before is not None:
            symbol_start = max(start_ts, last_bar_before - timedelta(days=2))

        calls_before = selected_provider.api_calls_made
        symbol_errors: list[dict[str, Any]] = []
        symbol_warnings: list[dict[str, Any]] = []
        status = "SUCCEEDED"
        bars_added = 0
        last_bar_after = last_bar_before
        item_start = datetime.now(UTC)
        try:
            fetched = selected_provider.fetch_ohlc(
                [symbol],
                timeframe=tf,
                start=symbol_start,
                end=end_ts,
            )
            incoming = fetched.get(symbol, pd.DataFrame())
            merged, bars_added = _merge_ohlcv(existing, incoming)
            if bars_added > 0:
                instrument = store.find_instrument(session, symbol=symbol)
                kind_for_save = (
                    instrument.kind if instrument is not None else _infer_instrument_kind(symbol)
                )
                lot_size = instrument.lot_size if instrument is not None else 1
                underlying = instrument.underlying if instrument is not None else None
                tick_size = instrument.tick_size if instrument is not None else 0.05
                store.save_ohlcv(
                    session=session,
                    symbol=symbol,
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
                after = store.load_ohlcv(symbol=symbol, timeframe=tf)
                if not after.empty:
                    ts_after = pd.to_datetime(after["datetime"], utc=True, errors="coerce").max()
                    if pd.notna(ts_after):
                        last_bar_after = ts_after.to_pydatetime()
            else:
                status = "SKIPPED"
                symbol_warnings.append(
                    {"code": "no_new_rows", "message": "Provider fetch returned no new bars."}
                )
        except Exception as exc:  # noqa: BLE001
            status = "FAILED"
            symbol_errors.append({"code": "provider_fetch_failed", "message": str(exc)})
            errors.extend(symbol_errors)

        calls_delta = max(0, selected_provider.api_calls_made - calls_before)
        api_calls_consumed += calls_delta
        api_calls_consumed = min(api_calls_consumed, call_cap)
        bars_added_total += int(max(0, bars_added))
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
                symbol=symbol,
                status=status,
                bars_added=int(max(0, bars_added)),
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
    run.api_calls = int(api_calls_consumed)
    run.duration_seconds = round((datetime.now(UTC) - started_at).total_seconds(), 3)
    run.warnings_json = warnings
    run.errors_json = errors
    run.ended_at = datetime.now(UTC)
    session.add(run)

    if run.status == "FAILED":
        emit_operate_event(
            session,
            severity="ERROR",
            category="DATA",
            message="Provider update run failed.",
            details={
                "run_id": run.id,
                "provider_kind": kind,
                "bundle_id": bundle_id,
                "timeframe": tf,
                "symbols_attempted": run.symbols_attempted,
                "symbols_failed": run.symbols_failed,
                "api_calls": run.api_calls,
            },
            correlation_id=correlation_id,
        )
    elif warnings:
        emit_operate_event(
            session,
            severity="WARN",
            category="DATA",
            message="Provider update run completed with warnings.",
            details={
                "run_id": run.id,
                "provider_kind": kind,
                "bundle_id": bundle_id,
                "timeframe": tf,
                "symbols_attempted": run.symbols_attempted,
                "symbols_succeeded": run.symbols_succeeded,
                "bars_added": run.bars_added,
                "api_calls": run.api_calls,
                "warnings_count": len(warnings),
            },
            correlation_id=correlation_id,
        )
    else:
        emit_operate_event(
            session,
            severity="INFO",
            category="DATA",
            message="Provider update run completed successfully.",
            details={
                "run_id": run.id,
                "provider_kind": kind,
                "bundle_id": bundle_id,
                "timeframe": tf,
                "symbols_attempted": run.symbols_attempted,
                "symbols_succeeded": run.symbols_succeeded,
                "bars_added": run.bars_added,
                "api_calls": run.api_calls,
            },
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
