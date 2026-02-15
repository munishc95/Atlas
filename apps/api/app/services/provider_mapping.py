from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DatasetBundle, InstrumentMap, MappingImportRun
from app.services.data_store import DataStore
from app.services.operate_events import emit_operate_event

_PROVIDER = "UPSTOX"
_MODE_UPSERT = "UPSERT"
_MODE_REPLACE = "REPLACE"


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _resolve_mapping_path(*, settings: Settings, raw: str) -> Path:
    candidate = Path(str(raw or "").strip())
    if not str(candidate):
        candidate = Path(settings.data_inbox_root) / "_metadata" / "upstox_instruments.csv"
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _column_lookup(columns: list[str]) -> dict[str, str]:
    return {str(value).strip().lower(): str(value) for value in columns}


def _pick_column(lookup: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def _load_mapping(path: Path) -> dict[str, str]:
    suffix = path.suffix.lower()
    payload: dict[str, str] = {}
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key, value in raw.items():
                symbol = str(key).strip().upper()
                instrument_key = str(value).strip()
                if symbol and instrument_key:
                    payload[symbol] = instrument_key
            return payload
        if isinstance(raw, list):
            rows = raw
        else:
            raise APIError(
                code="invalid_mapping_json",
                message="Mapping JSON must be an object or list.",
            )
        frame = pd.DataFrame(rows)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise APIError(
            code="unsupported_mapping_file_type",
            message="Mapping file must be .csv or .json",
            details={"path": str(path)},
        )

    if frame.empty:
        raise APIError(code="empty_mapping_file", message="Mapping file has no rows.")

    lookup = _column_lookup(list(frame.columns))
    symbol_col = _pick_column(
        lookup, ("symbol", "ticker", "tradingsymbol", "underlying", "atlas_symbol")
    )
    key_col = _pick_column(
        lookup, ("instrument_key", "instrument", "instrument_token", "upstox_key", "key")
    )
    if symbol_col is None or key_col is None:
        raise APIError(
            code="invalid_mapping_schema",
            message="Mapping file must include symbol and instrument_key columns.",
            details={"columns": list(frame.columns)},
        )

    for _, row in frame.iterrows():
        symbol = str(row.get(symbol_col, "")).strip().upper()
        instrument_key = str(row.get(key_col, "")).strip()
        if symbol and instrument_key:
            payload[symbol] = instrument_key

    if not payload:
        raise APIError(code="empty_mapping_file", message="No valid symbol mappings found.")
    return payload


def _resolve_target_bundle(
    *,
    session: Session,
    store: DataStore,
    bundle_id: int | None,
    timeframe: str = "1d",
) -> tuple[int | None, list[str]]:
    if bundle_id is not None and int(bundle_id) > 0:
        bundle = session.get(DatasetBundle, int(bundle_id))
        if bundle is None:
            raise APIError(
                code="bundle_not_found",
                message=f"Bundle {bundle_id} not found.",
                status_code=404,
            )
        return int(bundle.id) if bundle.id is not None else None, store.get_bundle_symbols(
            session, int(bundle.id), timeframe=timeframe
        )
    latest_bundle = session.exec(select(DatasetBundle).order_by(DatasetBundle.created_at.desc())).first()
    if latest_bundle is None or latest_bundle.id is None:
        return None, []
    return int(latest_bundle.id), store.get_bundle_symbols(
        session, int(latest_bundle.id), timeframe=timeframe
    )


def get_upstox_mapping_status(
    *,
    session: Session,
    store: DataStore,
    bundle_id: int | None = None,
    timeframe: str = "1d",
    sample_limit: int = 20,
) -> dict[str, Any]:
    target_bundle_id, symbols = _resolve_target_bundle(
        session=session,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
    )
    universe = sorted({str(symbol).upper() for symbol in symbols if str(symbol).strip()})
    mapped: set[str] = set()
    if universe:
        rows = session.exec(
            select(InstrumentMap)
            .where(InstrumentMap.provider == _PROVIDER)
            .where(InstrumentMap.symbol.in_(universe))
        ).all()
        mapped = {str(row.symbol).upper() for row in rows if str(row.instrument_key or "").strip()}
    missing = [symbol for symbol in universe if symbol not in mapped]
    last_import = session.exec(
        select(MappingImportRun)
        .where(MappingImportRun.provider == _PROVIDER)
        .order_by(MappingImportRun.created_at.desc(), MappingImportRun.id.desc())
    ).first()
    return {
        "provider": _PROVIDER,
        "bundle_id": target_bundle_id,
        "timeframe": timeframe,
        "mapped_count": int(len(mapped)),
        "missing_count": int(len(missing)),
        "total_symbols": int(len(universe)),
        "sample_missing_symbols": missing[: max(1, _safe_int(sample_limit, 20))],
        "last_import_at": (
            last_import.created_at.isoformat()
            if last_import is not None and isinstance(last_import.created_at, datetime)
            else None
        ),
        "last_import_id": int(last_import.id) if last_import is not None and last_import.id else None,
    }


def list_upstox_missing_symbols(
    *,
    session: Session,
    store: DataStore,
    bundle_id: int | None = None,
    timeframe: str = "1d",
    limit: int = 100,
) -> list[str]:
    status = get_upstox_mapping_status(
        session=session,
        store=store,
        bundle_id=bundle_id,
        timeframe=timeframe,
        sample_limit=max(1, int(limit)),
    )
    return list(status.get("sample_missing_symbols", []))


def import_upstox_mapping_file(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    path: str,
    mode: str = _MODE_UPSERT,
    bundle_id: int | None = None,
    correlation_id: str | None = None,
) -> MappingImportRun:
    mode_token = str(mode or _MODE_UPSERT).strip().upper()
    if mode_token not in {_MODE_UPSERT, _MODE_REPLACE}:
        raise APIError(
            code="invalid_mapping_import_mode",
            message="mode must be UPSERT or REPLACE",
            details={"mode": mode},
        )
    resolved_path = _resolve_mapping_path(settings=settings, raw=path)
    if not resolved_path.exists() or not resolved_path.is_file():
        raise APIError(
            code="mapping_file_not_found",
            message=f"Mapping file not found: {resolved_path}",
            status_code=404,
        )

    started = time.perf_counter()
    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    inserted_count = 0
    updated_count = 0
    removed_count = 0
    mapping = _load_mapping(resolved_path)

    run = MappingImportRun(
        provider=_PROVIDER,
        mode=mode_token,
        source_path=str(resolved_path),
        file_hash=_hash_file(resolved_path),
        status="RUNNING",
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    now = datetime.now(UTC)
    existing_rows = session.exec(
        select(InstrumentMap).where(InstrumentMap.provider == _PROVIDER)
    ).all()
    existing_by_symbol = {str(row.symbol).upper(): row for row in existing_rows}

    if mode_token == _MODE_REPLACE:
        to_delete = [row for symbol, row in existing_by_symbol.items() if symbol not in mapping]
        for row in to_delete:
            session.delete(row)
            removed_count += 1

    for symbol, instrument_key in sorted(mapping.items()):
        current = existing_by_symbol.get(symbol)
        if current is None:
            session.add(
                InstrumentMap(
                    provider=_PROVIDER,
                    symbol=symbol,
                    instrument_key=instrument_key,
                    last_refreshed=now,
                )
            )
            inserted_count += 1
            continue
        if str(current.instrument_key).strip() != instrument_key:
            current.instrument_key = instrument_key
            current.last_refreshed = now
            session.add(current)
            updated_count += 1

    status_payload = get_upstox_mapping_status(
        session=session,
        store=store,
        bundle_id=bundle_id,
        timeframe="1d",
        sample_limit=20,
    )
    run.status = "FAILED" if errors else "SUCCEEDED"
    run.mapped_count = int(status_payload.get("mapped_count", 0))
    run.missing_count = int(status_payload.get("missing_count", 0))
    run.inserted_count = int(inserted_count)
    run.updated_count = int(updated_count)
    run.removed_count = int(removed_count)
    run.warnings_json = warnings
    run.errors_json = errors
    run.duration_seconds = round(time.perf_counter() - started, 3)
    session.add(run)

    severity = "INFO" if run.missing_count == 0 and not warnings else "WARN"
    emit_operate_event(
        session,
        severity=severity,
        category="DATA",
        message="upstox_mapping_import_completed",
        details={
            "run_id": run.id,
            "mode": mode_token,
            "path": str(resolved_path),
            "mapped_count": run.mapped_count,
            "missing_count": run.missing_count,
            "inserted_count": run.inserted_count,
            "updated_count": run.updated_count,
            "removed_count": run.removed_count,
            "sample_missing_symbols": status_payload.get("sample_missing_symbols", []),
        },
        correlation_id=correlation_id,
    )
    session.commit()
    session.refresh(run)
    return run

