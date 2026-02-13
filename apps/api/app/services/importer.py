from __future__ import annotations

import hashlib
import io
from typing import Any

import pandas as pd
from fastapi import UploadFile
from sqlmodel import Session

from app.core.exceptions import APIError
from app.services.data_store import DataStore
from app.utils.resample import parse_bar_windows, resample_intraday_to_session_bars


REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def _apply_mapping(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    renamed = frame.rename(columns={v: k for k, v in mapping.items() if v in frame.columns})
    missing = [col for col in REQUIRED_COLUMNS if col not in renamed.columns]
    if missing:
        raise APIError(
            code="invalid_mapping",
            message="Mapped columns are missing required OHLCV fields",
            details={"missing": missing},
        )
    return renamed[REQUIRED_COLUMNS]


def _validate_numeric(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=REQUIRED_COLUMNS)
    if out.empty:
        raise APIError(code="empty_data", message="No valid rows after mapping/validation")
    return out.sort_values("datetime")


def _checksum(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def import_ohlcv_upload(
    session: Session,
    store: DataStore,
    upload: UploadFile,
    symbol: str,
    timeframe: str,
    mapping: dict[str, str] | None,
    provider: str,
    bar_windows: str,
    instrument_kind: str = "EQUITY_CASH",
    underlying: str | None = None,
    lot_size: int | None = None,
    tick_size: float = 0.05,
    bundle_id: int | None = None,
    bundle_name: str | None = None,
    bundle_description: str | None = None,
) -> dict[str, Any]:
    raw = upload.file.read()
    return import_ohlcv_bytes(
        session=session,
        store=store,
        raw=raw,
        filename=upload.filename,
        symbol=symbol,
        timeframe=timeframe,
        mapping=mapping,
        provider=provider,
        bar_windows=bar_windows,
        instrument_kind=instrument_kind,
        underlying=underlying,
        lot_size=lot_size,
        tick_size=tick_size,
        bundle_id=bundle_id,
        bundle_name=bundle_name,
        bundle_description=bundle_description,
    )


def import_ohlcv_bytes(
    session: Session,
    store: DataStore,
    raw: bytes,
    filename: str | None,
    symbol: str,
    timeframe: str,
    mapping: dict[str, str] | None,
    provider: str,
    bar_windows: str,
    instrument_kind: str = "EQUITY_CASH",
    underlying: str | None = None,
    lot_size: int | None = None,
    tick_size: float = 0.05,
    bundle_id: int | None = None,
    bundle_name: str | None = None,
    bundle_description: str | None = None,
) -> dict[str, Any]:
    if not raw:
        raise APIError(code="empty_file", message="Uploaded file is empty")

    if filename and filename.lower().endswith(".parquet"):
        frame = pd.read_parquet(io.BytesIO(raw))
    else:
        frame = pd.read_csv(io.BytesIO(raw))

    frame = _apply_mapping(frame, mapping or {}) if mapping else frame
    frame = _validate_numeric(frame)

    if timeframe == "4h_ish_resampled":
        windows = parse_bar_windows(bar_windows)
        frame = resample_intraday_to_session_bars(frame, windows)
        timeframe = "4h_ish"
    instrument_kind = str(instrument_kind or "EQUITY_CASH").upper()
    if instrument_kind in {"STOCK_FUT", "INDEX_FUT"} and (lot_size is None or int(lot_size) <= 0):
        raise APIError(
            code="invalid_instrument_metadata",
            message="Futures import requires a positive lot_size.",
        )

    dataset = store.save_ohlcv(
        session=session,
        symbol=symbol,
        timeframe=timeframe,
        frame=frame,
        provider=provider,
        checksum=_checksum(raw),
        instrument_kind=instrument_kind,
        underlying=underlying,
        lot_size=lot_size,
        tick_size=tick_size,
        bundle_id=bundle_id,
        bundle_name=bundle_name,
        bundle_description=bundle_description,
    )

    return {
        "dataset_id": dataset.id,
        "bundle_id": dataset.bundle_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(len(frame)),
        "start": frame["datetime"].min().isoformat(),
        "end": frame["datetime"].max().isoformat(),
    }
