from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from pathlib import Path
from typing import Any
import re

import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import (
    BundleMembershipHistory,
    CorporateAction,
    DailyConfidenceAggregate,
    DataProvenance,
    DatasetBundle,
    TrainDataset,
    TrainDatasetRun,
)
from app.engine.indicators import atr, ema, rsi
from app.services.fast_mode import fast_mode_enabled


def _parse_iso_date(value: str | dt_date, *, field: str) -> dt_date:
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
            message=f"{field} must be YYYY-MM-DD.",
            details={field: token},
        ) from exc


def _normalize_token(value: str | None, *, allowed: set[str], default: str) -> str:
    token = str(value or default).strip().upper()
    if token in allowed:
        return token
    return default


def _slugify(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip()).strip("_").lower()
    return token or "train_dataset"


def _feature_enabled(config: dict[str, Any], key: str, *, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _dominant_provider(mix_counts: dict[str, Any]) -> str | None:
    if not isinstance(mix_counts, dict) or not mix_counts:
        return None
    items: list[tuple[str, int]] = []
    for key, value in mix_counts.items():
        try:
            items.append((str(key), int(value)))
        except (TypeError, ValueError):
            continue
    if not items:
        return None
    items.sort(key=lambda item: (-item[1], item[0]))
    return items[0][0]


def serialize_train_dataset(row: TrainDataset) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "name": str(row.name),
        "bundle_id": int(row.bundle_id),
        "timeframe": str(row.timeframe),
        "start_date": row.start_date.isoformat(),
        "end_date": row.end_date.isoformat(),
        "adjustment_mode": str(row.adjustment_mode),
        "membership_mode": str(row.membership_mode),
        "feature_config_json": dict(row.feature_config_json or {}),
        "label_config_json": dict(row.label_config_json or {}),
        "status": str(row.status),
        "row_count": int(row.row_count or 0),
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def serialize_train_dataset_run(row: TrainDatasetRun) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "dataset_id": int(row.dataset_id),
        "status": str(row.status),
        "started_at": row.started_at.isoformat() if row.started_at is not None else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at is not None else None,
        "row_count": int(row.row_count or 0),
        "output_path": row.output_path,
        "warnings_json": [dict(item) for item in (row.warnings_json or []) if isinstance(item, dict)],
        "errors_json": [dict(item) for item in (row.errors_json or []) if isinstance(item, dict)],
        "created_at": row.created_at.isoformat(),
        "updated_at": row.updated_at.isoformat(),
    }


def list_train_datasets(session: Session, *, limit: int = 100) -> list[TrainDataset]:
    stmt = (
        select(TrainDataset)
        .order_by(TrainDataset.updated_at.desc(), TrainDataset.id.desc())
        .limit(max(1, min(int(limit), 500)))
    )
    return list(session.exec(stmt).all())


def get_train_dataset(session: Session, dataset_id: int) -> TrainDataset | None:
    return session.get(TrainDataset, int(dataset_id))


def get_latest_train_dataset_run(session: Session, *, dataset_id: int) -> TrainDatasetRun | None:
    return session.exec(
        select(TrainDatasetRun)
        .where(TrainDatasetRun.dataset_id == int(dataset_id))
        .order_by(TrainDatasetRun.started_at.desc(), TrainDatasetRun.id.desc())
        .limit(1)
    ).first()


def create_train_dataset(
    session: Session,
    *,
    payload: dict[str, Any],
) -> TrainDataset:
    name = str(payload.get("name") or "").strip()
    if not name:
        raise APIError(code="invalid_payload", message="Dataset name is required.")
    existing = session.exec(select(TrainDataset).where(TrainDataset.name == name)).first()
    if existing is not None:
        raise APIError(
            code="duplicate_dataset_name",
            message="Train dataset name already exists.",
            details={"name": name},
        )
    bundle_id = int(payload.get("bundle_id") or 0)
    bundle = session.get(DatasetBundle, bundle_id)
    if bundle is None:
        raise APIError(code="not_found", message="Bundle not found.", status_code=404)
    start_date = _parse_iso_date(payload.get("start_date"), field="start_date")
    end_date = _parse_iso_date(payload.get("end_date"), field="end_date")
    if end_date < start_date:
        raise APIError(
            code="invalid_date_range",
            message="end_date must be on or after start_date.",
        )
    row = TrainDataset(
        name=name,
        bundle_id=bundle_id,
        timeframe=str(payload.get("timeframe") or "1d").strip() or "1d",
        start_date=start_date,
        end_date=end_date,
        adjustment_mode=_normalize_token(
            payload.get("adjustment_mode"),
            allowed={"RAW", "ADJUSTED"},
            default="RAW",
        ),
        membership_mode=_normalize_token(
            payload.get("membership_mode"),
            allowed={"CURRENT", "HISTORICAL"},
            default="CURRENT",
        ),
        feature_config_json=dict(payload.get("feature_config_json") or {}),
        label_config_json=dict(payload.get("label_config_json") or {}),
        status="CREATED",
        row_count=0,
        updated_at=datetime.now(UTC),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def _membership_intervals(
    session: Session,
    *,
    bundle_id: int,
) -> dict[str, list[tuple[dt_date, dt_date | None]]]:
    rows = list(
        session.exec(
            select(BundleMembershipHistory)
            .where(BundleMembershipHistory.bundle_id == int(bundle_id))
            .order_by(
                BundleMembershipHistory.symbol.asc(),
                BundleMembershipHistory.effective_from.asc(),
                BundleMembershipHistory.id.asc(),
            )
        ).all()
    )
    out: dict[str, list[tuple[dt_date, dt_date | None]]] = {}
    for row in rows:
        out.setdefault(str(row.symbol).upper(), []).append(
            (row.effective_from, row.effective_to)
        )
    return out


def _symbol_has_actions(session: Session, *, symbol: str) -> bool:
    row = session.exec(
        select(CorporateAction.id)
        .where(CorporateAction.symbol == str(symbol).upper())
        .limit(1)
    ).first()
    return row is not None


def _apply_membership_window(
    frame: pd.DataFrame,
    *,
    intervals: list[tuple[dt_date, dt_date | None]] | None,
) -> pd.DataFrame:
    if frame.empty or not intervals:
        return frame
    keep_mask = []
    dates = frame["trading_date"].tolist()
    for trading_day in dates:
        active = False
        for start_day, end_day in intervals:
            if trading_day < start_day:
                continue
            if end_day is not None and trading_day > end_day:
                continue
            active = True
            break
        keep_mask.append(active)
    return frame[pd.Series(keep_mask, index=frame.index)].reset_index(drop=True)


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy().sort_values("datetime").reset_index(drop=True)
    out["return_1d"] = out["close"].pct_change(1)
    out["return_5d"] = out["close"].pct_change(5)
    out["atr_14"] = atr(out, period=14)
    out["rsi_14"] = rsi(out["close"], period=14)
    out["ema_20"] = ema(out["close"], period=20)
    out["future_return_5d"] = out["close"].shift(-5) / out["close"] - 1.0
    return out


def _collect_source_maps(
    session: Session,
    *,
    bundle_id: int,
    timeframe: str,
    start_date: dt_date,
    end_date: dt_date,
) -> tuple[dict[tuple[str, dt_date], tuple[str | None, float | None]], dict[dt_date, tuple[str | None, float | None]]]:
    provenance_rows = list(
        session.exec(
            select(DataProvenance)
            .where(DataProvenance.bundle_id == int(bundle_id))
            .where(DataProvenance.timeframe == str(timeframe))
            .where(DataProvenance.bar_date >= start_date)
            .where(DataProvenance.bar_date <= end_date)
        ).all()
    )
    by_symbol_day: dict[tuple[str, dt_date], tuple[str | None, float | None]] = {}
    for row in provenance_rows:
        by_symbol_day[(str(row.symbol).upper(), row.bar_date)] = (
            str(row.source_provider).upper() if row.source_provider else None,
            float(row.confidence_score) if row.confidence_score is not None else None,
        )

    agg_rows = list(
        session.exec(
            select(DailyConfidenceAggregate)
            .where(DailyConfidenceAggregate.bundle_id == int(bundle_id))
            .where(DailyConfidenceAggregate.timeframe == str(timeframe))
            .where(DailyConfidenceAggregate.trading_date >= start_date)
            .where(DailyConfidenceAggregate.trading_date <= end_date)
            .order_by(DailyConfidenceAggregate.trading_date.asc())
        ).all()
    )
    by_day: dict[dt_date, tuple[str | None, float | None]] = {}
    for row in agg_rows:
        by_day[row.trading_date] = (
            _dominant_provider(dict(row.provider_mix_json or {})),
            float(row.avg_confidence),
        )
    return by_symbol_day, by_day


def build_train_dataset(
    session: Session,
    *,
    settings: Settings,
    store,
    dataset_id: int,
    force: bool = False,
    correlation_id: str | None = None,
) -> TrainDatasetRun:
    dataset = get_train_dataset(session, int(dataset_id))
    if dataset is None:
        raise APIError(code="not_found", message="Train dataset not found.", status_code=404)

    run = TrainDatasetRun(
        dataset_id=int(dataset.id or 0),
        status="RUNNING",
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    session.add(run)
    dataset.status = "RUNNING"
    dataset.updated_at = datetime.now(UTC)
    session.add(dataset)
    session.commit()
    session.refresh(run)
    session.refresh(dataset)

    warnings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    output_path = Path(settings.train_datasets_root) / f"{_slugify(dataset.name)}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        timeframe = str(dataset.timeframe or "1d").strip() or "1d"
        if timeframe != "1d":
            raise APIError(
                code="unsupported_timeframe",
                message="Train dataset builds currently support 1d timeframe only.",
                details={"timeframe": timeframe},
            )

        adjustment_mode = _normalize_token(
            dataset.adjustment_mode,
            allowed={"RAW", "ADJUSTED"},
            default="RAW",
        )
        membership_mode = _normalize_token(
            dataset.membership_mode,
            allowed={"CURRENT", "HISTORICAL"},
            default="CURRENT",
        )

        membership_intervals = _membership_intervals(session, bundle_id=int(dataset.bundle_id))
        if membership_mode == "HISTORICAL" and not membership_intervals:
            warnings.append(
                {
                    "code": "membership_history_missing",
                    "message": "No membership history found; falling back to CURRENT membership.",
                }
            )
            membership_mode = "CURRENT"

        if membership_mode == "HISTORICAL":
            symbols = sorted(membership_intervals.keys())
        else:
            symbols = store.get_bundle_symbols(
                session,
                int(dataset.bundle_id),
                timeframe=timeframe,
                membership_mode="CURRENT",
            )
        if fast_mode_enabled(settings):
            symbols = symbols[: max(1, min(len(symbols), int(settings.fast_mode_max_symbols_scan)))]
        if not symbols:
            raise APIError(
                code="missing_symbols",
                message="No bundle symbols available for train dataset build.",
                details={"bundle_id": int(dataset.bundle_id)},
            )

        by_symbol_day, by_day = _collect_source_maps(
            session,
            bundle_id=int(dataset.bundle_id),
            timeframe=timeframe,
            start_date=dataset.start_date,
            end_date=dataset.end_date,
        )
        exact_provenance_used = 0
        aggregate_fallback_used = 0
        warned_no_actions: set[str] = set()
        frames: list[pd.DataFrame] = []
        feature_cfg = dict(dataset.feature_config_json or {})
        label_cfg = dict(dataset.label_config_json or {})

        start_ts = pd.Timestamp(dataset.start_date, tz="UTC").to_pydatetime()
        end_ts = (
            pd.Timestamp(dataset.end_date, tz="UTC")
            + pd.Timedelta(hours=23, minutes=59, seconds=59)
        ).to_pydatetime()

        for symbol in symbols:
            frame = store.load_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start=start_ts,
                end=end_ts,
                session=session,
                adjustment_mode=adjustment_mode,
            )
            if frame.empty:
                continue

            if adjustment_mode == "ADJUSTED" and not _symbol_has_actions(session, symbol=symbol):
                if symbol not in warned_no_actions:
                    warnings.append(
                        {
                            "code": "corporate_actions_missing_for_symbol",
                            "message": f"No corporate actions found for {symbol}; raw prices used.",
                            "details": {"symbol": symbol},
                        }
                    )
                    warned_no_actions.add(symbol)

            working = frame.copy().sort_values("datetime").reset_index(drop=True)
            working["datetime"] = pd.to_datetime(working["datetime"], utc=True)
            working["trading_date"] = working["datetime"].dt.tz_convert("Asia/Kolkata").dt.date
            working = working[
                (working["trading_date"] >= dataset.start_date)
                & (working["trading_date"] <= dataset.end_date)
            ].reset_index(drop=True)
            if working.empty:
                continue
            if membership_mode == "HISTORICAL":
                working = _apply_membership_window(
                    working,
                    intervals=membership_intervals.get(symbol),
                )
                if working.empty:
                    continue

            working = _feature_frame(working)
            working["symbol"] = symbol

            provider_values: list[str | None] = []
            confidence_values: list[float | None] = []
            for trading_day in working["trading_date"].tolist():
                exact = by_symbol_day.get((symbol, trading_day))
                if exact is not None:
                    provider_values.append(exact[0])
                    confidence_values.append(exact[1])
                    exact_provenance_used += 1
                    continue
                fallback = by_day.get(trading_day)
                provider_values.append(fallback[0] if fallback is not None else None)
                confidence_values.append(fallback[1] if fallback is not None else None)
                aggregate_fallback_used += 1
            working["source_provider_dominant"] = provider_values
            working["confidence_score_day"] = confidence_values

            columns = [
                "symbol",
                "trading_date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "source_provider_dominant",
                "confidence_score_day",
            ]
            if _feature_enabled(feature_cfg, "return_1d", default=True):
                columns.append("return_1d")
            if _feature_enabled(feature_cfg, "return_5d", default=True):
                columns.append("return_5d")
            if _feature_enabled(feature_cfg, "atr_14", default=True):
                columns.append("atr_14")
            if _feature_enabled(feature_cfg, "rsi_14", default=True):
                columns.append("rsi_14")
            if _feature_enabled(feature_cfg, "ema_20", default=True):
                columns.append("ema_20")
            if _feature_enabled(label_cfg, "future_return_5d", default=True):
                columns.append("future_return_5d")
            frames.append(working.loc[:, columns].copy())

        if not frames:
            raise APIError(
                code="no_rows_materialized",
                message="Train dataset build produced no rows.",
                details={"dataset_id": int(dataset.id or 0)},
            )

        result = pd.concat(frames, ignore_index=True)
        result["trading_date"] = pd.to_datetime(result["trading_date"]).dt.date
        result = result.sort_values(["trading_date", "symbol"]).reset_index(drop=True)
        if output_path.exists() and not force:
            output_path.unlink()
        result.to_parquet(output_path, index=False)

        warnings.append(
            {
                "code": "confidence_source_summary",
                "message": "Confidence/provenance fields populated for dataset build.",
                "details": {
                    "symbol_level_rows": int(exact_provenance_used),
                    "aggregate_fallback_rows": int(aggregate_fallback_used),
                },
            }
        )

        dataset.status = "READY"
        dataset.row_count = int(len(result))
        dataset.updated_at = datetime.now(UTC)
        session.add(dataset)

        run.status = "SUCCEEDED"
        run.row_count = int(len(result))
        run.output_path = str(output_path)
        run.warnings_json = warnings
        run.errors_json = errors
        run.finished_at = datetime.now(UTC)
        run.updated_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        session.refresh(run)
        return run
    except Exception as exc:  # noqa: BLE001
        dataset.status = "FAILED"
        dataset.updated_at = datetime.now(UTC)
        session.add(dataset)
        run.status = "FAILED"
        run.errors_json = [
            {
                "code": getattr(exc, "code", "train_dataset_build_failed"),
                "message": str(exc),
            }
        ]
        run.warnings_json = warnings
        run.finished_at = datetime.now(UTC)
        run.updated_at = datetime.now(UTC)
        session.add(run)
        session.commit()
        session.refresh(run)
        raise


def dataset_download_info(
    session: Session,
    *,
    dataset_id: int,
) -> dict[str, Any]:
    dataset = get_train_dataset(session, dataset_id)
    if dataset is None:
        raise APIError(code="not_found", message="Train dataset not found.", status_code=404)
    latest_run = get_latest_train_dataset_run(session, dataset_id=int(dataset_id))
    path = Path(latest_run.output_path) if latest_run is not None and latest_run.output_path else None
    return {
        "dataset": serialize_train_dataset(dataset),
        "latest_run": serialize_train_dataset_run(latest_run) if latest_run is not None else None,
        "file_exists": bool(path is not None and path.exists()),
        "file_size_bytes": int(path.stat().st_size) if path is not None and path.exists() else 0,
    }
