from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date as dt_date, datetime
from pathlib import Path
from typing import Any
import json

import pandas as pd
from sqlmodel import Session, delete, select

from app.core.exceptions import APIError
from app.db.models import CorporateAction


SUPPORTED_ACTION_TYPES = {"SPLIT", "BONUS", "DIVIDEND"}


@dataclass(frozen=True)
class CorporateActionImportSummary:
    status: str
    path: str
    mode: str
    imported_count: int
    inserted_count: int
    updated_count: int
    deleted_count: int
    warnings: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "path": self.path,
            "mode": self.mode,
            "imported_count": int(self.imported_count),
            "inserted_count": int(self.inserted_count),
            "updated_count": int(self.updated_count),
            "deleted_count": int(self.deleted_count),
            "warnings": [dict(item) for item in self.warnings],
        }


def _normalize_mode(value: str | None) -> str:
    token = str(value or "UPSERT").strip().upper()
    return "REPLACE" if token == "REPLACE" else "UPSERT"


def _read_actions_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise APIError(
            code="file_not_found",
            message=f"Corporate actions file not found: {path}",
            status_code=404,
        )
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            rows = raw.get("items", raw.get("rows", []))
        else:
            rows = raw
        return pd.DataFrame(rows)
    raise APIError(
        code="unsupported_file_type",
        message="Corporate actions import supports CSV or JSON files.",
        details={"path": str(path)},
    )


def _coerce_ratio_parts(row: pd.Series) -> tuple[float, float]:
    if pd.notna(row.get("ratio_num")) and pd.notna(row.get("ratio_den")):
        try:
            return float(row["ratio_num"]), float(row["ratio_den"])
        except (TypeError, ValueError):
            pass
    ratio_token = row.get("ratio")
    if isinstance(ratio_token, str) and ":" in ratio_token:
        left, right = ratio_token.split(":", 1)
        try:
            return float(left.strip()), float(right.strip())
        except (TypeError, ValueError):
            return 1.0, 1.0
    return 1.0, 1.0


def _normalize_actions_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if frame.empty:
        raise APIError(code="invalid_payload", message="Corporate actions file is empty.")
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    required = {"symbol", "ex_date", "action_type"}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise APIError(
            code="invalid_payload",
            message="Corporate actions file missing required columns.",
            details={"missing_columns": missing},
        )
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper().str.strip()
    normalized["action_type"] = normalized["action_type"].astype(str).str.upper().str.strip()
    normalized["ex_date"] = pd.to_datetime(normalized["ex_date"], errors="coerce").dt.date
    normalized = normalized.dropna(subset=["symbol", "action_type", "ex_date"])
    warnings: list[dict[str, Any]] = []
    unsupported = sorted(
        {
            str(action)
            for action in normalized["action_type"].tolist()
            if str(action).upper() not in SUPPORTED_ACTION_TYPES
        }
    )
    if unsupported:
        warnings.append(
            {
                "code": "unsupported_action_types_skipped",
                "message": "Unsupported corporate action types were skipped.",
                "details": {"action_types": unsupported},
            }
        )
        normalized = normalized[normalized["action_type"].isin(sorted(SUPPORTED_ACTION_TYPES))]
    if normalized.empty:
        raise APIError(
            code="invalid_payload",
            message="No supported corporate actions found after normalization.",
        )
    ratio_parts = normalized.apply(_coerce_ratio_parts, axis=1, result_type="expand")
    normalized["ratio_num"] = pd.to_numeric(ratio_parts[0], errors="coerce").fillna(1.0)
    normalized["ratio_den"] = pd.to_numeric(ratio_parts[1], errors="coerce").fillna(1.0)
    normalized["cash_amount"] = pd.to_numeric(
        normalized["cash_amount"] if "cash_amount" in normalized.columns else 0.0,
        errors="coerce",
    )
    normalized["source"] = (
        normalized["source"].astype(str).str.strip()
        if "source" in normalized.columns
        else "local_import"
    )
    normalized = normalized.sort_values(["symbol", "ex_date", "action_type"]).drop_duplicates(
        subset=["symbol", "ex_date", "action_type"],
        keep="last",
    )
    return normalized.reset_index(drop=True), warnings


def import_corporate_actions(
    session: Session,
    *,
    path: str,
    mode: str = "UPSERT",
) -> dict[str, Any]:
    source_path = Path(path)
    normalized_mode = _normalize_mode(mode)
    frame = _read_actions_file(source_path)
    normalized, warnings = _normalize_actions_frame(frame)

    deleted_count = 0
    inserted_count = 0
    updated_count = 0
    if normalized_mode == "REPLACE":
        existing = list(session.exec(select(CorporateAction)).all())
        deleted_count = len(existing)
        session.exec(delete(CorporateAction))
        session.flush()

    for row in normalized.to_dict(orient="records"):
        existing = session.exec(
            select(CorporateAction)
            .where(CorporateAction.symbol == str(row["symbol"]))
            .where(CorporateAction.ex_date == row["ex_date"])
            .where(CorporateAction.action_type == str(row["action_type"]))
            .order_by(CorporateAction.id.desc())
        ).first()
        payload = {
            "symbol": str(row["symbol"]),
            "ex_date": row["ex_date"],
            "action_type": str(row["action_type"]),
            "ratio_num": float(row.get("ratio_num", 1.0) or 1.0),
            "ratio_den": float(row.get("ratio_den", 1.0) or 1.0),
            "cash_amount": (
                float(row["cash_amount"]) if pd.notna(row.get("cash_amount")) else None
            ),
            "source": str(row.get("source", "local_import") or "local_import"),
        }
        if existing is None:
            session.add(CorporateAction(**payload))
            inserted_count += 1
        else:
            existing.ratio_num = payload["ratio_num"]
            existing.ratio_den = payload["ratio_den"]
            existing.cash_amount = payload["cash_amount"]
            existing.source = payload["source"]
            session.add(existing)
            updated_count += 1

    session.commit()
    summary = CorporateActionImportSummary(
        status="SUCCEEDED",
        path=str(source_path),
        mode=normalized_mode,
        imported_count=int(len(normalized)),
        inserted_count=int(inserted_count),
        updated_count=int(updated_count),
        deleted_count=int(deleted_count),
        warnings=warnings,
    )
    return summary.as_dict()


def list_corporate_actions(
    session: Session,
    *,
    symbol: str | None = None,
    limit: int = 100,
) -> list[CorporateAction]:
    stmt = select(CorporateAction)
    if isinstance(symbol, str) and symbol.strip():
        stmt = stmt.where(CorporateAction.symbol == symbol.strip().upper())
    stmt = stmt.order_by(CorporateAction.ex_date.desc(), CorporateAction.id.desc())
    stmt = stmt.limit(max(1, min(int(limit), 500)))
    return list(session.exec(stmt).all())


def corporate_actions_status(
    session: Session,
    *,
    bundle_symbols: list[str] | None = None,
    adjustment_mode: str = "RAW",
) -> dict[str, Any]:
    rows = list(session.exec(select(CorporateAction)).all())
    type_counts: dict[str, int] = {}
    symbols_with_actions = {str(row.symbol).upper() for row in rows}
    for row in rows:
        action_type = str(row.action_type).upper()
        type_counts[action_type] = int(type_counts.get(action_type, 0)) + 1
    scoped_symbols = {
        str(item).upper().strip()
        for item in (bundle_symbols or [])
        if str(item).strip()
    }
    covered = len(scoped_symbols.intersection(symbols_with_actions)) if scoped_symbols else 0
    total = len(scoped_symbols)
    return {
        "action_count": int(len(rows)),
        "action_counts_by_type": {key: int(value) for key, value in sorted(type_counts.items())},
        "symbols_with_actions": int(len(symbols_with_actions)),
        "bundle_symbol_count": int(total),
        "bundle_symbols_with_actions": int(covered),
        "bundle_coverage_pct": (float((covered / total) * 100.0) if total > 0 else 0.0),
        "adjustment_mode": str(adjustment_mode).upper(),
    }


def _price_volume_factors(row: CorporateAction) -> tuple[float, float] | None:
    action_type = str(row.action_type or "").upper()
    ratio_num = float(row.ratio_num or 1.0)
    ratio_den = float(row.ratio_den or 1.0)
    if ratio_num <= 0 or ratio_den <= 0:
        return None
    if action_type == "SPLIT":
        price_factor = ratio_den / ratio_num
        volume_factor = ratio_num / ratio_den
        return price_factor, volume_factor
    if action_type == "BONUS":
        total_shares = ratio_num + ratio_den
        if total_shares <= 0:
            return None
        price_factor = ratio_den / total_shares
        volume_factor = total_shares / ratio_den
        return price_factor, volume_factor
    return None


def list_symbol_actions(session: Session, *, symbol: str) -> list[CorporateAction]:
    return list(
        session.exec(
            select(CorporateAction)
            .where(CorporateAction.symbol == symbol.upper())
            .order_by(CorporateAction.ex_date.asc(), CorporateAction.id.asc())
        ).all()
    )


def apply_adjustment_mode(
    session: Session,
    *,
    symbol: str,
    frame: pd.DataFrame,
    adjustment_mode: str = "RAW",
) -> tuple[pd.DataFrame, list[str]]:
    mode = str(adjustment_mode or "RAW").strip().upper()
    if mode != "ADJUSTED" or frame.empty:
        return frame.copy(), []
    actions = list_symbol_actions(session, symbol=symbol)
    if not actions:
        return frame.copy(), [f"no_corporate_actions:{symbol.upper()}"]

    adjusted = frame.copy().sort_values("datetime").reset_index(drop=True)
    adjusted["datetime"] = pd.to_datetime(adjusted["datetime"], utc=True, errors="coerce")
    adjusted = adjusted.dropna(subset=["datetime"])
    if adjusted.empty:
        return adjusted, []

    adjusted["_bar_date_ist"] = adjusted["datetime"].dt.tz_convert("Asia/Kolkata").dt.date
    for column in ("open", "high", "low", "close", "volume"):
        adjusted[column] = pd.to_numeric(adjusted[column], errors="coerce").astype(float)
    notes: list[str] = []
    for action in actions:
        factors = _price_volume_factors(action)
        if factors is None:
            notes.append(f"action_skipped:{action.action_type.lower()}:{action.ex_date.isoformat()}")
            continue
        price_factor, volume_factor = factors
        mask = adjusted["_bar_date_ist"] < action.ex_date
        if not bool(mask.any()):
            continue
        for column in ("open", "high", "low", "close"):
            adjusted.loc[mask, column] = adjusted.loc[mask, column] * price_factor
        adjusted.loc[mask, "volume"] = adjusted.loc[mask, "volume"] * volume_factor
    adjusted = adjusted.drop(columns=["_bar_date_ist"])
    return adjusted.reset_index(drop=True), notes


def serialize_corporate_action(row: CorporateAction) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "symbol": str(row.symbol),
        "ex_date": row.ex_date.isoformat(),
        "action_type": str(row.action_type),
        "ratio_num": float(row.ratio_num),
        "ratio_den": float(row.ratio_den),
        "cash_amount": float(row.cash_amount) if row.cash_amount is not None else None,
        "source": str(row.source),
        "created_at": row.created_at.astimezone(UTC).isoformat(),
    }
