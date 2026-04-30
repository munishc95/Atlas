from __future__ import annotations

from datetime import UTC, date as dt_date, datetime
from pathlib import Path
from typing import Any
import json

import pandas as pd
from sqlmodel import Session, delete, select

from app.core.exceptions import APIError
from app.db.models import BundleMembershipHistory, DatasetBundle


def _normalize_mode(value: str | None) -> str:
    token = str(value or "UPSERT").strip().upper()
    return "REPLACE" if token == "REPLACE" else "UPSERT"


def _read_rows(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise APIError(
            code="file_not_found",
            message=f"Membership history file not found: {path}",
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
        message="Membership history import supports CSV or JSON files.",
        details={"path": str(path)},
    )


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise APIError(code="invalid_payload", message="Membership history file is empty.")
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    required = {"symbol", "effective_from"}
    missing = sorted(required.difference(normalized.columns))
    if missing:
        raise APIError(
            code="invalid_payload",
            message="Membership history file missing required columns.",
            details={"missing_columns": missing},
        )
    normalized["symbol"] = normalized["symbol"].astype(str).str.upper().str.strip()
    normalized["effective_from"] = pd.to_datetime(
        normalized["effective_from"], errors="coerce"
    ).dt.date
    if "effective_to" in normalized.columns:
        normalized["effective_to"] = pd.to_datetime(
            normalized["effective_to"], errors="coerce"
        ).dt.date
    else:
        normalized["effective_to"] = None
    normalized["source"] = (
        normalized["source"].astype(str).str.strip()
        if "source" in normalized.columns
        else "local_import"
    )
    normalized = normalized.dropna(subset=["symbol", "effective_from"])
    normalized = normalized.sort_values(["symbol", "effective_from"]).drop_duplicates(
        subset=["symbol", "effective_from", "effective_to"],
        keep="last",
    )
    if normalized.empty:
        raise APIError(
            code="invalid_payload",
            message="No valid membership rows found after normalization.",
        )
    return normalized.reset_index(drop=True)


def import_bundle_membership_history(
    session: Session,
    *,
    bundle_id: int,
    path: str,
    mode: str = "UPSERT",
) -> dict[str, Any]:
    bundle = session.get(DatasetBundle, int(bundle_id))
    if bundle is None:
        raise APIError(code="not_found", message="Bundle not found.", status_code=404)
    source_path = Path(path)
    normalized_mode = _normalize_mode(mode)
    frame = _normalize_frame(_read_rows(source_path))
    deleted_count = 0
    inserted_count = 0
    updated_count = 0

    if normalized_mode == "REPLACE":
        existing = list(
            session.exec(
                select(BundleMembershipHistory).where(
                    BundleMembershipHistory.bundle_id == int(bundle_id)
                )
            ).all()
        )
        deleted_count = len(existing)
        session.exec(
            delete(BundleMembershipHistory).where(
                BundleMembershipHistory.bundle_id == int(bundle_id)
            )
        )
        session.flush()

    for row in frame.to_dict(orient="records"):
        effective_to = row.get("effective_to")
        if pd.isna(effective_to):
            effective_to = None
        existing = session.exec(
            select(BundleMembershipHistory)
            .where(BundleMembershipHistory.bundle_id == int(bundle_id))
            .where(BundleMembershipHistory.symbol == str(row["symbol"]))
            .where(BundleMembershipHistory.effective_from == row["effective_from"])
            .order_by(BundleMembershipHistory.id.desc())
        ).first()
        payload = {
            "bundle_id": int(bundle_id),
            "symbol": str(row["symbol"]),
            "effective_from": row["effective_from"],
            "effective_to": effective_to,
            "source": str(row.get("source", "local_import") or "local_import"),
        }
        if existing is None:
            session.add(BundleMembershipHistory(**payload))
            inserted_count += 1
        else:
            existing.effective_to = payload["effective_to"]
            existing.source = payload["source"]
            session.add(existing)
            updated_count += 1

    today = datetime.now(UTC).date()
    current_symbols = active_bundle_symbols(
        session,
        bundle_id=int(bundle_id),
        asof_date=today,
    )
    if current_symbols:
        bundle.symbols_json = sorted(current_symbols)
        session.add(bundle)
    session.commit()
    return {
        "status": "SUCCEEDED",
        "bundle_id": int(bundle_id),
        "path": str(source_path),
        "mode": normalized_mode,
        "imported_count": int(len(frame)),
        "inserted_count": int(inserted_count),
        "updated_count": int(updated_count),
        "deleted_count": int(deleted_count),
    }


def list_membership_history(
    session: Session,
    *,
    bundle_id: int,
) -> list[BundleMembershipHistory]:
    return list(
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


def bundle_has_membership_history(session: Session, *, bundle_id: int) -> bool:
    row = session.exec(
        select(BundleMembershipHistory.id)
        .where(BundleMembershipHistory.bundle_id == int(bundle_id))
        .limit(1)
    ).first()
    return row is not None


def active_bundle_symbols(
    session: Session,
    *,
    bundle_id: int,
    asof_date: dt_date,
) -> list[str]:
    rows = list_membership_history(session, bundle_id=int(bundle_id))
    active: list[str] = []
    for row in rows:
        if row.effective_from > asof_date:
            continue
        if row.effective_to is not None and row.effective_to < asof_date:
            continue
        active.append(str(row.symbol).upper())
    return list(dict.fromkeys(sorted(active)))


def serialize_membership_row(row: BundleMembershipHistory) -> dict[str, Any]:
    return {
        "id": int(row.id) if row.id is not None else None,
        "bundle_id": int(row.bundle_id),
        "symbol": str(row.symbol),
        "effective_from": row.effective_from.isoformat(),
        "effective_to": row.effective_to.isoformat() if row.effective_to is not None else None,
        "source": str(row.source),
        "created_at": row.created_at.astimezone(UTC).isoformat(),
    }
