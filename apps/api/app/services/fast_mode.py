from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import Dataset, DatasetBundle


def fast_mode_enabled(settings: Settings) -> bool:
    return bool(settings.fast_mode_enabled)


def clamp_scan_symbols(
    *,
    settings: Settings,
    requested: int,
    hard_cap: int | None = None,
) -> int:
    value = max(1, int(requested))
    if hard_cap is not None:
        value = min(value, max(1, int(hard_cap)))
    if fast_mode_enabled(settings):
        value = min(value, max(1, int(settings.fast_mode_max_symbols_scan)))
    return value


def clamp_optuna_trials(*, settings: Settings, requested: int) -> int:
    value = max(1, int(requested))
    if fast_mode_enabled(settings):
        value = min(value, max(1, int(settings.fast_mode_max_optuna_trials)))
    return value


def clamp_job_timeout_seconds(*, settings: Settings, requested: int | None) -> int:
    base = int(settings.job_default_timeout_seconds) if requested is None else max(1, int(requested))
    if fast_mode_enabled(settings):
        return min(base, max(30, int(settings.fast_mode_job_timeout_seconds)))
    return base


def resolve_seed(*, settings: Settings, value: Any, default: int = 7) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        if fast_mode_enabled(settings):
            return int(settings.fast_mode_seed)
        return int(default)


def prefer_sample_bundle_id(session: Session, *, settings: Settings) -> int | None:
    if not fast_mode_enabled(settings):
        return None
    rows = list(session.exec(select(DatasetBundle).order_by(DatasetBundle.created_at.desc())).all())
    if not rows:
        return None

    def _is_sample(row: DatasetBundle) -> bool:
        tokens = [
            str(row.name or "").lower(),
            str(row.provider or "").lower(),
            " ".join(str(item).lower() for item in (row.symbols_json or [])),
        ]
        return any("sample" in token or "nifty500" in token for token in tokens)

    for row in rows:
        if row.id is not None and _is_sample(row):
            return int(row.id)
    return int(rows[0].id) if rows and rows[0].id is not None else None


def prefer_sample_dataset_id(
    session: Session,
    *,
    settings: Settings,
    timeframe: str,
) -> int | None:
    if not fast_mode_enabled(settings):
        return None
    rows = list(
        session.exec(
            select(Dataset)
            .where(Dataset.timeframe == str(timeframe))
            .order_by(Dataset.created_at.desc())
        ).all()
    )
    if not rows:
        return None

    def _is_sample(row: Dataset) -> bool:
        tokens = [
            str(row.symbol or "").lower(),
            str(row.provider or "").lower(),
        ]
        return any("sample" in token or "nifty500" in token for token in tokens)

    for row in rows:
        if row.id is not None and _is_sample(row):
            return int(row.id)
    return int(rows[0].id) if rows and rows[0].id is not None else None

