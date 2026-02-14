from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from typing import Any

import numpy as np
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import PaperRun, PortfolioRiskSnapshot


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _cfg_float(
    settings: Settings,
    overrides: dict[str, Any] | None,
    key: str,
    fallback: float,
) -> float:
    if isinstance(overrides, dict):
        value = overrides.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(getattr(settings, key, fallback))


def _cfg_bool(
    settings: Settings,
    overrides: dict[str, Any] | None,
    key: str,
    fallback: bool,
) -> bool:
    if isinstance(overrides, dict):
        value = overrides.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            token = value.strip().lower()
            if token in {"true", "1", "yes", "on"}:
                return True
            if token in {"false", "0", "no", "off"}:
                return False
    return bool(getattr(settings, key, fallback))


@dataclass
class PortfolioRiskConfig:
    enabled: bool
    target_vol_annual: float
    lookback_days: int
    min_scale: float
    max_scale: float
    max_gross_exposure_pct: float
    max_single_name_exposure_pct: float
    max_sector_exposure_pct: float
    corr_clamp_enabled: bool
    corr_threshold: float
    corr_reduce_factor: float


def resolve_portfolio_risk_config(
    *,
    settings: Settings,
    overrides: dict[str, Any] | None,
) -> PortfolioRiskConfig:
    operate_mode = str(
        (overrides or {}).get("operate_mode", settings.operate_mode)
    ).strip().lower()
    explicit_enabled = (overrides or {}).get("risk_overlay_enabled")
    if isinstance(explicit_enabled, bool):
        enabled = explicit_enabled
    elif isinstance(explicit_enabled, str):
        enabled = explicit_enabled.strip().lower() in {"true", "1", "yes", "on"}
    else:
        enabled = bool(operate_mode == "live")

    min_scale = max(0.0, _cfg_float(settings, overrides, "risk_overlay_min_scale", settings.risk_overlay_min_scale))
    max_scale = max(min_scale, _cfg_float(settings, overrides, "risk_overlay_max_scale", settings.risk_overlay_max_scale))
    corr_reduce = _cfg_float(
        settings,
        overrides,
        "risk_overlay_corr_reduce_factor",
        settings.risk_overlay_corr_reduce_factor,
    )

    return PortfolioRiskConfig(
        enabled=enabled,
        target_vol_annual=max(
            0.0,
            _cfg_float(
                settings,
                overrides,
                "risk_overlay_target_vol_annual",
                settings.risk_overlay_target_vol_annual,
            ),
        ),
        lookback_days=max(
            5,
            _safe_int(
                (overrides or {}).get("risk_overlay_lookback_days", settings.risk_overlay_lookback_days),
                settings.risk_overlay_lookback_days,
            ),
        ),
        min_scale=min_scale,
        max_scale=max_scale,
        max_gross_exposure_pct=max(
            0.0,
            _cfg_float(
                settings,
                overrides,
                "risk_overlay_max_gross_exposure_pct",
                settings.risk_overlay_max_gross_exposure_pct,
            ),
        ),
        max_single_name_exposure_pct=max(
            0.0,
            _cfg_float(
                settings,
                overrides,
                "risk_overlay_max_single_name_exposure_pct",
                settings.risk_overlay_max_single_name_exposure_pct,
            ),
        ),
        max_sector_exposure_pct=max(
            0.0,
            _cfg_float(
                settings,
                overrides,
                "risk_overlay_max_sector_exposure_pct",
                settings.risk_overlay_max_sector_exposure_pct,
            ),
        ),
        corr_clamp_enabled=_cfg_bool(
            settings,
            overrides,
            "risk_overlay_corr_clamp_enabled",
            settings.risk_overlay_corr_clamp_enabled,
        ),
        corr_threshold=max(
            0.0,
            _cfg_float(
                settings,
                overrides,
                "risk_overlay_corr_threshold",
                settings.risk_overlay_corr_threshold,
            ),
        ),
        corr_reduce_factor=max(0.0, min(1.0, corr_reduce)),
    )


def _window_returns(
    *,
    session: Session,
    bundle_id: int | None,
    policy_id: int | None,
    lookback_days: int,
    asof: datetime | None,
) -> list[float]:
    stmt = (
        select(PaperRun)
        .where(PaperRun.mode == "LIVE")
        .order_by(PaperRun.asof_ts.desc(), PaperRun.id.desc())
        .limit(max(lookback_days * 3, lookback_days))
    )
    if bundle_id is not None:
        stmt = stmt.where(PaperRun.bundle_id == int(bundle_id))
    if policy_id is not None:
        stmt = stmt.where(PaperRun.policy_id == int(policy_id))
    if asof is not None:
        stmt = stmt.where(PaperRun.asof_ts <= asof)
    rows = list(session.exec(stmt).all())
    rows.reverse()
    returns: list[float] = []
    for row in rows:
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        equity_before = _safe_float(summary.get("equity_before"), 0.0)
        net_pnl = _safe_float(summary.get("net_pnl"), 0.0)
        if equity_before <= 1e-9:
            continue
        returns.append(net_pnl / equity_before)
    if len(returns) > lookback_days:
        returns = returns[-lookback_days:]
    return returns


def realized_volatility_annual(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return float(np.std(np.array(returns, dtype=float), ddof=0) * sqrt(252.0))


def compute_risk_scale(
    *,
    realized_vol: float,
    target_vol: float,
    min_scale: float,
    max_scale: float,
    eps: float = 1e-9,
) -> float:
    if target_vol <= 0:
        return max(min_scale, min(1.0, max_scale))
    scale_raw = target_vol / max(realized_vol, eps)
    return float(max(min_scale, min(max_scale, scale_raw)))


def create_portfolio_risk_snapshot(
    session: Session,
    *,
    settings: Settings,
    bundle_id: int | None,
    policy_id: int | None,
    overrides: dict[str, Any] | None,
    asof: datetime | None = None,
) -> dict[str, Any]:
    config = resolve_portfolio_risk_config(settings=settings, overrides=overrides)
    if asof is None:
        asof = datetime.now(timezone.utc)
    elif asof.tzinfo is None:
        asof = asof.replace(tzinfo=timezone.utc)
    else:
        asof = asof.astimezone(timezone.utc)

    returns = _window_returns(
        session=session,
        bundle_id=bundle_id,
        policy_id=policy_id,
        lookback_days=config.lookback_days,
        asof=asof,
    )
    realized = realized_volatility_annual(returns)
    if not config.enabled:
        scale = 1.0
    elif len(returns) < 2:
        scale = 1.0
    else:
        scale = compute_risk_scale(
            realized_vol=realized,
            target_vol=config.target_vol_annual,
            min_scale=config.min_scale,
            max_scale=config.max_scale,
        )
    notes = {
        "return_points": len(returns),
        "lookback_days": config.lookback_days,
        "enabled": bool(config.enabled),
        "insufficient_history": len(returns) < 2,
        "returns_preview": [float(round(item, 8)) for item in returns[-5:]],
    }
    row = PortfolioRiskSnapshot(
        ts=asof,
        bundle_id=bundle_id,
        policy_id=policy_id,
        realized_vol=float(realized),
        target_vol=float(config.target_vol_annual),
        scale=float(scale),
        notes_json=notes,
    )
    session.add(row)
    session.flush()

    return {
        "snapshot_id": int(row.id) if row.id is not None else None,
        "enabled": bool(config.enabled),
        "realized_vol": float(realized),
        "target_vol": float(config.target_vol_annual),
        "risk_scale": float(scale),
        "lookback_days": int(config.lookback_days),
        "notes": notes,
        "caps": {
            "max_gross_exposure_pct": float(config.max_gross_exposure_pct),
            "max_single_name_exposure_pct": float(config.max_single_name_exposure_pct),
            "max_sector_exposure_pct": float(config.max_sector_exposure_pct),
        },
        "corr_clamp": {
            "enabled": bool(config.corr_clamp_enabled),
            "threshold": float(config.corr_threshold),
            "reduce_factor": float(config.corr_reduce_factor),
        },
    }
