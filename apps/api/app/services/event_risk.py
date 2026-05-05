from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import Settings, get_settings


STATUS_RANK = {"PASS": 2, "WARN": 1, "FAIL": 0}
SEVERITY_STATUS = {
    "INFO": "PASS",
    "LOW": "PASS",
    "WARN": "WARN",
    "WARNING": "WARN",
    "HIGH": "WARN",
    "BLOCK": "FAIL",
    "FAIL": "FAIL",
    "CRITICAL": "FAIL",
}


@dataclass(frozen=True)
class EventRisk:
    event_date: date
    scope: str
    symbol: str
    event_type: str
    severity: str
    title: str
    source: str
    blackout_before_days: int
    blackout_after_days: int

    @property
    def status(self) -> str:
        return SEVERITY_STATUS.get(self.severity, "WARN")

    def applies_to(self, *, symbol: str | None) -> bool:
        scope = self.scope.upper()
        if scope == "MARKET":
            return True
        if scope == "SYMBOL":
            return bool(symbol) and self.symbol.upper() == str(symbol).upper()
        return False

    def is_active_on(self, day: date) -> bool:
        start = self.event_date - timedelta(days=max(0, self.blackout_before_days))
        end = self.event_date + timedelta(days=max(0, self.blackout_after_days))
        return start <= day <= end

    def as_payload(self) -> dict[str, Any]:
        return {
            "event_date": self.event_date.isoformat(),
            "scope": self.scope,
            "symbol": self.symbol or None,
            "event_type": self.event_type,
            "severity": self.severity,
            "title": self.title,
            "source": self.source,
            "blackout_before_days": int(self.blackout_before_days),
            "blackout_after_days": int(self.blackout_after_days),
        }


def _coerce_day(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _coerce_int(value: Any, default: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _status_min(left: str, right: str) -> str:
    left_norm = str(left or "PASS").upper()
    right_norm = str(right or "PASS").upper()
    return left_norm if STATUS_RANK.get(left_norm, 2) <= STATUS_RANK.get(right_norm, 2) else right_norm


def _calendar_paths(settings: Settings, overrides: dict[str, Any] | None = None) -> list[Path]:
    scope = dict(overrides or {})
    manual = str(
        scope.get("event_risk_calendar_path")
        or getattr(settings, "event_risk_calendar_path", "")
        or ""
    ).strip()
    generated = str(
        scope.get("event_risk_generated_calendar_path")
        or getattr(settings, "event_risk_generated_calendar_path", "")
        or ""
    ).strip()
    paths: list[Path] = [Path(manual or "data/reference/event_risk_calendar.csv")]
    if generated:
        paths.append(Path(generated))
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def load_event_risk_calendar(
    settings: Settings | None = None,
    overrides: dict[str, Any] | None = None,
) -> list[EventRisk]:
    resolved = settings or get_settings()
    scope = dict(overrides or {})
    enabled = bool(scope.get("event_risk_enabled", getattr(resolved, "event_risk_enabled", True)))
    if not enabled:
        return []
    frames = [pd.read_csv(path) for path in _calendar_paths(resolved, scope) if path.exists()]
    if not frames:
        return []
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        return []
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"event_date", "scope", "event_type", "severity"}
    if not required.issubset(frame.columns):
        return []

    rows: list[EventRisk] = []
    for row in frame.to_dict(orient="records"):
        event_day = _coerce_day(row.get("event_date"))
        if event_day is None:
            continue
        scope = str(row.get("scope", "MARKET") or "MARKET").strip().upper()
        if scope not in {"MARKET", "SYMBOL"}:
            continue
        symbol = str(row.get("symbol", "") or "").strip().upper()
        if scope == "SYMBOL" and not symbol:
            continue
        event_type = str(row.get("event_type", "EVENT") or "EVENT").strip().upper()
        severity = str(row.get("severity", "WARN") or "WARN").strip().upper()
        rows.append(
            EventRisk(
                event_date=event_day,
                scope=scope,
                symbol=symbol,
                event_type=event_type,
                severity=severity,
                title=str(row.get("title", event_type) or event_type).strip(),
                source=str(row.get("source", "local_calendar") or "local_calendar").strip(),
                blackout_before_days=_coerce_int(row.get("blackout_before_days"), 1),
                blackout_after_days=_coerce_int(row.get("blackout_after_days"), 0),
            )
        )
    unique: dict[tuple[date, str, str, str, str], EventRisk] = {}
    for event in rows:
        unique[
            (
                event.event_date,
                event.scope,
                event.symbol,
                event.event_type,
                event.source,
            )
        ] = event
    return sorted(unique.values(), key=lambda item: (item.event_date, item.scope, item.symbol, item.event_type))


def evaluate_event_risk(
    *,
    asof_date: date,
    symbol: str | None = None,
    settings: Settings | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    events = [
        event
        for event in load_event_risk_calendar(settings, overrides)
        if event.applies_to(symbol=symbol) and event.is_active_on(asof_date)
    ]
    status = "PASS"
    flags: list[str] = []
    for event in events:
        status = _status_min(status, event.status)
        flag_scope = event.symbol if event.scope == "SYMBOL" else "market"
        flags.append(
            f"event_risk:{event.event_type.lower()}:{flag_scope}:{event.event_date.isoformat()}"
        )
    return {
        "status": status,
        "flags": list(dict.fromkeys(flags)),
        "events": [event.as_payload() for event in events],
    }
