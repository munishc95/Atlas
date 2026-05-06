from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import func
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import ForwardSignalJournal, utc_now
from app.services.data_store import DataStore
from app.services.paper import preview_policy_signals


TERMINAL_STATUSES = {"STOP_HIT", "T2_HIT", "EXPIRED"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not pd.notna(number):
        return default
    return number


def _as_int(value: Any, default: int = 0) -> int:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return default
    return number


def _parse_dt(value: Any) -> datetime:
    if value is None or value == "":
        return utc_now()
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _signal_prices(signal: dict[str, Any]) -> tuple[float, float, float, float]:
    side = str(signal.get("side", "BUY")).strip().upper()
    entry = _as_float(signal.get("entry_price", signal.get("price")))
    stop = _as_float(signal.get("stop_price"))
    risk = abs(entry - stop)
    if risk <= 0:
        risk = _as_float(signal.get("stop_distance"))
    if risk <= 0:
        risk = max(entry * 0.02, 0.01)
    if stop <= 0:
        stop = entry - risk if side == "BUY" else entry + risk
    fallback_t1 = entry + risk if side == "BUY" else entry - risk
    fallback_t2 = entry + (2.0 * risk) if side == "BUY" else entry - (2.0 * risk)
    t1 = _as_float(signal.get("target_1_price", signal.get("target_price")), fallback_t1)
    t2 = _as_float(signal.get("target_2_price"), fallback_t2)
    return entry, stop, t1, t2


def _signal_digest(
    *,
    bundle_id: int | None,
    timeframe: str,
    signal: dict[str, Any],
    signal_at: datetime,
    fill_at: datetime,
    entry: float,
    stop: float,
    target_1: float,
    target_2: float,
) -> str:
    payload = {
        "bundle_id": bundle_id,
        "timeframe": timeframe,
        "symbol": str(signal.get("symbol", "")).strip().upper(),
        "side": str(signal.get("side", "BUY")).strip().upper(),
        "template": str(signal.get("template", "")).strip(),
        "signal_at": signal_at.isoformat(),
        "fill_at": fill_at.isoformat(),
        "entry": round(entry, 6),
        "stop": round(stop, 6),
        "target_1": round(target_1, 6),
        "target_2": round(target_2, 6),
    }
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _latest_local_close_after_fill(
    *,
    session: Session,
    store: DataStore,
    symbol: str,
    timeframe: str,
    fill_at: datetime,
) -> tuple[pd.Timestamp | None, float | None]:
    frame = store.load_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start=fill_at,
        session=session,
    )
    if frame.empty:
        return None, None
    frame = frame.sort_values("datetime").reset_index(drop=True)
    latest = frame.iloc[-1]
    return pd.Timestamp(latest["datetime"]), _as_float(latest.get("close"), default=float("nan"))


def _capture_filter_reasons(
    *,
    session: Session,
    store: DataStore,
    signal: dict[str, Any],
    timeframe: str,
    fill_at: datetime,
    max_entry_extension_pct: float,
) -> list[str]:
    reasons: list[str] = []
    side = str(signal.get("side", "BUY")).strip().upper()
    entry, stop, target_1, target_2 = _signal_prices(signal)
    planned_qty = _as_int(signal.get("planned_qty"))
    position_status = str(signal.get("position_size_status", "OK")).strip().upper()
    quality_status = str(signal.get("quality_status", "PASS")).strip().upper()
    if quality_status != "PASS":
        reasons.append(f"quality_{quality_status.lower()}")
    if position_status not in {"", "OK"}:
        reasons.append(f"position_size_{position_status.lower()}")
    if planned_qty <= 0:
        reasons.append("planned_qty_zero")
    if side not in {"BUY", "SELL"}:
        reasons.append("unsupported_side")
    if min(entry, stop, target_1, target_2) <= 0:
        reasons.append("invalid_price_plan")
    if side == "BUY" and not (stop < entry < target_1 <= target_2):
        reasons.append("invalid_buy_risk_reward")
    if side == "SELL" and not (target_2 <= target_1 < entry < stop):
        reasons.append("invalid_sell_risk_reward")

    latest_dt, latest_close = _latest_local_close_after_fill(
        session=session,
        store=store,
        symbol=str(signal.get("symbol", "")).strip().upper(),
        timeframe=timeframe,
        fill_at=fill_at,
    )
    if latest_dt is not None and latest_close is not None and pd.notna(latest_close):
        latest_date = latest_dt.date()
        if latest_date > fill_at.date():
            extension = max(0.0, float(max_entry_extension_pct)) / 100.0
            if side == "BUY" and latest_close > entry * (1.0 + extension):
                reasons.append("entry_extended")
            if side == "SELL" and latest_close < entry * (1.0 - extension):
                reasons.append("entry_extended")
    return reasons


def _capture_payload(payload: dict[str, Any]) -> dict[str, Any]:
    timeframe = str(payload.get("timeframe") or "1d").strip().lower() or "1d"
    preview_payload: dict[str, Any] = {
        "regime": str(payload.get("regime") or "TREND_UP"),
        "bundle_id": payload.get("bundle_id"),
        "dataset_id": payload.get("dataset_id"),
        "policy_id": payload.get("policy_id"),
        "timeframes": [timeframe],
        "symbol_scope": payload.get("symbol_scope") or "all",
        "max_symbols_scan": payload.get("max_symbols_scan") or 500,
        "max_runtime_seconds": payload.get("max_runtime_seconds") or 60,
        "seed": payload.get("seed"),
        "asof": payload.get("asof"),
    }
    return {key: value for key, value in preview_payload.items() if value is not None}


def capture_forward_signals(
    *,
    session: Session,
    settings: Settings,
    store: DataStore,
    payload: dict[str, Any],
) -> dict[str, Any]:
    preview_payload = _capture_payload(payload)
    preview = preview_policy_signals(
        session=session,
        settings=settings,
        payload=preview_payload,
        store=store,
    )
    timeframe = str(preview_payload.get("timeframes", ["1d"])[0])
    bundle_id = preview.get("bundle_id")
    try:
        bundle_id = int(bundle_id) if bundle_id is not None else None
    except (TypeError, ValueError):
        bundle_id = None
    max_entry_extension_pct = _as_float(payload.get("max_entry_extension_pct"), 1.0)

    skipped: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for signal in preview.get("signals", []) or []:
        if not isinstance(signal, dict):
            continue
        fill_at = _parse_dt(signal.get("fill_at"))
        reasons = _capture_filter_reasons(
            session=session,
            store=store,
            signal=signal,
            timeframe=timeframe,
            fill_at=fill_at,
            max_entry_extension_pct=max_entry_extension_pct,
        )
        if reasons:
            skipped.append(
                {
                    "symbol": str(signal.get("symbol", "")).strip().upper(),
                    "template": str(signal.get("template", "")),
                    "reasons": reasons,
                }
            )
            continue
        eligible.append(signal)

    by_symbol: dict[str, dict[str, Any]] = {}
    for signal in eligible:
        symbol = str(signal.get("symbol", "")).strip().upper()
        current = by_symbol.get(symbol)
        if current is None or _as_float(signal.get("signal_strength")) > _as_float(
            current.get("signal_strength")
        ):
            if current is not None:
                skipped.append(
                    {
                        "symbol": symbol,
                        "template": str(current.get("template", "")),
                        "reasons": ["duplicate_lower_strength"],
                    }
                )
            by_symbol[symbol] = signal
        else:
            skipped.append(
                {
                    "symbol": symbol,
                    "template": str(signal.get("template", "")),
                    "reasons": ["duplicate_lower_strength"],
                }
            )

    captured_rows: list[ForwardSignalJournal] = []
    captured_count = 0
    updated_count = 0
    for signal in by_symbol.values():
        symbol = str(signal.get("symbol", "")).strip().upper()
        side = str(signal.get("side", "BUY")).strip().upper()
        signal_at = _parse_dt(signal.get("signal_at"))
        fill_at = _parse_dt(signal.get("fill_at"))
        entry, stop, target_1, target_2 = _signal_prices(signal)
        extension = max(0.0, max_entry_extension_pct) / 100.0
        digest = _signal_digest(
            bundle_id=bundle_id,
            timeframe=timeframe,
            signal=signal,
            signal_at=signal_at,
            fill_at=fill_at,
            entry=entry,
            stop=stop,
            target_1=target_1,
            target_2=target_2,
        )
        row = session.exec(
            select(ForwardSignalJournal).where(ForwardSignalJournal.signal_digest == digest)
        ).first()
        if row is None:
            row = ForwardSignalJournal(
                bundle_id=bundle_id,
                timeframe=timeframe,
                symbol=symbol,
                side=side,
                template=str(signal.get("template", "")),
                quality_status=str(signal.get("quality_status", "PASS")).strip().upper(),
                status="OPEN",
                signal_at=signal_at,
                fill_at=fill_at,
                signal_date=signal_at.date(),
                fill_date=fill_at.date(),
                entry_price=entry,
                stop_price=stop,
                target_1_price=target_1,
                target_2_price=target_2,
                planned_qty=_as_int(signal.get("planned_qty")),
                planned_risk_amount=_as_float(signal.get("planned_risk_amount")),
                planned_position_value=_as_float(signal.get("planned_position_value")),
                risk_per_share=_as_float(signal.get("risk_per_share")),
                quality_score=_as_float(signal.get("quality_score")),
                signal_strength=_as_float(signal.get("signal_strength")),
                min_entry_price=entry * (1.0 - extension),
                max_entry_price=entry * (1.0 + extension),
                signal_digest=digest,
                reasons_json=["captured_forward_test"],
                signal_json=dict(signal),
            )
            captured_count += 1
        else:
            row.signal_json = dict(signal)
            row.quality_status = str(signal.get("quality_status", row.quality_status)).strip().upper()
            row.planned_qty = _as_int(signal.get("planned_qty"), row.planned_qty)
            row.planned_risk_amount = _as_float(
                signal.get("planned_risk_amount"),
                row.planned_risk_amount,
            )
            row.planned_position_value = _as_float(
                signal.get("planned_position_value"),
                row.planned_position_value,
            )
            row.updated_at = utc_now()
            updated_count += 1
        session.add(row)
        captured_rows.append(row)
    session.commit()
    for row in captured_rows:
        session.refresh(row)

    return {
        "captured_count": captured_count,
        "updated_count": updated_count,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "journal": [serialize_forward_journal(row) for row in captured_rows],
        "preview": {
            "bundle_id": preview.get("bundle_id"),
            "dataset_id": preview.get("dataset_id"),
            "generated_signals_count": preview.get("generated_signals_count", 0),
            "candidate_quality": preview.get("candidate_quality", {}),
            "scan_truncated": preview.get("scan_truncated", False),
            "scanned_symbols": preview.get("scanned_symbols", 0),
            "total_symbols": preview.get("total_symbols", 0),
        },
        "filters": {
            "quality_status": "PASS",
            "position_size_status": "OK",
            "planned_qty": ">0",
            "max_entry_extension_pct": max_entry_extension_pct,
            "dedupe": "strongest_signal_per_symbol",
        },
    }


def _bar_time(row: pd.Series) -> datetime:
    return _parse_dt(row.get("datetime"))


def _evaluate_row(row: ForwardSignalJournal, frame: pd.DataFrame, horizon_bars: int) -> None:
    observed = frame.sort_values("datetime").head(horizon_bars).reset_index(drop=True)
    if observed.empty:
        return
    side = row.side.strip().upper()
    entry = float(row.entry_price)
    latest = observed.iloc[-1]
    row.latest_price = _as_float(latest.get("close"))
    row.latest_bar_date = _bar_time(latest).date()
    row.bars_observed = int(len(observed))
    row.horizon_bars = int(horizon_bars)

    max_favorable = row.max_favorable_pct
    max_adverse = row.max_adverse_pct
    t1_hit = row.t1_hit_at is not None
    t2_hit = row.t2_hit_at is not None
    stop_hit = row.stop_hit_at is not None

    for _, bar in observed.iterrows():
        bar_dt = _bar_time(bar)
        high = _as_float(bar.get("high"))
        low = _as_float(bar.get("low"))
        if entry > 0 and side == "BUY":
            max_favorable = max(max_favorable, ((high - entry) / entry) * 100.0)
            max_adverse = min(max_adverse, ((low - entry) / entry) * 100.0)
            hit_stop_now = low <= row.stop_price
            hit_t1_now = high >= row.target_1_price
            hit_t2_now = high >= row.target_2_price
        elif entry > 0:
            max_favorable = max(max_favorable, ((entry - low) / entry) * 100.0)
            max_adverse = min(max_adverse, ((entry - high) / entry) * 100.0)
            hit_stop_now = high >= row.stop_price
            hit_t1_now = low <= row.target_1_price
            hit_t2_now = low <= row.target_2_price
        else:
            continue

        if hit_stop_now and not t1_hit:
            row.stop_hit_at = bar_dt
            row.exit_at = bar_dt
            row.exit_reason = "STOP"
            row.status = "STOP_HIT"
            stop_hit = True
            break
        if hit_t1_now and not t1_hit:
            row.t1_hit_at = bar_dt
            t1_hit = True
        if hit_t2_now:
            row.t2_hit_at = bar_dt
            row.exit_at = bar_dt
            row.exit_reason = "TARGET_2"
            row.status = "T2_HIT"
            t2_hit = True
            break
        if hit_stop_now and t1_hit and not stop_hit:
            row.stop_hit_at = bar_dt
            row.exit_at = bar_dt
            row.exit_reason = "T1_THEN_STOP"
            stop_hit = True
            break

    latest_close = _as_float(latest.get("close"))
    if entry > 0 and side == "BUY":
        row.close_return_pct = ((latest_close - entry) / entry) * 100.0
    elif entry > 0:
        row.close_return_pct = ((entry - latest_close) / entry) * 100.0
    row.max_favorable_pct = max_favorable
    row.max_adverse_pct = max_adverse
    if t2_hit:
        row.status = "T2_HIT"
    elif row.status == "STOP_HIT":
        pass
    elif t1_hit:
        row.status = "T1_HIT"
        if row.exit_at is None and len(observed) >= horizon_bars:
            row.exit_at = row.t1_hit_at
            row.exit_reason = "TARGET_1"
    elif len(observed) >= horizon_bars:
        row.status = "EXPIRED"
        row.exit_at = _bar_time(latest)
        row.exit_reason = "TIME"
    else:
        row.status = "OPEN"
    row.updated_at = utc_now()


def evaluate_forward_journal(
    *,
    session: Session,
    store: DataStore,
    bundle_id: int | None = None,
    timeframe: str = "1d",
    horizon_bars: int = 5,
    status: str | None = None,
) -> dict[str, Any]:
    statuses = [status.strip().upper()] if status else ["OPEN", "T1_HIT"]
    stmt = select(ForwardSignalJournal).where(ForwardSignalJournal.status.in_(statuses))
    if bundle_id is not None:
        stmt = stmt.where(ForwardSignalJournal.bundle_id == bundle_id)
    if timeframe:
        stmt = stmt.where(ForwardSignalJournal.timeframe == timeframe)
    rows = list(session.exec(stmt).all())
    updated = 0
    for row in rows:
        if row.status in TERMINAL_STATUSES and not status:
            continue
        frame = store.load_ohlcv(
            symbol=row.symbol,
            timeframe=row.timeframe,
            start=row.fill_at,
            session=session,
        )
        if frame.empty:
            continue
        before = (
            row.status,
            row.bars_observed,
            row.latest_price,
            row.close_return_pct,
            row.max_favorable_pct,
            row.max_adverse_pct,
        )
        _evaluate_row(row, frame, horizon_bars=horizon_bars)
        after = (
            row.status,
            row.bars_observed,
            row.latest_price,
            row.close_return_pct,
            row.max_favorable_pct,
            row.max_adverse_pct,
        )
        if before != after:
            updated += 1
        session.add(row)
    session.commit()
    return {
        "updated_count": updated,
        "evaluated_count": len(rows),
        "summary": forward_journal_summary(session, bundle_id=bundle_id, timeframe=timeframe),
    }


def forward_journal_summary(
    session: Session,
    *,
    bundle_id: int | None = None,
    timeframe: str | None = "1d",
) -> dict[str, Any]:
    stmt = select(ForwardSignalJournal)
    if bundle_id is not None:
        stmt = stmt.where(ForwardSignalJournal.bundle_id == bundle_id)
    if timeframe:
        stmt = stmt.where(ForwardSignalJournal.timeframe == timeframe)
    rows = list(session.exec(stmt).all())
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    completed = [row for row in rows if row.status in {"STOP_HIT", "T1_HIT", "T2_HIT", "EXPIRED"}]
    winners = [row for row in completed if row.status in {"T1_HIT", "T2_HIT"}]
    win_rate = (len(winners) / len(completed)) if completed else 0.0
    avg_return = (
        sum(float(row.close_return_pct) for row in completed) / len(completed)
        if completed
        else 0.0
    )
    return {
        "total": len(rows),
        "counts": counts,
        "completed_count": len(completed),
        "t1_or_better_count": len(winners),
        "t1_or_better_rate": win_rate,
        "avg_close_return_pct": avg_return,
    }


def list_forward_journal(
    session: Session,
    *,
    page: int = 1,
    page_size: int = 50,
    bundle_id: int | None = None,
    timeframe: str | None = "1d",
    status: str | None = None,
) -> tuple[list[ForwardSignalJournal], int, dict[str, Any]]:
    stmt = select(ForwardSignalJournal)
    count_stmt = select(func.count()).select_from(ForwardSignalJournal)
    if bundle_id is not None:
        stmt = stmt.where(ForwardSignalJournal.bundle_id == bundle_id)
        count_stmt = count_stmt.where(ForwardSignalJournal.bundle_id == bundle_id)
    if timeframe:
        stmt = stmt.where(ForwardSignalJournal.timeframe == timeframe)
        count_stmt = count_stmt.where(ForwardSignalJournal.timeframe == timeframe)
    if status:
        normalized = status.strip().upper()
        stmt = stmt.where(ForwardSignalJournal.status == normalized)
        count_stmt = count_stmt.where(ForwardSignalJournal.status == normalized)
    offset = max(0, page - 1) * page_size
    rows = list(
        session.exec(
            stmt.order_by(
                ForwardSignalJournal.signal_at.desc(),
                ForwardSignalJournal.signal_strength.desc(),
                ForwardSignalJournal.id.desc(),
            )
            .offset(offset)
            .limit(page_size)
        ).all()
    )
    total = int(session.exec(count_stmt).one() or 0)
    summary = forward_journal_summary(session, bundle_id=bundle_id, timeframe=timeframe)
    return rows, total, summary


def serialize_forward_journal(row: ForwardSignalJournal) -> dict[str, Any]:
    return {
        "id": row.id,
        "bundle_id": row.bundle_id,
        "timeframe": row.timeframe,
        "symbol": row.symbol,
        "side": row.side,
        "template": row.template,
        "quality_status": row.quality_status,
        "status": row.status,
        "signal_at": row.signal_at,
        "fill_at": row.fill_at,
        "signal_date": row.signal_date,
        "fill_date": row.fill_date,
        "entry_price": row.entry_price,
        "stop_price": row.stop_price,
        "target_1_price": row.target_1_price,
        "target_2_price": row.target_2_price,
        "planned_qty": row.planned_qty,
        "planned_risk_amount": row.planned_risk_amount,
        "planned_position_value": row.planned_position_value,
        "risk_per_share": row.risk_per_share,
        "quality_score": row.quality_score,
        "signal_strength": row.signal_strength,
        "min_entry_price": row.min_entry_price,
        "max_entry_price": row.max_entry_price,
        "latest_price": row.latest_price,
        "latest_bar_date": row.latest_bar_date,
        "max_favorable_pct": row.max_favorable_pct,
        "max_adverse_pct": row.max_adverse_pct,
        "close_return_pct": row.close_return_pct,
        "t1_hit_at": row.t1_hit_at,
        "t2_hit_at": row.t2_hit_at,
        "stop_hit_at": row.stop_hit_at,
        "exit_at": row.exit_at,
        "exit_reason": row.exit_reason,
        "bars_observed": row.bars_observed,
        "horizon_bars": row.horizon_bars,
        "reasons_json": row.reasons_json,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }
