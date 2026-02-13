from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
from typing import Any

from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DailyReport, PaperRun


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


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


def _as_date(value: date | None) -> date:
    return value or _utc_now().date()


def _runs_for_day(
    session: Session,
    *,
    report_date: date,
    bundle_id: int | None = None,
    policy_id: int | None = None,
) -> list[PaperRun]:
    start = datetime.combine(report_date, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(report_date, datetime.max.time(), tzinfo=timezone.utc)
    stmt = (
        select(PaperRun)
        .where(PaperRun.asof_ts >= start)
        .where(PaperRun.asof_ts <= end)
        .order_by(PaperRun.asof_ts.asc())
    )
    if bundle_id is not None:
        stmt = stmt.where(PaperRun.bundle_id == bundle_id)
    if policy_id is not None:
        stmt = stmt.where(PaperRun.policy_id == policy_id)
    return list(session.exec(stmt).all())


def _histogram(rows: list[PaperRun], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        source = summary.get(key, {})
        if not isinstance(source, dict):
            continue
        for item, count in source.items():
            counter[str(item)] += _safe_int(count, 0)
    return dict(counter)


def _sum_summary(rows: list[PaperRun], key: str) -> float:
    total = 0.0
    for row in rows:
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        total += _safe_float(summary.get(key), 0.0)
    return float(total)


def _latest_existing_report(
    session: Session,
    *,
    report_date: date,
    bundle_id: int | None,
    policy_id: int | None,
) -> DailyReport | None:
    stmt = select(DailyReport).where(DailyReport.date == report_date)
    if bundle_id is None:
        stmt = stmt.where(DailyReport.bundle_id.is_(None))
    else:
        stmt = stmt.where(DailyReport.bundle_id == bundle_id)
    if policy_id is None:
        stmt = stmt.where(DailyReport.policy_id.is_(None))
    else:
        stmt = stmt.where(DailyReport.policy_id == policy_id)
    return session.exec(stmt.order_by(DailyReport.id.desc())).first()


def build_daily_report_content(
    *,
    report_date: date,
    rows: list[PaperRun],
    bundle_id: int | None,
    policy_id: int | None,
) -> dict[str, Any]:
    if not rows:
        return {
            "date": report_date.isoformat(),
            "bundle_id": bundle_id,
            "policy_id": policy_id,
            "summary": {
                "runs": 0,
                "entries": 0,
                "exits": 0,
                "positions_open": 0,
                "positions_closed": 0,
                "realized_pnl": 0.0,
                "unrealized_pnl": 0.0,
                "net_pnl": 0.0,
                "costs": 0.0,
                "drawdown": 0.0,
                "kill_switch_active": False,
            },
            "explainability": {
                "selected_reason_histogram": {},
                "skipped_reason_histogram": {},
                "scan_truncated_runs": 0,
            },
            "risk": {
                "avg_exposure": 0.0,
                "positions_peak": 0,
                "adv_cap_hits": 0,
                "correlation_cap_hits": 0,
                "sector_cap_hits": 0,
            },
            "links": {"paper_run_ids": [], "order_ids": []},
        }

    first = rows[0]
    last = rows[-1]
    first_summary = first.summary_json if isinstance(first.summary_json, dict) else {}
    last_summary = last.summary_json if isinstance(last.summary_json, dict) else {}

    entries = int(sum(_safe_int((row.summary_json or {}).get("positions_opened"), 0) for row in rows))
    exits = int(sum(_safe_int((row.summary_json or {}).get("positions_closed"), 0) for row in rows))
    net_pnl = float(sum(_safe_float((row.summary_json or {}).get("net_pnl"), 0.0) for row in rows))
    total_cost = float(sum(_safe_float((row.summary_json or {}).get("total_cost"), 0.0) for row in rows))
    selected_hist = _histogram(rows, "selected_reason_histogram")
    skipped_hist = _histogram(rows, "skipped_reason_histogram")
    scan_truncated = int(sum(1 for row in rows if bool(row.scan_truncated)))
    avg_exposure = float(
        sum(_safe_float((row.summary_json or {}).get("exposure"), 0.0) for row in rows)
        / max(1, len(rows))
    )
    positions_peak = max(
        [_safe_int((row.summary_json or {}).get("positions_after"), 0) for row in rows] or [0]
    )

    return {
        "date": report_date.isoformat(),
        "bundle_id": bundle_id,
        "policy_id": policy_id,
        "summary": {
            "runs": len(rows),
            "entries": entries,
            "exits": exits,
            "positions_open": _safe_int(last_summary.get("positions_after"), 0),
            "positions_closed": exits,
            "realized_pnl": _sum_summary(rows, "realized_pnl"),
            "unrealized_pnl": _sum_summary(rows, "unrealized_pnl"),
            "net_pnl": net_pnl,
            "costs": total_cost,
            "drawdown": _safe_float(last_summary.get("drawdown"), 0.0),
            "kill_switch_active": bool(last_summary.get("kill_switch_active", False)),
            "signals_source": str(last.signals_source),
            "regime": str(last.regime),
        },
        "explainability": {
            "selected_reason_histogram": selected_hist,
            "skipped_reason_histogram": skipped_hist,
            "scan_truncated_runs": scan_truncated,
            "scanned_symbols": int(sum(_safe_int(row.scanned_symbols, 0) for row in rows)),
            "evaluated_candidates": int(
                sum(_safe_int(row.evaluated_candidates, 0) for row in rows)
            ),
        },
        "risk": {
            "avg_exposure": avg_exposure,
            "positions_peak": positions_peak,
            "adv_cap_hits": _safe_int(skipped_hist.get("adv_cap_zero_qty", 0), 0),
            "correlation_cap_hits": _safe_int(skipped_hist.get("correlation_threshold", 0), 0),
            "sector_cap_hits": _safe_int(skipped_hist.get("sector_concentration", 0), 0),
            "equity_start": _safe_float(first_summary.get("equity_before"), 0.0),
            "equity_end": _safe_float(last_summary.get("equity_after"), 0.0),
        },
        "links": {
            "paper_run_ids": [int(row.id) for row in rows if row.id is not None],
            "job_ids": [
                str((row.summary_json or {}).get("job_id"))
                for row in rows
                if isinstance((row.summary_json or {}).get("job_id"), str)
            ],
            "order_ids": [
                int(item)
                for row in rows
                for item in (
                    (row.summary_json or {}).get("new_order_ids", [])
                    if isinstance((row.summary_json or {}).get("new_order_ids"), list)
                    else []
                )
                if isinstance(item, int)
            ],
        },
    }


def generate_daily_report(
    session: Session,
    settings: Settings,
    *,
    report_date: date | None = None,
    bundle_id: int | None = None,
    policy_id: int | None = None,
    overwrite: bool = True,
) -> DailyReport:
    resolved_date = _as_date(report_date)
    if not overwrite:
        existing = _latest_existing_report(
            session,
            report_date=resolved_date,
            bundle_id=bundle_id,
            policy_id=policy_id,
        )
        if existing is not None:
            return existing

    rows = _runs_for_day(
        session,
        report_date=resolved_date,
        bundle_id=bundle_id,
        policy_id=policy_id,
    )
    content = build_daily_report_content(
        report_date=resolved_date,
        rows=rows,
        bundle_id=bundle_id,
        policy_id=policy_id,
    )

    existing = _latest_existing_report(
        session,
        report_date=resolved_date,
        bundle_id=bundle_id,
        policy_id=policy_id,
    )
    if existing is not None:
        existing.content_json = content
        existing.created_at = _utc_now()
        session.add(existing)
        session.commit()
        session.refresh(existing)
        return existing

    row = DailyReport(
        date=resolved_date,
        bundle_id=bundle_id,
        policy_id=policy_id,
        content_json=content,
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def list_daily_reports(
    session: Session,
    *,
    report_date: date | None,
    bundle_id: int | None,
    policy_id: int | None,
) -> list[DailyReport]:
    stmt = select(DailyReport).order_by(DailyReport.date.desc(), DailyReport.created_at.desc())
    if report_date is not None:
        stmt = stmt.where(DailyReport.date == report_date)
    if bundle_id is not None:
        stmt = stmt.where(DailyReport.bundle_id == bundle_id)
    if policy_id is not None:
        stmt = stmt.where(DailyReport.policy_id == policy_id)
    return list(session.exec(stmt).all())


def get_daily_report(session: Session, report_id: int) -> DailyReport:
    row = session.get(DailyReport, report_id)
    if row is None:
        raise APIError(code="not_found", message="Daily report not found", status_code=404)
    return row
