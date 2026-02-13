from __future__ import annotations

from collections import Counter, defaultdict
import calendar
from datetime import date, datetime, timezone
from io import BytesIO
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DailyReport, MonthlyReport, PaperRun


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


def _runs_for_month(
    session: Session,
    *,
    month_start: date,
    month_end: date,
    bundle_id: int | None = None,
    policy_id: int | None = None,
) -> list[PaperRun]:
    start = datetime.combine(month_start, datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(month_end, datetime.max.time(), tzinfo=timezone.utc)
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


def _latest_existing_daily_report(
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


def _latest_existing_monthly_report(
    session: Session,
    *,
    month: str,
    bundle_id: int | None,
    policy_id: int | None,
) -> MonthlyReport | None:
    stmt = select(MonthlyReport).where(MonthlyReport.month == month)
    if bundle_id is None:
        stmt = stmt.where(MonthlyReport.bundle_id.is_(None))
    else:
        stmt = stmt.where(MonthlyReport.bundle_id == bundle_id)
    if policy_id is None:
        stmt = stmt.where(MonthlyReport.policy_id.is_(None))
    else:
        stmt = stmt.where(MonthlyReport.policy_id == policy_id)
    return session.exec(stmt.order_by(MonthlyReport.id.desc())).first()


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
                "safe_mode_active": False,
                "safe_mode_runs": 0,
                "data_quality_status": None,
                "mode": "LIVE",
                "shadow_note": None,
            },
            "explainability": {
                "selected_reason_histogram": {},
                "skipped_reason_histogram": {},
                "scan_truncated_runs": 0,
                "data_quality_warning_messages": [],
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
    safe_mode_runs = int(
        sum(
            1
            for row in rows
            if bool(((row.summary_json or {}) if isinstance(row.summary_json, dict) else {}).get("safe_mode_active"))
        )
    )
    quality_warning_messages: list[str] = []
    for row in rows:
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        warnings = summary.get("data_quality_warn_summary", [])
        if not isinstance(warnings, list):
            continue
        for item in warnings:
            if not isinstance(item, dict):
                continue
            message = str(item.get("message", "")).strip()
            if message and message not in quality_warning_messages:
                quality_warning_messages.append(message)
            if len(quality_warning_messages) >= 5:
                break
        if len(quality_warning_messages) >= 5:
            break
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
            "safe_mode_active": bool(last_summary.get("safe_mode_active", False)),
            "safe_mode_runs": safe_mode_runs,
            "data_quality_status": last_summary.get("data_quality_status"),
            "signals_source": str(last.signals_source),
            "regime": str(last.regime),
            "mode": str(last_summary.get("execution_mode", last.mode or "LIVE")),
            "shadow_note": (
                str(last_summary.get("shadow_note"))
                if str(last_summary.get("execution_mode", "")).upper() == "SHADOW"
                else None
            ),
        },
        "explainability": {
            "selected_reason_histogram": selected_hist,
            "skipped_reason_histogram": skipped_hist,
            "scan_truncated_runs": scan_truncated,
            "data_quality_warning_messages": quality_warning_messages,
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


def _month_bounds(month: str) -> tuple[date, date]:
    try:
        year, month_num = month.split("-")
        year_int = int(year)
        month_int = int(month_num)
        if month_int < 1 or month_int > 12:
            raise ValueError
    except ValueError as exc:
        raise APIError(code="invalid_month", message="month must be YYYY-MM.") from exc
    first = date(year_int, month_int, 1)
    last = date(year_int, month_int, calendar.monthrange(year_int, month_int)[1])
    return first, last


def _resolve_month(month: str | None) -> str:
    if month is None:
        today = _utc_now().date()
        return f"{today.year:04d}-{today.month:02d}"
    return month


def build_monthly_report_content(
    *,
    month: str,
    rows: list[PaperRun],
    bundle_id: int | None,
    policy_id: int | None,
) -> dict[str, Any]:
    first_day, last_day = _month_bounds(month)
    if not rows:
        return {
            "month": month,
            "bundle_id": bundle_id,
            "policy_id": policy_id,
            "window": {"start_date": first_day.isoformat(), "end_date": last_day.isoformat()},
            "summary": {
                "runs": 0,
                "trading_days": 0,
                "entries": 0,
                "exits": 0,
                "net_pnl": 0.0,
                "costs": 0.0,
                "max_drawdown": 0.0,
                "best_day": None,
                "worst_day": None,
            },
            "daily_breakdown": [],
            "explainability": {
                "selected_reason_histogram": {},
                "skipped_reason_histogram": {},
            },
            "links": {"paper_run_ids": []},
        }

    selected_hist = _histogram(rows, "selected_reason_histogram")
    skipped_hist = _histogram(rows, "skipped_reason_histogram")
    total_entries = int(sum(_safe_int((row.summary_json or {}).get("positions_opened"), 0) for row in rows))
    total_exits = int(sum(_safe_int((row.summary_json or {}).get("positions_closed"), 0) for row in rows))
    net_pnl = float(sum(_safe_float((row.summary_json or {}).get("net_pnl"), 0.0) for row in rows))
    costs = float(sum(_safe_float((row.summary_json or {}).get("total_cost"), 0.0) for row in rows))

    per_day: dict[str, dict[str, Any]] = defaultdict(lambda: {"runs": 0, "net_pnl": 0.0, "costs": 0.0})
    for row in rows:
        key = row.asof_ts.date().isoformat()
        per_day[key]["runs"] += 1
        per_day[key]["net_pnl"] += _safe_float((row.summary_json or {}).get("net_pnl"), 0.0)
        per_day[key]["costs"] += _safe_float((row.summary_json or {}).get("total_cost"), 0.0)

    daily_breakdown = [
        {
            "date": day,
            "runs": int(values["runs"]),
            "net_pnl": float(values["net_pnl"]),
            "costs": float(values["costs"]),
        }
        for day, values in sorted(per_day.items(), key=lambda item: item[0])
    ]

    cumulative = 1_000_000.0
    peak = cumulative
    max_dd = 0.0
    for item in daily_breakdown:
        cumulative += float(item["net_pnl"])
        peak = max(peak, cumulative)
        dd = (cumulative / peak - 1.0) if peak > 0 else 0.0
        max_dd = min(max_dd, dd)

    best_day = max(daily_breakdown, key=lambda item: float(item["net_pnl"]))
    worst_day = min(daily_breakdown, key=lambda item: float(item["net_pnl"]))
    return {
        "month": month,
        "bundle_id": bundle_id,
        "policy_id": policy_id,
        "window": {"start_date": first_day.isoformat(), "end_date": last_day.isoformat()},
        "summary": {
            "runs": len(rows),
            "trading_days": len(daily_breakdown),
            "entries": total_entries,
            "exits": total_exits,
            "net_pnl": net_pnl,
            "costs": costs,
            "max_drawdown": float(max_dd),
            "best_day": best_day,
            "worst_day": worst_day,
        },
        "daily_breakdown": daily_breakdown,
        "explainability": {
            "selected_reason_histogram": selected_hist,
            "skipped_reason_histogram": skipped_hist,
        },
        "links": {"paper_run_ids": [int(row.id) for row in rows if row.id is not None]},
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
        existing = _latest_existing_daily_report(
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

    existing = _latest_existing_daily_report(
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


def generate_monthly_report(
    session: Session,
    settings: Settings,
    *,
    month: str | None = None,
    bundle_id: int | None = None,
    policy_id: int | None = None,
    overwrite: bool = True,
) -> MonthlyReport:
    resolved_month = _resolve_month(month)
    first_day, last_day = _month_bounds(resolved_month)
    if not overwrite:
        existing = _latest_existing_monthly_report(
            session,
            month=resolved_month,
            bundle_id=bundle_id,
            policy_id=policy_id,
        )
        if existing is not None:
            return existing

    rows = _runs_for_month(
        session,
        month_start=first_day,
        month_end=last_day,
        bundle_id=bundle_id,
        policy_id=policy_id,
    )
    content = build_monthly_report_content(
        month=resolved_month,
        rows=rows,
        bundle_id=bundle_id,
        policy_id=policy_id,
    )
    existing = _latest_existing_monthly_report(
        session,
        month=resolved_month,
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

    row = MonthlyReport(
        month=resolved_month,
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


def list_monthly_reports(
    session: Session,
    *,
    month: str | None,
    bundle_id: int | None,
    policy_id: int | None,
) -> list[MonthlyReport]:
    stmt = select(MonthlyReport).order_by(MonthlyReport.month.desc(), MonthlyReport.created_at.desc())
    if month is not None:
        stmt = stmt.where(MonthlyReport.month == month)
    if bundle_id is not None:
        stmt = stmt.where(MonthlyReport.bundle_id == bundle_id)
    if policy_id is not None:
        stmt = stmt.where(MonthlyReport.policy_id == policy_id)
    return list(session.exec(stmt).all())


def get_daily_report(session: Session, report_id: int) -> DailyReport:
    row = session.get(DailyReport, report_id)
    if row is None:
        raise APIError(code="not_found", message="Daily report not found", status_code=404)
    return row


def get_monthly_report(session: Session, report_id: int) -> MonthlyReport:
    row = session.get(MonthlyReport, report_id)
    if row is None:
        raise APIError(code="not_found", message="Monthly report not found", status_code=404)
    return row


def _extract_equity_series(rows: list[PaperRun]) -> tuple[list[datetime], list[float], list[float]]:
    if not rows:
        return [], [], []
    rows = sorted(rows, key=lambda item: item.asof_ts)
    first_summary = rows[0].summary_json if isinstance(rows[0].summary_json, dict) else {}
    equity = _safe_float(first_summary.get("equity_before"), 1_000_000.0)
    points_t: list[datetime] = []
    points_equity: list[float] = []
    for row in rows:
        summary = row.summary_json if isinstance(row.summary_json, dict) else {}
        eq_after = _safe_float(summary.get("equity_after"), equity + _safe_float(summary.get("net_pnl"), 0.0))
        points_t.append(row.asof_ts)
        points_equity.append(eq_after)
        equity = eq_after
    if not points_equity:
        return [], [], []
    arr = np.array(points_equity, dtype=float)
    running_max = np.maximum.accumulate(arr)
    dd = np.where(running_max > 0, (arr / running_max) - 1.0, 0.0)
    return points_t, points_equity, dd.tolist()


def _chart_png(
    *,
    title: str,
    x_values: list[datetime],
    equity: list[float],
    drawdown: list[float],
) -> BytesIO:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    ax_eq, ax_dd = axes
    ax_eq.plot(x_values, equity, color="#1f77b4", linewidth=1.8)
    ax_eq.set_title(title)
    ax_eq.grid(alpha=0.2)
    ax_eq.set_ylabel("Equity")
    ax_eq.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))

    ax_dd.plot(x_values, drawdown, color="#d62728", linewidth=1.4)
    ax_dd.fill_between(x_values, drawdown, 0.0, color="#d62728", alpha=0.15)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.grid(alpha=0.2)
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    out = BytesIO()
    fig.tight_layout()
    fig.savefig(out, format="png", dpi=150)
    plt.close(fig)
    out.seek(0)
    return out


def _draw_header(pdf: canvas.Canvas, *, title: str, subtitle: str) -> float:
    width, height = A4
    y = height - 42
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(36, y, title)
    y -= 18
    pdf.setFont("Helvetica", 10)
    pdf.setFillGray(0.35)
    pdf.drawString(36, y, subtitle)
    pdf.setFillGray(0.0)
    return y - 20


def _draw_metric_row(pdf: canvas.Canvas, y: float, metrics: list[tuple[str, str]]) -> float:
    x = 36
    col_width = (A4[0] - 72) / max(1, len(metrics))
    for label, value in metrics:
        pdf.roundRect(x, y - 28, col_width - 8, 32, 6, stroke=1, fill=0)
        pdf.setFont("Helvetica", 8)
        pdf.setFillGray(0.35)
        pdf.drawString(x + 8, y - 12, label)
        pdf.setFillGray(0.0)
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(x + 8, y - 24, value)
        x += col_width
    return y - 42


def _draw_reason_block(
    pdf: canvas.Canvas,
    *,
    y: float,
    title: str,
    reasons: dict[str, int],
) -> float:
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(36, y, title)
    y -= 12
    pdf.setFont("Helvetica", 8)
    if not reasons:
        pdf.setFillGray(0.35)
        pdf.drawString(36, y, "No entries")
        pdf.setFillGray(0.0)
        return y - 14
    for idx, (reason, count) in enumerate(
        sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:8],
        start=1,
    ):
        pdf.drawString(36, y, f"{idx}. {reason}: {count}")
        y -= 10
    return y - 4


def _draw_signal_table(
    pdf: canvas.Canvas,
    *,
    y: float,
    rows: list[tuple[str, str, str]],
) -> float:
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(36, y, "Top Executed Signals")
    y -= 14
    pdf.setFont("Helvetica-Bold", 8)
    pdf.drawString(36, y, "Symbol")
    pdf.drawString(180, y, "Side")
    pdf.drawString(240, y, "Instrument")
    y -= 10
    pdf.setFont("Helvetica", 8)
    if not rows:
        pdf.setFillGray(0.35)
        pdf.drawString(36, y, "No executed signals recorded.")
        pdf.setFillGray(0.0)
        return y - 12
    for symbol, side, instrument in rows[:12]:
        pdf.drawString(36, y, symbol)
        pdf.drawString(180, y, side)
        pdf.drawString(240, y, instrument)
        y -= 10
    return y - 6


def _report_runs_from_content(session: Session, content: dict[str, Any]) -> list[PaperRun]:
    links = content.get("links", {}) if isinstance(content, dict) else {}
    run_ids = links.get("paper_run_ids") if isinstance(links, dict) else []
    valid_ids = [int(item) for item in run_ids if isinstance(item, int)]
    if not valid_ids:
        return []
    rows = session.exec(
        select(PaperRun).where(PaperRun.id.in_(valid_ids)).order_by(PaperRun.asof_ts.asc())
    ).all()
    return list(rows)


def render_daily_report_pdf(
    session: Session,
    *,
    report: DailyReport,
) -> bytes:
    content = report.content_json if isinstance(report.content_json, dict) else {}
    summary = content.get("summary", {}) if isinstance(content, dict) else {}
    explainability = content.get("explainability", {}) if isinstance(content, dict) else {}
    risk = content.get("risk", {}) if isinstance(content, dict) else {}
    rows = _report_runs_from_content(session, content)

    x_values, equity, drawdown = _extract_equity_series(rows)
    if not x_values:
        x_values = [datetime.combine(report.date, datetime.min.time(), tzinfo=timezone.utc)]
        equity = [_safe_float(risk.get("equity_end"), 1_000_000.0)]
        drawdown = [_safe_float(summary.get("drawdown"), 0.0)]
    chart = _chart_png(
        title=f"Daily Tear Sheet - {report.date.isoformat()}",
        x_values=x_values,
        equity=equity,
        drawdown=drawdown,
    )

    signal_rows: list[tuple[str, str, str]] = []
    for run in rows:
        run_summary = run.summary_json if isinstance(run.summary_json, dict) else {}
        selected = run_summary.get("selected_signals", [])
        if not isinstance(selected, list):
            continue
        for item in selected:
            if not isinstance(item, dict):
                continue
            signal_rows.append(
                (
                    str(item.get("symbol", "-")),
                    str(item.get("side", "-")),
                    str(item.get("instrument_kind", "-")),
                )
            )

    out = BytesIO()
    pdf = canvas.Canvas(out, pagesize=A4)
    y = _draw_header(
        pdf,
        title="Atlas Daily Tear Sheet",
        subtitle=(
            f"Date: {report.date.isoformat()}  |  Bundle: {report.bundle_id or '-'}  |  "
            f"Policy: {report.policy_id or '-'}"
        ),
    )
    y = _draw_metric_row(
        pdf,
        y,
        [
            ("Net PnL", f"{_safe_float(summary.get('net_pnl')):,.2f}"),
            ("Costs", f"{_safe_float(summary.get('costs')):,.2f}"),
            ("Drawdown", f"{_safe_float(summary.get('drawdown')):.2%}"),
            ("Signals Source", str(summary.get("signals_source", "-"))),
        ],
    )
    pdf.drawImage(ImageReader(chart), 36, y - 220, width=A4[0] - 72, height=210, preserveAspectRatio=True)
    y -= 232
    y = _draw_signal_table(pdf, y=y, rows=signal_rows)
    y = _draw_reason_block(
        pdf,
        y=y,
        title="Top Selected Reasons",
        reasons=(
            explainability.get("selected_reason_histogram", {})
            if isinstance(explainability, dict)
            else {}
        ),
    )
    y = _draw_reason_block(
        pdf,
        y=y,
        title="Top Skipped Reasons",
        reasons=(
            explainability.get("skipped_reason_histogram", {})
            if isinstance(explainability, dict)
            else {}
        ),
    )
    if y < 72:
        pdf.showPage()
        y = _draw_header(
            pdf,
            title="Atlas Daily Tear Sheet",
            subtitle="Continued",
        )
    pdf.setFont("Helvetica", 8)
    pdf.setFillGray(0.35)
    pdf.drawString(
        36,
        26,
        "Research + paper trading only. Not financial advice. Past performance does not guarantee future results.",
    )
    pdf.setFillGray(0.0)
    pdf.save()
    out.seek(0)
    return out.read()


def render_monthly_report_pdf(
    session: Session,
    *,
    report: MonthlyReport,
) -> bytes:
    content = report.content_json if isinstance(report.content_json, dict) else {}
    summary = content.get("summary", {}) if isinstance(content, dict) else {}
    explainability = content.get("explainability", {}) if isinstance(content, dict) else {}
    daily = content.get("daily_breakdown", []) if isinstance(content, dict) else []
    x_values: list[datetime] = []
    equity: list[float] = []
    current = 1_000_000.0
    for item in daily if isinstance(daily, list) else []:
        if not isinstance(item, dict):
            continue
        dt_value = item.get("date")
        if not isinstance(dt_value, str):
            continue
        try:
            asof = datetime.fromisoformat(dt_value).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        current += _safe_float(item.get("net_pnl"), 0.0)
        x_values.append(asof)
        equity.append(current)
    if not x_values:
        month_start, _ = _month_bounds(report.month)
        x_values = [datetime.combine(month_start, datetime.min.time(), tzinfo=timezone.utc)]
        equity = [1_000_000.0 + _safe_float(summary.get("net_pnl"), 0.0)]
    arr = np.array(equity, dtype=float)
    peaks = np.maximum.accumulate(arr)
    drawdown = np.where(peaks > 0, (arr / peaks) - 1.0, 0.0).tolist()
    chart = _chart_png(
        title=f"Monthly Tear Sheet - {report.month}",
        x_values=x_values,
        equity=equity,
        drawdown=drawdown,
    )

    out = BytesIO()
    pdf = canvas.Canvas(out, pagesize=A4)
    y = _draw_header(
        pdf,
        title="Atlas Monthly Tear Sheet",
        subtitle=(
            f"Month: {report.month}  |  Bundle: {report.bundle_id or '-'}  |  "
            f"Policy: {report.policy_id or '-'}"
        ),
    )
    y = _draw_metric_row(
        pdf,
        y,
        [
            ("Net PnL", f"{_safe_float(summary.get('net_pnl')):,.2f}"),
            ("Costs", f"{_safe_float(summary.get('costs')):,.2f}"),
            ("Max Drawdown", f"{_safe_float(summary.get('max_drawdown')):.2%}"),
            ("Trading Days", str(_safe_int(summary.get("trading_days"), 0))),
        ],
    )
    pdf.drawImage(ImageReader(chart), 36, y - 220, width=A4[0] - 72, height=210, preserveAspectRatio=True)
    y -= 230
    y = _draw_reason_block(
        pdf,
        y=y,
        title="Top Selected Reasons",
        reasons=(
            explainability.get("selected_reason_histogram", {})
            if isinstance(explainability, dict)
            else {}
        ),
    )
    y = _draw_reason_block(
        pdf,
        y=y,
        title="Top Skipped Reasons",
        reasons=(
            explainability.get("skipped_reason_histogram", {})
            if isinstance(explainability, dict)
            else {}
        ),
    )
    pdf.setFont("Helvetica", 8)
    pdf.setFillGray(0.35)
    pdf.drawString(
        36,
        26,
        "Research + paper trading only. Not financial advice. Past performance does not guarantee future results.",
    )
    pdf.setFillGray(0.0)
    pdf.save()
    out.seek(0)
    return out.read()
