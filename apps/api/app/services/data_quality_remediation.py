from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DataQualityReport, DatasetBundle, PaperPosition, PaperRun
from app.services.data_quality import get_latest_data_quality_report
from app.services.data_store import DataStore


DEFAULT_SYMBOL_GATE_CODES = {
    "corporate_action_anomaly",
    "gap_exceeds_threshold",
    "return_outliers",
}

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_dates_from_text(value: Any) -> list[str]:
    if not isinstance(value, str):
        return []
    return DATE_RE.findall(value)


def _date_range_from_dates(dates: list[str], *, source: str) -> dict[str, Any] | None:
    clean = sorted({item for item in dates if isinstance(item, str) and DATE_RE.fullmatch(item)})
    if not clean:
        return None
    return {
        "start": clean[0],
        "end": clean[-1],
        "sample": clean[:10],
        "source": source,
    }


def _merge_date_ranges(ranges: list[dict[str, Any]]) -> dict[str, Any] | None:
    starts = [str(item.get("start")) for item in ranges if item.get("start")]
    ends = [str(item.get("end")) for item in ranges if item.get("end")]
    if not starts or not ends:
        return None
    samples: list[str] = []
    for item in ranges:
        for date_value in item.get("sample", []):
            token = str(date_value)
            if token not in samples:
                samples.append(token)
    return {
        "start": min(starts),
        "end": max(ends),
        "sample": sorted(samples)[:10],
        "source": "merged",
    }


def _issue_date_range_from_details(issue: dict[str, Any]) -> dict[str, Any] | None:
    details = issue.get("details", {})
    dates: list[str] = []
    if isinstance(details, dict):
        for key in (
            "missing_dates",
            "missing_symbols_dates",
            "affected_dates",
            "outlier_dates",
            "jump_dates",
        ):
            raw = details.get(key)
            if isinstance(raw, list):
                dates.extend(str(item) for item in raw)
        for key in (
            "last_bar_day_ist",
            "expected_latest_trading_day",
            "start_date",
            "end_date",
        ):
            dates.extend(_parse_dates_from_text(details.get(key)))
    dates.extend(_parse_dates_from_text(issue.get("message")))
    return _date_range_from_dates(dates, source="issue_details")


def _return_anomaly_range(
    frame: pd.DataFrame,
    *,
    issue: dict[str, Any],
) -> dict[str, Any] | None:
    if frame.empty or "close" not in frame.columns or "datetime" not in frame.columns:
        return None
    if len(frame) < 10:
        return None

    ordered = frame.copy().sort_values("datetime").reset_index(drop=True)
    ordered["datetime"] = pd.to_datetime(ordered["datetime"], utc=True)
    close = pd.Series(ordered["close"], dtype=float)
    returns = close.pct_change()
    code = str(issue.get("code", ""))
    details = issue.get("details", {}) if isinstance(issue.get("details"), dict) else {}

    if code == "corporate_action_anomaly":
        threshold = _safe_float(details.get("jump_threshold"), 0.35)
        mask = returns.abs() >= threshold
    elif code == "return_outliers":
        threshold = _safe_float(details.get("zscore_threshold"), 8.0)
        valid_returns = returns.dropna()
        if valid_returns.empty:
            return None
        std = float(valid_returns.std(ddof=0))
        if std <= 1e-12:
            return None
        z = ((valid_returns - valid_returns.mean()) / std).abs()
        mask = pd.Series(False, index=returns.index)
        mask.loc[z.index] = z > threshold
    else:
        return None

    dates = [
        item.date().isoformat()
        for item in ordered.loc[mask.fillna(False), "datetime"].tolist()
        if pd.notna(item)
    ]
    return _date_range_from_dates(dates, source="computed_ohlcv")


def _expand_symbol_issues(issues: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    report_level: list[dict[str, Any]] = []

    for issue in issues:
        if not isinstance(issue, dict):
            continue
        symbol = str(issue.get("symbol", "")).strip().upper()
        if symbol:
            by_symbol[symbol].append(dict(issue))
            continue

        details = issue.get("details", {}) if isinstance(issue.get("details"), dict) else {}
        sample_symbols: list[str] = []
        for key in (
            "inactive_symbols_sample",
            "missing_symbols_sample",
            "low_confidence_symbols_sample",
        ):
            raw = details.get(key)
            if isinstance(raw, list):
                sample_symbols.extend(str(item).strip().upper() for item in raw if str(item).strip())
        if sample_symbols:
            for item in sorted(set(sample_symbols)):
                expanded = dict(issue)
                expanded["symbol"] = item
                expanded["expanded_from_report_level"] = True
                by_symbol[item].append(expanded)
        else:
            report_level.append(dict(issue))

    return dict(by_symbol), report_level


def _recommended_action(codes: set[str]) -> str:
    if codes & {"missing_ohlcv", "gap_exceeds_threshold"}:
        return "backfill_ohlcv_history"
    if "corporate_action_anomaly" in codes:
        return "verify_corporate_action_adjustment"
    if "return_outliers" in codes:
        return "review_outlier_bars_or_add_exception"
    if codes & {
        "stale_data",
        "inactive_symbols_detected",
        "coverage_below_warn_threshold",
        "coverage_below_fail_threshold",
        "low_confidence_latest_day",
        "fallback_source_live_mode",
    }:
        return "refresh_or_replace_latest_source_data"
    if codes & {"invalid_ohlc_ranges", "duplicate_timestamps", "non_monotonic_timestamps"}:
        return "repair_parquet_rows"
    return "manual_review"


def _priority(
    *,
    severities: set[str],
    blocked_by_symbol_gate: bool,
    currently_open_position: bool,
    selected_in_latest_run: bool,
) -> str:
    if "FAIL" in severities:
        return "P0"
    if currently_open_position or selected_in_latest_run:
        return "P1"
    if blocked_by_symbol_gate:
        return "P2"
    return "P3"


def _issue_summary(issue: dict[str, Any], date_range: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "code": str(issue.get("code", "")),
        "severity": str(issue.get("severity", "")),
        "message": str(issue.get("message", "")),
        "date_range": date_range,
        "details": dict(issue.get("details", {})) if isinstance(issue.get("details"), dict) else {},
        "expanded_from_report_level": bool(issue.get("expanded_from_report_level", False)),
    }


def _latest_selected_symbols(session: Session, *, bundle_id: int | None) -> set[str]:
    stmt = select(PaperRun).order_by(PaperRun.asof_ts.desc(), PaperRun.id.desc())
    if bundle_id is not None:
        stmt = stmt.where(PaperRun.bundle_id == bundle_id)
    row = session.exec(stmt.limit(1)).first()
    if row is None or not isinstance(row.summary_json, dict):
        return set()
    selected = row.summary_json.get("selected_signals", [])
    if not isinstance(selected, list):
        return set()
    symbols: set[str] = set()
    for item in selected:
        if isinstance(item, dict):
            token = str(item.get("underlying_symbol") or item.get("symbol") or "").strip().upper()
            if token:
                symbols.add(token)
    return symbols


def build_data_quality_remediation_payload(
    report: DataQualityReport,
    *,
    bundle_name: str | None = None,
    frame_lookup: Callable[[str], pd.DataFrame] | None = None,
    open_position_symbols: set[str] | None = None,
    latest_selected_symbols: set[str] | None = None,
    symbol_gate_codes: set[str] | None = None,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    gate_codes = set(symbol_gate_codes or DEFAULT_SYMBOL_GATE_CODES)
    issues = [dict(item) for item in (report.issues_json or []) if isinstance(item, dict)]
    by_symbol, report_level_issues = _expand_symbol_issues(issues)
    open_symbols = {str(item).upper() for item in (open_position_symbols or set())}
    latest_symbols = {str(item).upper() for item in (latest_selected_symbols or set())}

    rows: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    priority_counts: Counter[str] = Counter()
    code_counts: Counter[str] = Counter()
    blocked_count = 0

    for symbol in sorted(by_symbol):
        symbol_issues = by_symbol[symbol]
        codes = {str(item.get("code", "")) for item in symbol_issues if str(item.get("code", ""))}
        severities = {
            str(item.get("severity", "")).upper()
            for item in symbol_issues
            if str(item.get("severity", ""))
        }
        for code in codes:
            code_counts[code] += sum(1 for item in symbol_issues if item.get("code") == code)

        frame: pd.DataFrame | None = None
        issue_rows: list[dict[str, Any]] = []
        date_ranges: list[dict[str, Any]] = []
        for issue in symbol_issues:
            date_range = _issue_date_range_from_details(issue)
            if date_range is None and frame_lookup is not None:
                if frame is None:
                    try:
                        frame = frame_lookup(symbol)
                    except Exception:  # noqa: BLE001
                        frame = pd.DataFrame()
                date_range = _return_anomaly_range(frame, issue=issue)
            if date_range is not None:
                date_ranges.append(date_range)
            issue_rows.append(_issue_summary(issue, date_range))

        blocked_by_gate = bool(codes & gate_codes)
        blocked_count += int(blocked_by_gate)
        action = _recommended_action(codes)
        priority = _priority(
            severities=severities,
            blocked_by_symbol_gate=blocked_by_gate,
            currently_open_position=symbol in open_symbols,
            selected_in_latest_run=symbol in latest_symbols,
        )
        action_counts[action] += 1
        priority_counts[priority] += 1

        rows.append(
            {
                "symbol": symbol,
                "priority": priority,
                "recommended_action": action,
                "blocked_by_symbol_gate": blocked_by_gate,
                "currently_open_position": symbol in open_symbols,
                "selected_in_latest_run": symbol in latest_symbols,
                "severity": "FAIL" if "FAIL" in severities else "WARN",
                "issue_count": len(symbol_issues),
                "codes": sorted(codes),
                "code_counts": dict(Counter(str(item.get("code", "")) for item in symbol_issues)),
                "latest_affected_range": _merge_date_ranges(date_ranges),
                "issues": issue_rows,
            }
        )

    priority_rank = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    rows.sort(
        key=lambda item: (
            priority_rank.get(str(item["priority"]), 9),
            -int(item["issue_count"]),
            str(item["symbol"]),
        )
    )

    return {
        "generated_at": (generated_at or _utc_now()).isoformat(),
        "report": {
            "id": report.id,
            "bundle_id": report.bundle_id,
            "bundle_name": bundle_name,
            "timeframe": report.timeframe,
            "status": report.status,
            "created_at": report.created_at.isoformat(),
            "last_bar_ts": report.last_bar_ts.isoformat() if report.last_bar_ts else None,
            "coverage_pct": float(report.coverage_pct),
            "total_symbols": int(report.total_symbols),
            "issues_count": len(issues),
        },
        "summary": {
            "symbols_with_issues": len(rows),
            "blocked_by_symbol_gate_count": blocked_count,
            "priority_counts": dict(priority_counts),
            "action_counts": dict(action_counts),
            "code_counts": dict(code_counts),
            "report_level_issues_count": len(report_level_issues),
        },
        "symbol_gate_codes": sorted(gate_codes),
        "symbols": rows,
        "report_level_issues": report_level_issues,
    }


def remediation_markdown(payload: dict[str, Any], *, max_symbols: int = 80) -> str:
    report = payload.get("report", {})
    summary = payload.get("summary", {})
    lines = [
        "# Data Quality Remediation Report",
        "",
        f"- Report id: {report.get('id')}",
        f"- Bundle: {report.get('bundle_name') or report.get('bundle_id')}",
        f"- Timeframe: {report.get('timeframe')}",
        f"- Status: {report.get('status')}",
        f"- Coverage: {float(report.get('coverage_pct', 0.0)):.2f}%",
        f"- Issues: {report.get('issues_count')}",
        f"- Symbols with issues: {summary.get('symbols_with_issues')}",
        f"- Blocked by symbol gate: {summary.get('blocked_by_symbol_gate_count')}",
        "",
        "## Action Counts",
        "",
    ]
    for action, count in sorted(dict(summary.get("action_counts", {})).items()):
        lines.append(f"- {action}: {count}")
    lines.extend(["", "## Priority Counts", ""])
    for priority, count in sorted(dict(summary.get("priority_counts", {})).items()):
        lines.append(f"- {priority}: {count}")
    lines.extend(["", "## Top Symbols", ""])
    lines.append("| Priority | Symbol | Action | Codes | Date Range | Gate |")
    lines.append("|---|---:|---|---|---|---|")
    for row in list(payload.get("symbols", []))[:max(1, int(max_symbols))]:
        date_range = row.get("latest_affected_range") or {}
        range_text = ""
        if date_range:
            range_text = f"{date_range.get('start')} to {date_range.get('end')}"
        gate = "yes" if row.get("blocked_by_symbol_gate") else "no"
        lines.append(
            "| "
            f"{row.get('priority')} | "
            f"{row.get('symbol')} | "
            f"{row.get('recommended_action')} | "
            f"{', '.join(row.get('codes', []))} | "
            f"{range_text or 'unknown'} | "
            f"{gate} |"
        )
    return "\n".join(lines) + "\n"


def write_data_quality_remediation_files(
    payload: dict[str, Any],
    *,
    output_dir: str | Path = "data/reports/data-quality",
) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report = payload.get("report", {})
    bundle_id = report.get("bundle_id") or "none"
    timeframe = str(report.get("timeframe") or "1d").replace("/", "_")
    report_id = report.get("id") or "latest"
    stem = f"data-quality-remediation-bundle-{bundle_id}-{timeframe}-report-{report_id}"
    json_path = target_dir / f"{stem}.json"
    md_path = target_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(remediation_markdown(payload), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def generate_data_quality_remediation_report(
    *,
    session: Session,
    settings: Settings,
    bundle_id: int,
    timeframe: str = "1d",
    report_id: int | None = None,
    write_files: bool = False,
    output_dir: str | Path = "data/reports/data-quality",
    store: DataStore | None = None,
) -> dict[str, Any]:
    if report_id is not None:
        report = session.get(DataQualityReport, int(report_id))
        if report is None:
            raise APIError(code="not_found", message="Data quality report not found", status_code=404)
    else:
        report = get_latest_data_quality_report(
            session,
            bundle_id=int(bundle_id),
            timeframe=str(timeframe),
        )
        if report is None:
            raise APIError(code="not_found", message="No data quality report found", status_code=404)

    bundle = session.get(DatasetBundle, int(bundle_id))
    resolved_store = store or DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )
    open_symbols = {
        str(row.metadata_json.get("underlying_symbol", row.symbol)).upper()
        if isinstance(row.metadata_json, dict)
        else str(row.symbol).upper()
        for row in session.exec(select(PaperPosition)).all()
    }
    latest_symbols = _latest_selected_symbols(session, bundle_id=int(bundle_id))

    def frame_lookup(symbol: str) -> pd.DataFrame:
        return resolved_store.load_ohlcv(
            symbol=symbol,
            timeframe=str(report.timeframe or timeframe),
            session=session,
            adjustment_mode=settings.data_adjustment_mode,
        )

    payload = build_data_quality_remediation_payload(
        report,
        bundle_name=bundle.name if bundle is not None else None,
        frame_lookup=frame_lookup,
        open_position_symbols=open_symbols,
        latest_selected_symbols=latest_symbols,
    )
    if write_files:
        payload["files"] = write_data_quality_remediation_files(payload, output_dir=output_dir)
    return payload
