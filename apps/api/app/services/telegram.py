from __future__ import annotations

import json
from typing import Any
from urllib import error, request

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import DailyReport, MonthlyReport

DISCLAIMER = "Research + paper trading only. Not financial advice."
MAX_TELEGRAM_TEXT_CHARS = 3900


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


def _format_money(value: Any) -> str:
    return f"{_safe_float(value):,.2f}"


def _format_pct(value: Any) -> str:
    return f"{_safe_float(value):.2%}"


def _format_scale(value: Any) -> str:
    return f"{_safe_float(value, 1.0) * 100:.1f}%"


def _mask_secret(value: str | None) -> str | None:
    if not value:
        return None
    clean = str(value).strip()
    if len(clean) <= 6:
        return "***"
    return f"{clean[:3]}...{clean[-3:]}"


def _compact_items(source: Any, *, limit: int = 3) -> list[tuple[str, int]]:
    if not isinstance(source, dict):
        return []
    rows = [
        (str(key), _safe_int(value))
        for key, value in source.items()
        if str(key).strip() and _safe_int(value) > 0
    ]
    return sorted(rows, key=lambda item: item[1], reverse=True)[:limit]


def _add_histogram_lines(lines: list[str], title: str, source: Any) -> None:
    items = _compact_items(source)
    if not items:
        lines.append(f"{title}: none")
        return
    lines.append(f"{title}:")
    for reason, count in items:
        lines.append(f"- {reason}: {count}")


def _truncate_message(text: str) -> str:
    clean = text.strip()
    if len(clean) <= MAX_TELEGRAM_TEXT_CHARS:
        return clean
    suffix = "\n\n[Truncated in Telegram. Open Atlas for the full report.]"
    return f"{clean[: MAX_TELEGRAM_TEXT_CHARS - len(suffix)]}{suffix}"


def telegram_status_payload(settings: Settings) -> dict[str, Any]:
    token_configured = bool(settings.telegram_bot_token)
    chat_configured = bool(settings.telegram_chat_id)
    missing = []
    if not token_configured:
        missing.append("ATLAS_TELEGRAM_BOT_TOKEN")
    if not chat_configured:
        missing.append("ATLAS_TELEGRAM_CHAT_ID")
    if not settings.telegram_enabled:
        missing.append("ATLAS_TELEGRAM_ENABLED=true")
    return {
        "enabled": bool(settings.telegram_enabled),
        "configured": token_configured and chat_configured,
        "ready": bool(settings.telegram_enabled and token_configured and chat_configured),
        "send_reports": bool(settings.telegram_send_reports),
        "bot_token_configured": token_configured,
        "chat_id_configured": chat_configured,
        "chat_id_hint": _mask_secret(settings.telegram_chat_id),
        "missing": missing,
    }


def _require_telegram_settings(settings: Settings) -> tuple[str, str]:
    if not settings.telegram_enabled:
        raise APIError(
            code="telegram_disabled",
            message="Telegram notifications are disabled.",
            details={"required": ["ATLAS_TELEGRAM_ENABLED=true"]},
        )
    missing = []
    if not settings.telegram_bot_token:
        missing.append("ATLAS_TELEGRAM_BOT_TOKEN")
    if not settings.telegram_chat_id:
        missing.append("ATLAS_TELEGRAM_CHAT_ID")
    if missing:
        raise APIError(
            code="telegram_not_configured",
            message="Telegram bot token or chat ID is not configured.",
            details={"missing": missing},
        )
    return str(settings.telegram_bot_token), str(settings.telegram_chat_id)


def send_telegram_message(settings: Settings, text: str) -> dict[str, Any]:
    token, chat_id = _require_telegram_settings(settings)
    message = _truncate_message(text)
    if not message:
        raise APIError(code="empty_telegram_message", message="Telegram message cannot be empty.")

    payload = {
        "chat_id": chat_id,
        "text": message,
        "disable_web_page_preview": True,
    }
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=body,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "Atlas/0.1",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(settings.telegram_timeout_seconds)) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise APIError(
            code="telegram_send_failed",
            message="Telegram rejected the message.",
            status_code=502,
            details={"status_code": exc.code, "body": detail},
        ) from exc
    except (TimeoutError, error.URLError, OSError) as exc:
        raise APIError(
            code="telegram_unavailable",
            message="Could not reach Telegram.",
            status_code=502,
            details={"reason": exc.__class__.__name__},
        ) from exc

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise APIError(
            code="telegram_bad_response",
            message="Telegram returned an invalid response.",
            status_code=502,
        ) from exc

    if not bool(decoded.get("ok")):
        raise APIError(
            code="telegram_send_failed",
            message="Telegram rejected the message.",
            status_code=502,
            details={"description": str(decoded.get("description", ""))[:500]},
        )

    result = decoded.get("result", {}) if isinstance(decoded, dict) else {}
    message_id = result.get("message_id") if isinstance(result, dict) else None
    return {
        "status": "SENT",
        "message_id": message_id,
        "chat_id_hint": _mask_secret(chat_id),
    }


def try_send_telegram_message(settings: Settings, text: str) -> dict[str, Any]:
    try:
        return send_telegram_message(settings, text)
    except APIError as exc:
        payload: dict[str, Any] = {
            "status": "FAILED",
            "error": {"code": exc.code, "message": exc.message},
        }
        if exc.details is not None:
            payload["error"]["details"] = exc.details
        return payload


def format_daily_report_message(report: DailyReport) -> str:
    content = report.content_json if isinstance(report.content_json, dict) else {}
    summary = content.get("summary", {}) if isinstance(content, dict) else {}
    explainability = content.get("explainability", {}) if isinstance(content, dict) else {}
    risk = content.get("risk", {}) if isinstance(content, dict) else {}
    confidence = content.get("confidence_gate", {}) if isinstance(content, dict) else {}
    confidence_scaling = (
        content.get("confidence_risk_scaling", {}) if isinstance(content, dict) else {}
    )
    selected_reasons = (
        explainability.get("selected_reason_histogram", {})
        if isinstance(explainability, dict)
        else {}
    )
    skipped_reasons = (
        explainability.get("skipped_reason_histogram", {})
        if isinstance(explainability, dict)
        else {}
    )

    lines = [
        "Atlas Daily Report",
        f"Date: {report.date.isoformat()}",
        f"Bundle: {report.bundle_id or '-'} | Policy: {report.policy_id or '-'}",
        "",
        (
            f"Runs: {_safe_int(summary.get('runs'))} | "
            f"Entries: {_safe_int(summary.get('entries'))} | "
            f"Exits: {_safe_int(summary.get('exits'))}"
        ),
        (
            f"Open positions: {_safe_int(summary.get('positions_open'))} | "
            f"Peak positions: {_safe_int(risk.get('positions_peak'))}"
        ),
        f"Net PnL: {_format_money(summary.get('net_pnl'))} | Costs: {_format_money(summary.get('costs'))}",
        (
            f"Drawdown: {_format_pct(summary.get('drawdown'))} | "
            f"Avg exposure: {_format_pct(risk.get('avg_exposure'))}"
        ),
        f"Mode: {summary.get('mode', 'LIVE')} | Regime: {summary.get('regime', '-')}",
        (
            "Kill switch: ACTIVE"
            if bool(summary.get("kill_switch_active"))
            else "Kill switch: clear"
        ),
        (
            "Safe mode: ACTIVE"
            if bool(summary.get("safe_mode_active"))
            else "Safe mode: clear"
        ),
        (
            f"Confidence: {confidence.get('decision', 'PASS')} | "
            f"Avg: {_safe_float(confidence.get('avg_confidence')):.1f} | "
            f"Risk scale: {_format_scale(confidence_scaling.get('effective_risk_scale', 1.0))}"
        ),
    ]
    shadow_note = summary.get("shadow_note")
    if shadow_note:
        lines.append(f"Shadow note: {shadow_note}")
    no_trade_reasons = summary.get("no_trade_reasons", [])
    if isinstance(no_trade_reasons, list) and no_trade_reasons:
        lines.append(f"No-trade reasons: {', '.join(str(item) for item in no_trade_reasons[:3])}")
    lines.append("")
    _add_histogram_lines(lines, "Top selected reasons", selected_reasons)
    _add_histogram_lines(lines, "Top skipped reasons", skipped_reasons)
    lines.extend(["", DISCLAIMER])
    return _truncate_message("\n".join(lines))


def format_monthly_report_message(report: MonthlyReport) -> str:
    content = report.content_json if isinstance(report.content_json, dict) else {}
    summary = content.get("summary", {}) if isinstance(content, dict) else {}
    explainability = content.get("explainability", {}) if isinstance(content, dict) else {}
    confidence_scaling = (
        content.get("confidence_risk_scaling", {}) if isinstance(content, dict) else {}
    )
    selected_reasons = (
        explainability.get("selected_reason_histogram", {})
        if isinstance(explainability, dict)
        else {}
    )
    skipped_reasons = (
        explainability.get("skipped_reason_histogram", {})
        if isinstance(explainability, dict)
        else {}
    )

    best_day = summary.get("best_day", {}) if isinstance(summary.get("best_day"), dict) else {}
    worst_day = summary.get("worst_day", {}) if isinstance(summary.get("worst_day"), dict) else {}
    lines = [
        "Atlas Monthly Report",
        f"Month: {report.month}",
        f"Bundle: {report.bundle_id or '-'} | Policy: {report.policy_id or '-'}",
        "",
        (
            f"Runs: {_safe_int(summary.get('runs'))} | "
            f"Trading days: {_safe_int(summary.get('trading_days'))}"
        ),
        (
            f"Entries: {_safe_int(summary.get('entries'))} | "
            f"Exits: {_safe_int(summary.get('exits'))}"
        ),
        f"Net PnL: {_format_money(summary.get('net_pnl'))} | Costs: {_format_money(summary.get('costs'))}",
        f"Max drawdown: {_format_pct(summary.get('max_drawdown'))}",
        (
            f"Best day: {best_day.get('date', '-')} "
            f"({_format_money(best_day.get('net_pnl'))})"
        ),
        (
            f"Worst day: {worst_day.get('date', '-')} "
            f"({_format_money(worst_day.get('net_pnl'))})"
        ),
        (
            f"Confidence scale avg/min/max: "
            f"{_format_scale(confidence_scaling.get('avg_scale', 1.0))} / "
            f"{_format_scale(confidence_scaling.get('min_scale', 1.0))} / "
            f"{_format_scale(confidence_scaling.get('max_scale', 1.0))}"
        ),
        "",
    ]
    _add_histogram_lines(lines, "Top selected reasons", selected_reasons)
    _add_histogram_lines(lines, "Top skipped reasons", skipped_reasons)
    lines.extend(["", DISCLAIMER])
    return _truncate_message("\n".join(lines))


def send_daily_report_to_telegram(settings: Settings, report: DailyReport) -> dict[str, Any]:
    return send_telegram_message(settings, format_daily_report_message(report))


def send_monthly_report_to_telegram(settings: Settings, report: MonthlyReport) -> dict[str, Any]:
    return send_telegram_message(settings, format_monthly_report_message(report))


def maybe_send_daily_report_to_telegram(
    settings: Settings,
    report: DailyReport,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not (force or settings.telegram_send_reports):
        return {"status": "SKIPPED", "reason": "auto_send_disabled"}
    return try_send_telegram_message(settings, format_daily_report_message(report))


def maybe_send_monthly_report_to_telegram(
    settings: Settings,
    report: MonthlyReport,
    *,
    force: bool = False,
) -> dict[str, Any]:
    if not (force or settings.telegram_send_reports):
        return {"status": "SKIPPED", "reason": "auto_send_disabled"}
    return try_send_telegram_message(settings, format_monthly_report_message(report))
