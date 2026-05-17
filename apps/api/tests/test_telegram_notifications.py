from __future__ import annotations

import json
from datetime import date

from app.core.config import Settings
from app.db.models import DailyReport, MonthlyReport
from app.services import telegram as telegram_service


def _settings(**overrides):
    return Settings(_env_file=None, **overrides)


def test_telegram_status_reports_missing_configuration() -> None:
    settings = _settings(
        telegram_enabled=False,
        telegram_bot_token=None,
        telegram_chat_id=None,
    )

    payload = telegram_service.telegram_status_payload(settings)

    assert payload["ready"] is False
    assert payload["configured"] is False
    assert "ATLAS_TELEGRAM_BOT_TOKEN" in payload["missing"]
    assert "ATLAS_TELEGRAM_CHAT_ID" in payload["missing"]


def test_daily_report_message_contains_guardrails_and_summary() -> None:
    report = DailyReport(
        date=date(2026, 5, 17),
        bundle_id=7,
        policy_id=11,
        content_json={
            "summary": {
                "runs": 2,
                "entries": 1,
                "exits": 0,
                "positions_open": 1,
                "net_pnl": 1250.25,
                "costs": 70.0,
                "drawdown": -0.0125,
                "mode": "SHADOW",
                "regime": "TREND_UP",
                "kill_switch_active": False,
                "safe_mode_active": True,
                "shadow_note": "Data quality forced simulated execution.",
            },
            "risk": {"positions_peak": 1, "avg_exposure": 0.18},
            "confidence_gate": {"decision": "PASS", "avg_confidence": 82.5},
            "confidence_risk_scaling": {"effective_risk_scale": 0.75},
            "explainability": {
                "selected_reason_histogram": {"policy_selected": 1},
                "skipped_reason_histogram": {"max_positions_reached": 2},
            },
        },
    )

    message = telegram_service.format_daily_report_message(report)

    assert "Atlas Daily Report" in message
    assert "Date: 2026-05-17" in message
    assert "Net PnL: 1,250.25" in message
    assert "Safe mode: ACTIVE" in message
    assert "Risk scale: 75.0%" in message
    assert telegram_service.DISCLAIMER in message


def test_monthly_report_message_contains_best_and_worst_days() -> None:
    report = MonthlyReport(
        month="2026-05",
        bundle_id=7,
        policy_id=11,
        content_json={
            "summary": {
                "runs": 8,
                "trading_days": 4,
                "entries": 3,
                "exits": 2,
                "net_pnl": 2500.0,
                "costs": 120.0,
                "max_drawdown": -0.02,
                "best_day": {"date": "2026-05-14", "net_pnl": 1400.0},
                "worst_day": {"date": "2026-05-15", "net_pnl": -300.0},
            },
            "confidence_risk_scaling": {"avg_scale": 0.9, "min_scale": 0.5, "max_scale": 1.0},
            "explainability": {
                "selected_reason_histogram": {"policy_selected": 3},
                "skipped_reason_histogram": {},
            },
        },
    )

    message = telegram_service.format_monthly_report_message(report)

    assert "Atlas Monthly Report" in message
    assert "Month: 2026-05" in message
    assert "Best day: 2026-05-14 (1,400.00)" in message
    assert "Worst day: 2026-05-15 (-300.00)" in message


def test_send_telegram_message_posts_json(monkeypatch) -> None:
    settings = _settings(
        telegram_enabled=True,
        telegram_bot_token="123456:ABC",
        telegram_chat_id="987654321",
    )
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"ok": true, "result": {"message_id": 42}}'

    def fake_urlopen(req, timeout):  # noqa: ANN001
        captured["url"] = req.full_url
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(telegram_service.request, "urlopen", fake_urlopen)

    result = telegram_service.send_telegram_message(settings, "Atlas test")

    assert result["status"] == "SENT"
    assert result["message_id"] == 42
    assert "123456:ABC" in captured["url"]
    assert captured["payload"]["chat_id"] == "987654321"
    assert captured["payload"]["text"] == "Atlas test"
