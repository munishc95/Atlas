from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from app.db.models import DataQualityReport
from app.services.data_quality_remediation import (
    build_data_quality_remediation_payload,
    remediation_markdown,
)


def _jump_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=12, freq="D", tz="UTC")
    close = [100, 101, 102, 103, 104, 180, 181, 182, 183, 184, 185, 186]
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": [1_000_000] * len(close),
        }
    )


def test_remediation_groups_symbols_and_recommends_actions() -> None:
    report = DataQualityReport(
        id=42,
        bundle_id=7,
        timeframe="1d",
        status="WARN",
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        coverage_pct=99.2,
        total_symbols=500,
        issues_json=[
            {
                "severity": "WARN",
                "code": "gap_exceeds_threshold",
                "message": "Detected 6 missing trading bars between 2026-01-03 and 2026-01-12.",
                "symbol": "GAP_A",
                "details": {
                    "fail_cutoff_date": "2026-03-27",
                    "missing_bars": 6,
                    "missing_dates": ["2026-01-05", "2026-01-06"],
                },
            },
            {
                "severity": "WARN",
                "code": "corporate_action_anomaly",
                "message": "Detected split-like jump(s) in close returns.",
                "symbol": "JUMP_A",
                "details": {"jump_count": 1, "jump_threshold": 0.35},
            },
            {
                "severity": "WARN",
                "code": "inactive_symbols_detected",
                "message": "2 symbols marked inactive for selection due to stale data.",
                "details": {"inactive_symbols_sample": ["STALE_A", "STALE_B"]},
            },
        ],
    )

    payload = build_data_quality_remediation_payload(
        report,
        bundle_name="test-bundle",
        frame_lookup=lambda symbol: _jump_frame() if symbol == "JUMP_A" else pd.DataFrame(),
        open_position_symbols={"JUMP_A"},
        latest_selected_symbols={"GAP_A"},
        generated_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )

    rows = {row["symbol"]: row for row in payload["symbols"]}
    assert payload["summary"]["symbols_with_issues"] == 4
    assert payload["summary"]["blocked_by_symbol_gate_count"] == 2

    assert rows["GAP_A"]["recommended_action"] == "backfill_ohlcv_history"
    assert rows["GAP_A"]["selected_in_latest_run"] is True
    assert rows["GAP_A"]["priority"] == "P1"
    assert rows["GAP_A"]["latest_affected_range"]["end"] == "2026-01-12"
    assert rows["GAP_A"]["latest_affected_range"]["end"] != "2026-03-27"

    assert rows["JUMP_A"]["recommended_action"] == "verify_corporate_action_adjustment"
    assert rows["JUMP_A"]["currently_open_position"] is True
    assert rows["JUMP_A"]["latest_affected_range"]["source"] == "merged"

    assert rows["STALE_A"]["recommended_action"] == "refresh_or_replace_latest_source_data"
    assert rows["STALE_A"]["blocked_by_symbol_gate"] is False

    markdown = remediation_markdown(payload, max_symbols=4)
    assert "Data Quality Remediation Report" in markdown
    assert "GAP_A" in markdown
