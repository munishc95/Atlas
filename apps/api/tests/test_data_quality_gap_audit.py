from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from app.db.models import DataQualityReport
from app.services.data_quality_gap_audit import (
    EXCHANGE_NO_ROW_MEMBERSHIP_UNKNOWN,
    LOCAL_MISSING_EXCHANGE_PRESENT,
    OUTSIDE_HISTORICAL_MEMBERSHIP,
    PROVIDER_DAY_UNAVAILABLE,
    build_data_quality_gap_audit_payload,
    gap_audit_markdown,
)


def _local_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-01", "2026-01-06"], utc=True)
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": [100.0, 106.0],
            "high": [101.0, 107.0],
            "low": [99.0, 105.0],
            "close": [100.5, 106.5],
            "volume": [1_000_000, 1_100_000],
        }
    )


def test_gap_audit_classifies_missing_dates_against_provider_and_membership() -> None:
    report = DataQualityReport(
        id=77,
        bundle_id=11,
        timeframe="1d",
        status="WARN",
        created_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
        coverage_pct=99.2,
        total_symbols=1,
        issues_json=[
            {
                "severity": "WARN",
                "code": "gap_exceeds_threshold",
                "symbol": "GAP_A",
                "message": "Detected 4 missing trading bars between 2026-01-01 and 2026-01-06.",
            }
        ],
    )
    trading_days = [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
        date(2026, 1, 4),
        date(2026, 1, 5),
        date(2026, 1, 6),
    ]

    def day_frame_lookup(day: date) -> pd.DataFrame:
        if day == date(2026, 1, 2):
            return pd.DataFrame({"symbol": ["GAP_A"], "close": [101.0]})
        if day == date(2026, 1, 5):
            return pd.DataFrame()
        return pd.DataFrame({"symbol": ["OTHER"], "close": [101.0]})

    def membership_lookup(symbol: str, day: date) -> bool | None:
        if symbol == "GAP_A" and day == date(2026, 1, 4):
            return False
        if symbol == "GAP_A" and day == date(2026, 1, 5):
            return True
        return None

    payload = build_data_quality_gap_audit_payload(
        report,
        bundle_name="test-bundle",
        max_gap_bars=0,
        frame_lookup=lambda symbol: _local_frame() if symbol == "GAP_A" else pd.DataFrame(),
        day_frame_lookup=day_frame_lookup,
        trading_days_lookup=lambda start, end: [
            day for day in trading_days if start <= day <= end
        ],
        membership_lookup=membership_lookup,
        membership_history_available=True,
        generated_at=datetime(2026, 5, 12, tzinfo=timezone.utc),
    )

    summary = payload["summary"]
    assert summary["symbols_audited"] == 1
    assert summary["gap_events_total"] == 1
    assert summary["missing_dates_total"] == 4
    assert summary["classification_counts"] == {
        EXCHANGE_NO_ROW_MEMBERSHIP_UNKNOWN: 1,
        LOCAL_MISSING_EXCHANGE_PRESENT: 1,
        OUTSIDE_HISTORICAL_MEMBERSHIP: 1,
        PROVIDER_DAY_UNAVAILABLE: 1,
    }
    assert summary["backfillable_missing_dates"] == 1
    assert summary["backfillable_symbols"] == 1

    row = payload["symbols"][0]
    assert row["symbol"] == "GAP_A"
    assert row["primary_classification"] == LOCAL_MISSING_EXCHANGE_PRESENT
    assert row["recommended_action"] == "backfill_local_rows"
    assert row["first_missing_date"] == "2026-01-02"
    assert row["last_missing_date"] == "2026-01-05"
    assert row["gaps"][0]["missing_dates_count"] == 4

    markdown = gap_audit_markdown(payload)
    assert "Data Quality Gap Audit" in markdown
    assert "GAP_A" in markdown
