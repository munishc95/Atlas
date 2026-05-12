from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import get_settings
from app.db.models import DataQualityException
from app.db.session import engine, init_db
from app.services.data_quality import run_data_quality_report
from app.services.data_quality_exceptions import (
    EXCEPTION_NO_TRADE_OR_SUSPENSION,
    active_no_trade_exception_dates,
)
from app.services.data_store import DataStore


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def test_no_trade_exceptions_suppress_daily_gap_issue() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"DQEX_{uuid4().hex[:6].upper()}"
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-02-02T10:00:00Z", "2026-02-06T10:00:00Z"],
                utc=True,
            ),
            "open": np.array([100.0, 104.0]),
            "high": np.array([101.0, 105.0]),
            "low": np.array([99.0, 103.0]),
            "close": np.array([100.5, 104.5]),
            "volume": np.array([2_000_000, 2_100_000]),
        }
    )

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame,
            provider=f"dqex-{uuid4().hex[:8]}",
            bundle_name=f"bundle-dqex-{uuid4().hex[:8]}",
        )
        assert dataset.bundle_id is not None
        for day in (date(2026, 2, 3), date(2026, 2, 4), date(2026, 2, 5)):
            session.add(
                DataQualityException(
                    bundle_id=int(dataset.bundle_id),
                    timeframe="1d",
                    symbol=symbol,
                    trading_date=day,
                    kind=EXCEPTION_NO_TRADE_OR_SUSPENSION,
                    status="ACTIVE",
                    source="test",
                    reason="Exchange source had no row in audit.",
                )
            )
        session.commit()

        exception_dates = active_no_trade_exception_dates(
            session,
            bundle_id=int(dataset.bundle_id),
            timeframe="1d",
            symbols=[symbol],
        )
        assert exception_dates[symbol] == {
            date(2026, 2, 3),
            date(2026, 2, 4),
            date(2026, 2, 5),
        }

        report = run_data_quality_report(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(dataset.bundle_id),
            timeframe="1d",
            overrides={
                "trading_calendar_segment": "EQUITIES",
                "operate_max_gap_bars": 0,
                "data_quality_gap_fail_lookback_days": 45,
                "data_quality_max_stale_minutes_1d": 10_000_000,
                "operate_mode": "offline",
            },
            reference_ts=datetime(2026, 2, 6, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        )

    gap_issues = [
        item for item in (report.issues_json or []) if item.get("code") == "gap_exceeds_threshold"
    ]
    assert gap_issues == []
