from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import DatasetBundle, PaperState
from app.db.session import engine, init_db
from app.services.data_quality import run_data_quality_report
from app.services.data_store import DataStore
from app.services.operate_scheduler import compute_next_scheduled_run_ist, run_auto_operate_once
from app.services.paper import get_or_create_paper_state
from app.services.trading_calendar import (
    get_session,
    is_trading_day,
    next_trading_day,
    previous_trading_day,
)


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def enqueue(self, task_path: str, *args: object, **kwargs: object) -> object:
        self.calls.append((task_path, args, kwargs))
        return {"task_path": task_path}


def _seed_bundle(session: Session) -> int:
    bundle = DatasetBundle(
        name=f"v22-bundle-{uuid4().hex[:8]}",
        provider="test",
        symbols_json=["NIFTY500"],
        supported_timeframes_json=["1d"],
    )
    session.add(bundle)
    session.commit()
    session.refresh(bundle)
    assert bundle.id is not None
    return int(bundle.id)


def test_calendar_holidays_and_special_sessions() -> None:
    settings = get_settings()

    assert is_trading_day(date(2026, 1, 26), settings=settings) is False  # Holiday
    assert is_trading_day(date(2026, 2, 1), settings=settings) is True  # Sunday special
    assert is_trading_day(date(2026, 1, 31), settings=settings) is False  # Weekend

    session = get_session(date(2026, 2, 1), settings=settings)
    assert session["is_special"] is True
    assert session["is_trading_day"] is True
    assert str(session["open_time"]) == "09:15"
    assert str(session["close_time"]) == "15:30"

    assert next_trading_day(date(2026, 1, 25), settings=settings) == date(2026, 1, 27)
    assert previous_trading_day(date(2026, 1, 26), settings=settings) == date(2026, 1, 23)


def test_scheduler_runs_on_special_session_weekend() -> None:
    init_db()
    settings = get_settings()
    now_ist = datetime(2026, 2, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Sunday special

    with Session(engine) as session:
        state = get_or_create_paper_state(session, settings)
        bundle_id = _seed_bundle(session)
        state.settings_json = {
            **(state.settings_json or {}),
            "trading_calendar_segment": "EQUITIES",
            "operate_auto_run_enabled": True,
            "operate_auto_run_time_ist": "09:00",
            "operate_last_auto_run_date": None,
            "active_policy_id": None,
        }
        session.add(state)
        session.commit()

        queue = _FakeQueue()
        fired = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=now_ist,
        )
        assert fired is True
        assert len(queue.calls) == 4
        assert [call[0] for call in queue.calls] == [
            "app.jobs.tasks.run_data_updates_job",
            "app.jobs.tasks.run_data_quality_job",
            "app.jobs.tasks.run_paper_step_job",
            "app.jobs.tasks.run_daily_report_job",
        ]

        refreshed = session.get(PaperState, 1)
        assert refreshed is not None
        assert str((refreshed.settings_json or {}).get("operate_last_auto_run_date")) == "2026-02-01"

        # same trading day should dedupe
        fired_again = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=now_ist.replace(hour=11, minute=0),
        )
        assert fired_again is False
        assert len(queue.calls) == 4

        assert bundle_id > 0


def test_compute_next_schedule_skips_holiday_and_respects_calendar() -> None:
    next_run = compute_next_scheduled_run_ist(
        auto_run_enabled=True,
        auto_run_time_ist="15:35",
        last_run_date=None,
        segment="EQUITIES",
        now_ist=datetime(2026, 1, 26, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")),  # holiday
    )
    assert next_run is not None
    assert next_run.startswith("2026-01-27T15:35")


def test_data_quality_gap_check_respects_calendar_holidays() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    symbol = f"GAPV22_{uuid4().hex[:6].upper()}"
    frame = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-01-23T09:15:00Z", "2026-01-27T09:15:00Z"],
                utc=True,
            ),
            "open": np.array([100.0, 101.0]),
            "high": np.array([101.0, 102.0]),
            "low": np.array([99.0, 100.0]),
            "close": np.array([100.5, 101.5]),
            "volume": np.array([2_000_000, 2_100_000]),
        }
    )

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame,
            provider=f"v22-gap-{uuid4().hex[:8]}",
            bundle_name=f"bundle-v22-gap-{uuid4().hex[:8]}",
        )
        assert dataset.bundle_id is not None
        report = run_data_quality_report(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(dataset.bundle_id),
            timeframe="1d",
            overrides={
                "trading_calendar_segment": "EQUITIES",
                "operate_max_gap_bars": 0,
                "data_quality_max_stale_minutes_1d": 10_000_000,
                "operate_mode": "offline",
            },
            reference_ts=datetime(2026, 1, 27, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
        )

        codes = {str(item.get("code", "")) for item in (report.issues_json or [])}
        assert "gap_exceeds_threshold" not in codes
