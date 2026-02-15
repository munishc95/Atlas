from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import (
    DatasetBundle,
    PaperOrder,
    PaperPosition,
    PaperRun,
    PaperState,
    Policy,
    ShadowPaperState,
)
from app.db.session import engine, init_db
from app.services.data_store import DataStore
from app.services.operate_scheduler import compute_next_scheduled_run_ist, run_auto_operate_once
from app.services.paper import get_or_create_paper_state, run_paper_step
from app.services.reports import generate_daily_report


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(rows: int = 90, start: float = 100.0, *, start_date: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start_date, periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 2_000_000),
        }
    )


def _reset_live_state(session: Session, settings) -> PaperState:
    for row in session.exec(select(PaperPosition)).all():
        session.delete(row)
    for row in session.exec(select(PaperOrder)).all():
        session.delete(row)
    state = get_or_create_paper_state(session, settings)
    state.equity = 1_000_000.0
    state.cash = 1_000_000.0
    state.peak_equity = 1_000_000.0
    state.drawdown = 0.0
    state.kill_switch_active = False
    state.cooldown_days_left = 0
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def enqueue(self, task_path: str, *args: object, **kwargs: object) -> object:
        self.calls.append((task_path, args, kwargs))
        return {"task_path": task_path}


def test_shadow_only_safe_mode_keeps_live_state_and_persists_shadow_state() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"SHDW_{unique}"

    frame = _frame(rows=80, start=95.0)
    frame.loc[8, "high"] = frame.loc[8, "low"] - 2.0  # force data quality FAIL

    with Session(engine) as session:
        for row in session.exec(select(ShadowPaperState)).all():
            session.delete(row)
        session.commit()

        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame,
            provider=f"shadow-{unique}",
            bundle_name=f"bundle-shadow-{unique}",
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)

        state = _reset_live_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "paper_mode": "strategy",
            "active_policy_id": None,
            "allowed_sides": ["BUY", "SELL"],
            "paper_use_simulator_engine": True,
            "operate_safe_mode_on_fail": True,
            "operate_safe_mode_action": "shadow_only",
            "operate_mode": "live",
            "data_quality_stale_severity": "WARN",
            "data_quality_stale_severity_override": False,
        }
        session.add(state)
        session.commit()
        session.refresh(state)

        cash_before = float(state.cash)
        equity_before = float(state.equity)

        first = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "signals": [
                    {
                        "symbol": symbol,
                        "side": "BUY",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.8,
                        "adv": 1_000_000_000.0,
                        "vol_scale": 0.01,
                    }
                ],
                "mark_prices": {},
                "asof": "2026-02-11T10:00:00+05:30",
                "seed": 17,
            },
            store=store,
        )
        assert str(first.get("execution_mode")) == "SHADOW"
        assert bool((first.get("safe_mode") or {}).get("active"))
        assert bool(first.get("live_state_mutated")) is False
        assert len(first.get("simulated_positions", [])) >= 1

        state_after_first = session.get(PaperState, 1)
        assert state_after_first is not None
        assert float(state_after_first.cash) == cash_before
        assert float(state_after_first.equity) == equity_before
        assert len(session.exec(select(PaperPosition)).all()) == 0
        assert len(session.exec(select(PaperOrder)).all()) == 0

        shadow = session.exec(
            select(ShadowPaperState)
            .where(ShadowPaperState.bundle_id == bundle_id)
            .where(ShadowPaperState.policy_id == 0)
        ).first()
        assert shadow is not None
        assert isinstance(shadow.state_json, dict)
        assert len(shadow.state_json.get("positions", [])) >= 1

        second = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "signals": [],
                "mark_prices": {symbol: 101.0},
                "asof": "2026-02-12T10:00:00+05:30",
                "seed": 17,
            },
            store=store,
        )
        assert str(second.get("execution_mode")) == "SHADOW"
        assert len(second.get("simulated_positions", [])) >= 1
        session.refresh(shadow)
        assert shadow.last_run_id is not None


def test_shadow_daily_report_includes_shadow_note() -> None:
    init_db()
    settings = get_settings()
    report_day = date(2030, 1, 15)

    with Session(engine) as session:
        session.add(
            PaperRun(
                bundle_id=None,
                policy_id=None,
                asof_ts=datetime.combine(report_day, datetime.min.time(), tzinfo=timezone.utc),
                mode="SHADOW",
                regime="TREND_UP",
                signals_source="generated",
                generated_signals_count=2,
                selected_signals_count=1,
                skipped_signals_count=1,
                scanned_symbols=12,
                evaluated_candidates=6,
                scan_truncated=False,
                summary_json={
                    "execution_mode": "SHADOW",
                    "shadow_note": "Shadow-only: no live state mutation; simulated trades shown for monitoring.",
                    "net_pnl": 100.0,
                    "realized_pnl": 100.0,
                    "unrealized_pnl": 0.0,
                    "total_cost": 10.0,
                    "positions_after": 1,
                    "positions_opened": 1,
                    "positions_closed": 0,
                    "safe_mode_active": True,
                    "selected_reason_histogram": {"provided": 1},
                    "skipped_reason_histogram": {"max_positions_reached": 1},
                },
                cost_summary_json={"total_cost": 10.0},
            )
        )
        session.commit()

        report = generate_daily_report(
            session=session,
            settings=settings,
            report_date=report_day,
            bundle_id=None,
            policy_id=None,
            overwrite=True,
        )
        summary = report.content_json.get("summary", {})
        assert str(summary.get("mode")) == "SHADOW"
        assert "Shadow-only" in str(summary.get("shadow_note", ""))


def test_live_mode_stale_data_fail_activates_safe_mode() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    unique = uuid4().hex[:8].upper()
    symbol = f"STALE_{unique}"

    old_frame = _frame(rows=40, start=70.0, start_date="2020-01-01")

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=old_frame,
            provider=f"stale-{unique}",
            bundle_name=f"bundle-stale-{unique}",
        )
        assert dataset.bundle_id is not None

        state = _reset_live_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "paper_mode": "strategy",
            "active_policy_id": None,
            "operate_safe_mode_on_fail": True,
            "operate_safe_mode_action": "exits_only",
            "operate_mode": "live",
            "data_quality_stale_severity": "WARN",
            "data_quality_stale_severity_override": False,
        }
        session.add(state)
        session.commit()

        result = run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": int(dataset.bundle_id),
                "timeframes": ["1d"],
                "signals": [
                    {
                        "symbol": symbol,
                        "side": "BUY",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.8,
                        "adv": 1_000_000_000.0,
                        "vol_scale": 0.01,
                    }
                ],
                "mark_prices": {},
                "asof": "2026-02-13T10:00:00+05:30",
            },
            store=store,
        )

        safe_mode = result.get("safe_mode", {})
        assert bool(safe_mode.get("active"))
        assert str(safe_mode.get("status")) == "FAIL"
        assert int(result.get("selected_signals_count", 0)) == 0
        assert any(
            str(item.get("reason")) == "data_quality_fail_safe_mode"
            for item in (result.get("skipped_signals") or [])
        )


def test_operate_scheduler_runs_once_per_trading_day_and_skips_duplicates() -> None:
    init_db()
    settings = get_settings()

    with Session(engine) as session:
        state = _reset_live_state(session, settings)
        bundle = DatasetBundle(
            name=f"bundle-scheduler-{uuid4().hex[:8]}",
            provider="test",
            symbols_json=["NIFTY500"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        assert bundle.id is not None

        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_run_enabled": True,
            "operate_auto_run_time_ist": "09:00",
            "operate_last_auto_run_date": None,
            "operate_auto_run_include_data_updates": True,
            "data_updates_provider_enabled": False,
            "active_policy_id": None,
        }
        session.add(state)
        session.commit()

        queue = _FakeQueue()
        now_ist = datetime(2026, 2, 13, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Friday
        triggered = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=now_ist,
        )
        assert triggered is True
        assert len(queue.calls) == 4
        assert [call[0] for call in queue.calls] == [
            "app.jobs.tasks.run_data_updates_job",
            "app.jobs.tasks.run_data_quality_job",
            "app.jobs.tasks.run_paper_step_job",
            "app.jobs.tasks.run_daily_report_job",
        ]
        assert {call[0] for call in queue.calls} == {
            "app.jobs.tasks.run_data_updates_job",
            "app.jobs.tasks.run_data_quality_job",
            "app.jobs.tasks.run_paper_step_job",
            "app.jobs.tasks.run_daily_report_job",
        }

        refreshed = session.get(PaperState, 1)
        assert refreshed is not None
        assert (
            str((refreshed.settings_json or {}).get("operate_last_auto_run_date")) == "2026-02-13"
        )

        second = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=now_ist + timedelta(minutes=10),
        )
        assert second is False
        assert len(queue.calls) == 4

    monday_next = compute_next_scheduled_run_ist(
        auto_run_enabled=True,
        auto_run_time_ist="15:35",
        last_run_date="2026-02-13",
        now_ist=datetime(2026, 2, 13, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    assert monday_next is not None
    assert monday_next.startswith("2026-02-16T15:35")


def test_operate_scheduler_auto_eval_weekly_queues_once() -> None:
    init_db()
    settings = get_settings()

    with Session(engine) as session:
        state = _reset_live_state(session, settings)
        bundle = DatasetBundle(
            name=f"bundle-auto-eval-{uuid4().hex[:8]}",
            provider="test",
            symbols_json=["NIFTY500"],
            supported_timeframes_json=["1d"],
        )
        policy = Policy(
            name=f"policy-auto-eval-{uuid4().hex[:6]}",
            definition_json={
                "universe": {"bundle_id": None, "symbol_scope": "all"},
                "timeframes": ["1d"],
                "regime_map": {"TREND_UP": {"strategy_key": "trend_breakout"}},
            },
        )
        session.add(bundle)
        session.add(policy)
        session.commit()
        session.refresh(bundle)
        session.refresh(policy)
        assert bundle.id is not None
        assert policy.id is not None
        policy.definition_json = {
            **(policy.definition_json or {}),
            "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all"},
        }
        session.add(policy)
        session.commit()

        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_run_enabled": False,
            "operate_auto_eval_enabled": True,
            "operate_auto_eval_frequency": "WEEKLY",
            "operate_auto_eval_day_of_week": 0,
            "operate_auto_eval_time_ist": "09:00",
            "operate_last_auto_eval_date": None,
            "active_policy_id": int(policy.id),
        }
        session.add(state)
        session.commit()

        queue = _FakeQueue()
        now_ist = datetime(2026, 2, 16, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Monday
        triggered = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=now_ist,
        )
        assert triggered is True
        assert [call[0] for call in queue.calls] == ["app.jobs.tasks.run_auto_eval_job"]

        refreshed = session.get(PaperState, 1)
        assert refreshed is not None
        assert (
            str((refreshed.settings_json or {}).get("operate_last_auto_eval_date")) == "2026-02-16"
        )

        second = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=now_ist + timedelta(minutes=15),
        )
        assert second is False
        assert len(queue.calls) == 1
