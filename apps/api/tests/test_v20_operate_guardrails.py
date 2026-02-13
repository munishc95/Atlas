from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import OperateEvent, PaperOrder, PaperPosition, PaperState, Policy, PolicyHealthSnapshot
from app.db.session import engine, init_db
from app.services.data_quality import STATUS_FAIL, run_data_quality_report
from app.services.data_store import DataStore
from app.services.paper import get_or_create_paper_state, run_paper_step
from app.services.policy_health import WARNING, apply_policy_health_actions


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _valid_frame(rows: int = 120, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
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


def _reset_paper_state(*, settings, allow_sell: bool = True) -> None:
    with Session(engine) as session:
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
        state.settings_json = {
            **(state.settings_json or {}),
            "paper_mode": "strategy",
            "active_policy_id": None,
            "allowed_sides": ["BUY", "SELL"] if allow_sell else ["BUY"],
            "paper_use_simulator_engine": False,
            "operate_safe_mode_on_fail": True,
            "operate_safe_mode_action": "exits_only",
        }
        session.add(state)
        session.commit()


def test_data_quality_report_detects_bad_sample_fixture() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    sample_path = Path("data/sample/NIFTY500_BAD_1d.csv")
    assert sample_path.exists()

    frame = pd.read_csv(sample_path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    provider = f"bad-csv-{uuid4().hex[:8]}"

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol="BADCSV_A",
            timeframe="1d",
            frame=frame,
            provider=provider,
            bundle_name=f"bundle-{provider}",
        )
        assert dataset.bundle_id is not None
        report = run_data_quality_report(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(dataset.bundle_id),
            timeframe="1d",
            reference_ts=datetime.now(timezone.utc),
        )

    assert report.status == STATUS_FAIL
    issue_codes = {str(item.get("code", "")) for item in (report.issues_json or [])}
    assert "duplicate_timestamps" in issue_codes
    assert "invalid_ohlc_ranges" in issue_codes


def test_safe_mode_on_quality_fail_blocks_entries_but_allows_exits() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    _reset_paper_state(settings=settings)

    provider = f"safe-fail-{uuid4().hex[:8]}"
    bad_frame = _valid_frame(rows=20, start=120.0)
    # Force invalid OHLC row -> quality FAIL.
    bad_frame.loc[5, "high"] = bad_frame.loc[5, "low"] - 2.0

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol="SAFEFAIL_A",
            timeframe="1d",
            frame=bad_frame,
            provider=provider,
            bundle_name=f"bundle-{provider}",
        )
        assert dataset.bundle_id is not None

        state = session.get(PaperState, 1)
        assert state is not None
        state.settings_json = {
            **(state.settings_json or {}),
            "operate_safe_mode_on_fail": True,
            "operate_safe_mode_action": "exits_only",
        }
        session.add(state)

        # Existing open long should be eligible for exit updates.
        session.add(
            PaperPosition(
                symbol="SAFEFAIL_A",
                side="BUY",
                instrument_kind="EQUITY_CASH",
                lot_size=1,
                qty_lots=1,
                qty=10,
                avg_price=100.0,
                stop_price=95.0,
                target_price=None,
                opened_at=datetime.now(timezone.utc) - timedelta(days=2),
            )
        )
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
                        "symbol": "SAFEFAIL_A",
                        "side": "BUY",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.8,
                        "adv": 1_000_000_000,
                        "vol_scale": 0.01,
                    }
                ],
                "mark_prices": {"SAFEFAIL_A": 90.0},
                "seed": 17,
                "asof": "2024-03-20T10:00:00+00:00",
            },
            store=store,
        )

        assert bool(((result.get("safe_mode") or {}).get("active")))
        assert int(result.get("selected_signals_count", 0)) == 0
        assert any(
            str(item.get("reason", "")) == "data_quality_fail_safe_mode"
            for item in (result.get("skipped_signals") or [])
        )

        remaining_positions = session.exec(select(PaperPosition)).all()
        assert not remaining_positions


def test_digest_mismatch_creates_operate_event() -> None:
    init_db()
    settings = get_settings()
    store = _store()
    _reset_paper_state(settings=settings)

    provider = f"digest-{uuid4().hex[:8]}"
    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol="DIGEST_A",
            timeframe="1d",
            frame=_valid_frame(rows=80, start=75.0),
            provider=provider,
            bundle_name=f"bundle-{provider}",
        )
        assert dataset.bundle_id is not None

        state = session.get(PaperState, 1)
        assert state is not None
        state.settings_json = {
            **(state.settings_json or {}),
            "operate_safe_mode_on_fail": False,
            "paper_use_simulator_engine": False,
        }
        session.add(state)
        session.commit()

        asof = "2024-03-20T10:00:00+00:00"
        run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": int(dataset.bundle_id),
                "timeframes": ["1d"],
                "signals": [
                    {
                        "symbol": "DIGEST_A",
                        "side": "BUY",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 100.0,
                        "stop_distance": 4.0,
                        "signal_strength": 0.9,
                        "adv": 1_000_000_000,
                        "vol_scale": 0.01,
                    }
                ],
                "mark_prices": {},
                "seed": 11,
                "asof": asof,
            },
            store=store,
        )
        run_paper_step(
            session=session,
            settings=settings,
            payload={
                "regime": "TREND_UP",
                "bundle_id": int(dataset.bundle_id),
                "timeframes": ["1d"],
                "signals": [],
                "mark_prices": {"DIGEST_A": 80.0},
                "seed": 11,
                "asof": asof,
            },
            store=store,
        )

        mismatch_events = session.exec(
            select(OperateEvent)
            .where(OperateEvent.category == "SYSTEM")
            .where(OperateEvent.message == "Digest mismatch detected for same day/config/data digest.")
            .order_by(OperateEvent.ts.desc())
        ).all()
        assert mismatch_events


def test_policy_warning_action_emits_operate_event() -> None:
    init_db()
    settings = get_settings()
    unique = uuid4().hex[:8]

    with Session(engine) as session:
        policy = Policy(
            name=f"health-event-{unique}",
            definition_json={
                "status": "ACTIVE",
                "baseline": {"max_drawdown": 0.1, "win_rate": 0.55, "period_return": 0.2},
                "regime_map": {
                    "TREND_UP": {
                        "strategy_key": "trend_breakout",
                        "risk_scale": 1.0,
                        "max_positions_scale": 1.0,
                    }
                },
            },
        )
        session.add(policy)
        session.commit()
        session.refresh(policy)
        assert policy.id is not None

        snapshot = PolicyHealthSnapshot(
            policy_id=int(policy.id),
            asof_date=date.today(),
            window_days=20,
            metrics_json={"max_drawdown": -0.13},
            status=WARNING,
            reasons_json=["warning condition"],
        )
        session.add(snapshot)
        session.commit()
        session.refresh(snapshot)

        apply_policy_health_actions(
            session,
            settings=settings,
            policy=policy,
            snapshot=snapshot,
            overrides={"drift_warning_risk_scale": 0.75},
        )
        session.commit()

        events = session.exec(
            select(OperateEvent)
            .where(OperateEvent.category == "POLICY")
            .where(OperateEvent.message == "Policy health warning triggered risk scaling.")
            .order_by(OperateEvent.ts.desc())
        ).all()
        assert events
