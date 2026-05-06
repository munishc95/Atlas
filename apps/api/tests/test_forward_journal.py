from __future__ import annotations

import os
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import ForwardSignalJournal, PaperOrder, PaperPosition, PaperState
from app.db.session import engine
from app.jobs.tasks import _operate_run_result
from app.main import app
from app.services.data_store import DataStore
from app.services.jobs import create_job


def _client_inline_jobs() -> TestClient:
    os.environ["ATLAS_JOBS_INLINE"] = "true"
    get_settings.cache_clear()
    return TestClient(app)


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(rows: int = 280, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 0.5,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 3_500_000),
        }
    )


def _reset_paper_state(session: Session) -> None:
    for row in session.exec(select(PaperPosition)).all():
        session.delete(row)
    for row in session.exec(select(PaperOrder)).all():
        session.delete(row)
    state = session.get(PaperState, 1)
    if state is not None:
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
            "active_ensemble_id": None,
            "allowed_sides": ["BUY"],
            "operate_mode": "offline",
            "data_quality_stale_severity": "WARN",
            "data_quality_stale_severity_override": True,
            "no_trade_enabled": False,
        }
        session.add(state)
    session.commit()


def test_forward_journal_captures_and_evaluates_signal_outcome() -> None:
    provider = f"fj-{uuid4().hex[:8]}"
    symbol = f"FJ_{uuid4().hex[:6].upper()}"
    bundle_name = f"forward-journal-{provider}"
    store = _store()
    frame = _frame()
    asof_idx = 240

    with _client_inline_jobs() as client:
        with Session(engine) as session:
            _reset_paper_state(session)
            dataset = store.save_ohlcv(
                session=session,
                symbol=symbol,
                timeframe="1d",
                frame=frame,
                provider=provider,
                bundle_name=bundle_name,
            )
            assert dataset.bundle_id is not None
            bundle_id = int(dataset.bundle_id)

        capture = client.post(
            "/api/paper/forward-journal/capture",
            json={
                "bundle_id": bundle_id,
                "timeframe": "1d",
                "symbol_scope": "all",
                "max_symbols_scan": 5,
                "asof": frame.iloc[asof_idx]["datetime"].isoformat(),
                "max_entry_extension_pct": 1000,
            },
        )
        assert capture.status_code == 200
        capture_payload = capture.json()["data"]
        assert capture_payload["captured_count"] >= 1
        row = capture_payload["journal"][0]
        assert row["symbol"] == symbol
        assert row["planned_qty"] > 0

        fill_ts = pd.Timestamp(row["fill_at"])
        if fill_ts.tzinfo is None:
            fill_ts = fill_ts.tz_localize("UTC")
        fill_idx = int(frame.index[frame["datetime"] == fill_ts][0])
        target_2 = float(row["target_2_price"])
        updated = frame.copy()
        updated.loc[fill_idx + 1, "high"] = target_2 + 2.0
        updated.loc[fill_idx + 1, "close"] = target_2 + 1.0
        updated.loc[fill_idx + 1, "low"] = min(
            float(updated.loc[fill_idx + 1, "low"]),
            float(row["entry_price"]),
        )
        with Session(engine) as session:
            store.save_ohlcv(
                session=session,
                symbol=symbol,
                timeframe="1d",
                frame=updated,
                provider=provider,
                bundle_id=bundle_id,
            )

        evaluation = client.post(
            "/api/paper/forward-journal/evaluate",
            json={"bundle_id": bundle_id, "timeframe": "1d", "horizon_bars": 5},
        )
        assert evaluation.status_code == 200
        assert evaluation.json()["data"]["evaluated_count"] >= 1

        listing = client.get(
            f"/api/paper/forward-journal?bundle_id={bundle_id}&timeframe=1d&page_size=10"
        )
        assert listing.status_code == 200
        rows = listing.json()["data"]
        assert rows
        assert rows[0]["status"] == "T2_HIT"
        assert rows[0]["max_favorable_pct"] > 0

        with Session(engine) as session:
            saved = session.exec(
                select(ForwardSignalJournal).where(ForwardSignalJournal.bundle_id == bundle_id)
            ).all()
            assert saved


def test_operate_run_captures_forward_journal_step() -> None:
    provider = f"fj-op-{uuid4().hex[:8]}"
    symbol = f"FJOP_{uuid4().hex[:6].upper()}"
    store = _store()
    frame = _frame()

    with _client_inline_jobs():
        with Session(engine) as session:
            _reset_paper_state(session)
            dataset = store.save_ohlcv(
                session=session,
                symbol=symbol,
                timeframe="1d",
                frame=frame,
                provider=provider,
                bundle_name=f"forward-journal-operate-{provider}",
            )
            assert dataset.bundle_id is not None
            bundle_id = int(dataset.bundle_id)
            job = create_job(session, "operate_run")

            result = _operate_run_result(
                session=session,
                settings=get_settings(),
                store=store,
                payload={
                    "bundle_id": bundle_id,
                    "timeframe": "1d",
                    "include_data_updates": False,
                    "asof": frame.iloc[-1]["datetime"].isoformat(),
                },
                job_id=str(job.id),
            )

            summary = result["summary"]
            journal = summary["forward_journal"]
            assert journal["status"] == "SUCCEEDED"
            assert journal["captured_count"] + journal["capture_updated_count"] >= 1
            assert "forward_journal" in [step["name"] for step in summary["steps"]]
            assert session.exec(
                select(ForwardSignalJournal).where(ForwardSignalJournal.bundle_id == bundle_id)
            ).first()
