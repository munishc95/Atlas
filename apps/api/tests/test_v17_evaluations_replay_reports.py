from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import get_settings
from app.db.models import PaperRun, PaperState, Policy
from app.db.session import engine
from app.main import app
from app.services.data_store import DataStore
from app.services.evaluations import execute_policy_evaluation
from app.services.replay import execute_replay_run


def _client_inline_jobs() -> TestClient:
    os.environ["ATLAS_JOBS_INLINE"] = "true"
    get_settings.cache_clear()
    return TestClient(app)


def _wait_job(client: TestClient, job_id: str, timeout: float = 120.0) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        payload = client.get(f"/api/jobs/{job_id}").json()["data"]
        if payload["status"] in {"SUCCEEDED", "FAILED", "DONE"}:
            return payload
        time.sleep(0.2)
    raise AssertionError(f"job {job_id} timed out")


def _frame(rows: int = 300, start: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 2_500_000),
        }
    )


def _breakout_then_upside_frame(rows: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    close = np.full(rows, 100.0)
    high = np.full(rows, 101.0)
    low = np.full(rows, 99.0)
    open_ = np.full(rows, 100.0)
    volume = np.full(rows, 2_500_000)

    decision_idx = 80
    fill_idx = decision_idx + 1
    open_[decision_idx] = 104.0
    high[decision_idx] = 116.0
    low[decision_idx] = 100.0
    close[decision_idx] = 115.0
    volume[decision_idx] = 5_000_000

    open_[fill_idx] = 116.0
    high[fill_idx] = 118.0
    low[fill_idx] = 115.0
    close[fill_idx] = 117.0
    for idx_pos in range(fill_idx + 1, rows):
        close[idx_pos] = 117.0 + (idx_pos - fill_idx) * 1.5
        open_[idx_pos] = close[idx_pos] - 0.4
        high[idx_pos] = close[idx_pos] + 1.0
        low[idx_pos] = close[idx_pos] - 1.0

    return pd.DataFrame(
        {
            "datetime": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _seed_bundle_and_policies() -> tuple[int, int, int]:
    settings = get_settings()
    store = DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    unique = uuid4().hex[:8]
    bundle_name = f"eval-bundle-{unique}"
    symbol = f"EVAL_{unique[:4].upper()}"
    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=_frame(),
            provider=f"provider-{unique}",
            bundle_name=bundle_name,
        )
        assert dataset.bundle_id is not None
        bundle_id = int(dataset.bundle_id)

        champion = Policy(
            name=f"Champion-{unique}",
            definition_json={
                "universe": {"bundle_id": bundle_id, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "regime_map": {
                    "TREND_UP": {
                        "strategy_key": "trend_breakout",
                        "params": {"trend_breakout": {"trend_period": 50, "breakout_lookback": 10}},
                        "risk_scale": 1.0,
                        "max_positions_scale": 1.0,
                    }
                },
            },
        )
        challenger = Policy(
            name=f"Challenger-{unique}",
            definition_json={
                "universe": {"bundle_id": bundle_id, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "regime_map": {
                    "TREND_UP": {
                        "strategy_key": "pullback_trend",
                        "params": {"pullback_trend": {"rsi_period": 4}},
                        "risk_scale": 1.0,
                        "max_positions_scale": 1.0,
                    }
                },
            },
        )
        session.add(champion)
        session.add(challenger)
        session.commit()
        session.refresh(champion)
        session.refresh(challenger)
        assert champion.id is not None
        assert challenger.id is not None
        return bundle_id, int(champion.id), int(challenger.id)


def _seed_signal_audit_policy() -> tuple[int, int]:
    settings = get_settings()
    store = DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    unique = uuid4().hex[:8]
    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=f"AUDIT_{unique[:4].upper()}",
            timeframe="1d",
            frame=_breakout_then_upside_frame(),
            provider=f"audit-provider-{unique}",
            bundle_name=f"audit-bundle-{unique}",
        )
        assert dataset.bundle_id is not None
        policy = Policy(
            name=f"AuditPolicy-{unique}",
            definition_json={
                "universe": {
                    "bundle_id": int(dataset.bundle_id),
                    "symbol_scope": "all",
                    "max_symbols_scan": 10,
                },
                "timeframes": ["1d"],
                "regime_map": {
                    "TREND_UP": {
                        "strategy_key": "trend_breakout",
                        "params": {
                            "trend_breakout": {
                                "trend_period": 10,
                                "breakout_lookback": 15,
                                "atr_stop_mult": 2.0,
                            }
                        },
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
        return int(dataset.bundle_id), int(policy.id)


def test_shadow_evaluation_does_not_mutate_live_paper_state() -> None:
    bundle_id, champion_id, challenger_id = _seed_bundle_and_policies()
    with _client_inline_jobs() as client:
        with Session(engine) as session:
            state = session.get(PaperState, 1)
            assert state is not None
            state.settings_json = {**(state.settings_json or {}), "paper_mode": "strategy", "active_policy_id": None}
            session.add(state)
            session.commit()
            before_state = {
                "equity": state.equity,
                "cash": state.cash,
                "active_policy_id": (state.settings_json or {}).get("active_policy_id"),
                "paper_mode": (state.settings_json or {}).get("paper_mode"),
            }
            before_positions = len(client.get("/api/paper/positions").json()["data"])
            before_orders = len(client.get("/api/paper/orders").json()["data"])

        response = client.post(
            "/api/evaluations/run",
            json={
                "bundle_id": bundle_id,
                "champion_policy_id": champion_id,
                "challenger_policy_ids": [challenger_id],
                "window_days": 20,
                "seed": 7,
            },
        )
        assert response.status_code == 200
        job_id = response.json()["data"]["job_id"]
        job = _wait_job(client, job_id)
        assert job["status"] == "SUCCEEDED"

        after_positions = len(client.get("/api/paper/positions").json()["data"])
        after_orders = len(client.get("/api/paper/orders").json()["data"])
        assert before_positions == after_positions
        assert before_orders == after_orders
        with Session(engine) as session:
            state = session.get(PaperState, 1)
            assert state is not None
            after_state = {
                "equity": state.equity,
                "cash": state.cash,
                "active_policy_id": (state.settings_json or {}).get("active_policy_id"),
                "paper_mode": (state.settings_json or {}).get("paper_mode"),
            }
        assert before_state == after_state


def test_evaluation_recommendation_is_deterministic_for_same_seed() -> None:
    bundle_id, champion_id, challenger_id = _seed_bundle_and_policies()
    settings = get_settings()
    store = DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    with Session(engine) as session:
        first = execute_policy_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={
                "bundle_id": bundle_id,
                "champion_policy_id": champion_id,
                "challenger_policy_ids": [challenger_id],
                "window_days": 30,
                "seed": 13,
            },
        )
        second = execute_policy_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={
                "bundle_id": bundle_id,
                "champion_policy_id": champion_id,
                "challenger_policy_ids": [challenger_id],
                "window_days": 30,
                "seed": 13,
            },
        )
    first_decision = (first.get("summary", {}) or {}).get("decision", {})
    second_decision = (second.get("summary", {}) or {}).get("decision", {})
    assert first_decision == second_decision


def test_replay_is_deterministic_for_same_seed() -> None:
    bundle_id, champion_id, _ = _seed_bundle_and_policies()
    settings = get_settings()
    store = DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    payload = {
        "bundle_id": bundle_id,
        "policy_id": champion_id,
        "start_date": "2023-10-01",
        "end_date": "2023-12-31",
        "seed": 19,
    }
    with Session(engine) as session:
        first = execute_replay_run(
            session=session,
            store=store,
            settings=settings,
            payload=payload,
        )
        second = execute_replay_run(
            session=session,
            store=store,
            settings=settings,
            payload=payload,
        )
    first_summary = first.get("summary", {}) or {}
    second_summary = second.get("summary", {}) or {}
    assert first_summary.get("digest") == second_summary.get("digest")
    assert first_summary.get("daily") == second_summary.get("daily")
    assert first_summary.get("signal_audit") == second_summary.get("signal_audit")


def test_replay_signal_audit_measures_point_in_time_upside() -> None:
    bundle_id, policy_id = _seed_signal_audit_policy()
    settings = get_settings()
    store = DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )
    with Session(engine) as session:
        result = execute_replay_run(
            session=session,
            store=store,
            settings=settings,
            payload={
                "bundle_id": bundle_id,
                "policy_id": policy_id,
                "start_date": "2024-02-01",
                "end_date": "2024-03-31",
                "seed": 7,
                "audit_offsets_days": [9],
                "audit_min_upside_pct": 5.0,
                "audit_max_signals_per_checkpoint": 5,
            },
        )

    audit = ((result.get("summary", {}) or {}).get("signal_audit", {}) or {})
    assert audit.get("enabled") is True
    assert audit.get("summary", {}).get("eligible_count", 0) >= 1
    assert audit.get("summary", {}).get("worked_count", 0) >= 1
    checkpoint = audit.get("checkpoints", [])[0]
    assert checkpoint["asof_date"] == "2024-03-22"
    first_signal = checkpoint["signals"][0]
    assert first_signal["worked"] is True
    assert first_signal["max_favorable_pct"] >= 5.0


def test_daily_report_pdf_endpoint_returns_pdf_bytes() -> None:
    bundle_id, policy_id, _ = _seed_bundle_and_policies()
    with Session(engine) as session:
        session.add(
            PaperRun(
                bundle_id=bundle_id,
                policy_id=policy_id,
                asof_ts=datetime.now(timezone.utc),
                regime="TREND_UP",
                signals_source="generated",
                generated_signals_count=4,
                selected_signals_count=1,
                skipped_signals_count=3,
                scanned_symbols=20,
                evaluated_candidates=8,
                scan_truncated=False,
                summary_json={
                    "equity_before": 1_000_000.0,
                    "equity_after": 1_001_250.0,
                    "net_pnl": 1_250.0,
                    "gross_pnl": 1_320.0,
                    "total_cost": 70.0,
                    "positions_before": 0,
                    "positions_after": 1,
                    "positions_opened": 1,
                    "positions_closed": 0,
                    "drawdown": -0.01,
                    "selected_reason_histogram": {"policy_selected": 1},
                    "skipped_reason_histogram": {"max_positions_reached": 2},
                    "selected_signals": [
                        {"symbol": "NIFTY500", "side": "BUY", "instrument_kind": "EQUITY_CASH"}
                    ],
                },
                cost_summary_json={"total_cost": 70.0},
            )
        )
        session.commit()

    with _client_inline_jobs() as client:
        generate = client.post(
            "/api/reports/daily/generate",
            json={"bundle_id": bundle_id, "policy_id": policy_id},
        )
        assert generate.status_code == 200
        job = _wait_job(client, generate.json()["data"]["job_id"])
        assert job["status"] == "SUCCEEDED"
        report_id = int((job["result_json"] or {})["id"])
        response = client.get(f"/api/reports/daily/{report_id}/export.pdf")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/pdf")
        assert len(response.content) > 1000
