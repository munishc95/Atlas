from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import PaperOrder, PaperPosition, PaperState, Symbol
from app.db.session import engine
from app.main import app


DATA_FILE = Path("data/sample/NIFTY500_1d.csv")


def _client_inline_jobs() -> TestClient:
    # Force local inline mode so tests can run without Redis/RQ process.
    import os

    os.environ["ATLAS_JOBS_INLINE"] = "true"
    get_settings.cache_clear()
    return TestClient(app)


def _wait_job(client: TestClient, job_id: str, timeout: float = 20.0) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        payload = client.get(f"/api/jobs/{job_id}").json()["data"]
        if payload["status"] in {"SUCCEEDED", "FAILED", "DONE"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} timed out")


def _import_sample(client: TestClient) -> None:
    with DATA_FILE.open("rb") as fh:
        resp = client.post(
            "/api/data/import",
            data={"symbol": "NIFTY500", "timeframe": "1d", "provider": "csv"},
            files={"file": (DATA_FILE.name, fh, "text/csv")},
        )
    assert resp.status_code == 200
    job_id = resp.json()["data"]["job_id"]
    job = _wait_job(client, job_id)
    assert job["status"] == "SUCCEEDED"


def test_backtest_job_lifecycle_inline() -> None:
    with _client_inline_jobs() as client:
        _import_sample(client)

        run = client.post(
            "/api/backtests/run",
            json={
                "symbol": "NIFTY500",
                "timeframe": "1d",
                "strategy_template": "trend_breakout",
                "params": {"trend_period": 200, "breakout_lookback": 20},
                "config": {"atr_stop_mult": 2.0, "atr_trail_mult": 2.0},
            },
        )
        assert run.status_code == 200

        job_id = run.json()["data"]["job_id"]
        job = _wait_job(client, job_id)
        assert job["status"] == "SUCCEEDED"
        assert "backtest_id" in (job["result_json"] or {})


def test_sse_progress_stream_smoke() -> None:
    with _client_inline_jobs() as client:
        _import_sample(client)

        run = client.post(
            "/api/backtests/run",
            json={
                "symbol": "NIFTY500",
                "timeframe": "1d",
                "strategy_template": "trend_breakout",
            },
        )
        job_id = run.json()["data"]["job_id"]

        seen_progress = False
        with client.stream("GET", f"/api/jobs/{job_id}/stream") as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith("event: progress"):
                    seen_progress = True
                    break

        assert seen_progress


def test_walkforward_uses_optuna_config() -> None:
    with _client_inline_jobs() as client:
        _import_sample(client)

        run = client.post(
            "/api/walkforward/run",
            json={
                "symbol": "NIFTY500",
                "timeframe": "1d",
                "strategy_template": "trend_breakout",
                "config": {
                    "trials": 5,
                    "sampler": "random",
                    "pruner": "none",
                    "seed": 42,
                    "timeout_seconds": 30,
                },
            },
        )
        assert run.status_code == 200

        job_id = run.json()["data"]["job_id"]
        job = _wait_job(client, job_id, timeout=60.0)
        assert job["status"] == "SUCCEEDED"

        result = job["result_json"] or {}
        run_id = result["run_id"]
        detail = client.get(f"/api/walkforward/{run_id}").json()["data"]
        summary = detail["summary_json"]

        assert summary["optimization"]["trials"] == 5
        assert summary["optimization"]["sampler"] == "random"
        assert len(summary["folds"]) >= 1


def test_idempotency_key_returns_same_job_id() -> None:
    with _client_inline_jobs() as client:
        _import_sample(client)

        payload = {
            "symbol": "NIFTY500",
            "timeframe": "1d",
            "strategy_template": "trend_breakout",
            "params": {"trend_period": 200, "breakout_lookback": 20},
        }
        headers = {"Idempotency-Key": "bt-idem-001"}

        first = client.post("/api/backtests/run", json=payload, headers=headers)
        second = client.post("/api/backtests/run", json=payload, headers=headers)

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["data"]["job_id"] == second.json()["data"]["job_id"]
        assert second.json()["data"].get("deduplicated") is True


def test_paper_diversification_limits_sector_concentration() -> None:
    with _client_inline_jobs() as client:
        reset = client.put(
            "/api/settings", json={"paper_mode": "strategy", "active_policy_id": None}
        )
        assert reset.status_code == 200

        with Session(engine) as session:
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
                session.add(state)
            for symbol in ("BANKA", "BANKB", "BANKC"):
                if session.exec(select(Symbol).where(Symbol.symbol == symbol)).first() is None:
                    session.add(Symbol(symbol=symbol, name=symbol, sector="FINANCIALS"))
            session.commit()

        run = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "signals": [
                    {
                        "symbol": "BANKA",
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 100,
                        "stop_distance": 10,
                        "signal_strength": 0.9,
                    },
                    {
                        "symbol": "BANKB",
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 100,
                        "stop_distance": 10,
                        "signal_strength": 0.8,
                    },
                    {
                        "symbol": "BANKC",
                        "side": "BUY",
                        "template": "trend_breakout",
                        "price": 100,
                        "stop_distance": 10,
                        "signal_strength": 0.7,
                    },
                ],
                "mark_prices": {},
            },
        )
        assert run.status_code == 200
        job_id = run.json()["data"]["job_id"]
        job = _wait_job(client, job_id)
        assert job["status"] == "SUCCEEDED"

        result = job["result_json"] or {}
        positions = result.get("positions", [])
        skipped = result.get("skipped_signals", [])
        assert len([p for p in positions if p["symbol"] in {"BANKA", "BANKB", "BANKC"}]) <= 2
        assert any(item.get("reason") == "sector_concentration" for item in skipped)
