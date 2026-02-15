from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import PaperOrder, PaperPosition, PaperState
from app.db.session import engine
from app.main import app
from app.services.research import evaluate_candidate_robustness


DATA_FILE = Path("data/sample/NIFTY500_1d.csv")


def _client_inline_jobs() -> TestClient:
    import os

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


def _import_sample(client: TestClient) -> None:
    with DATA_FILE.open("rb") as fh:
        response = client.post(
            "/api/data/import",
            data={"symbol": "NIFTY500", "timeframe": "1d", "provider": "csv"},
            files={"file": (DATA_FILE.name, fh, "text/csv")},
        )
    assert response.status_code == 200
    job_id = response.json()["data"]["job_id"]
    job = _wait_job(client, job_id)
    assert job["status"] == "SUCCEEDED"


def _run_small_research(client: TestClient, idempotency_key: str | None = None) -> dict:
    headers = {"Idempotency-Key": idempotency_key} if idempotency_key else {}
    response = client.post(
        "/api/research/run",
        json={
            "timeframes": ["1d"],
            "strategy_templates": ["trend_breakout", "pullback_trend"],
            "symbol_scope": "liquid",
            "config": {
                "trials_per_strategy": 4,
                "max_symbols": 1,
                "max_evaluations": 2,
                "min_trades": 1,
                "stress_pass_rate_threshold": 0.0,
                "sampler": "random",
                "pruner": "none",
                "seed": 19,
            },
        },
        headers=headers,
    )
    assert response.status_code == 200
    return response.json()["data"]


def _reset_paper_runtime() -> None:
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


def test_research_score_is_deterministic() -> None:
    kwargs = {
        "oos_metrics": {
            "calmar": 1.3,
            "cvar_95": -0.018,
            "turnover": 2.0,
            "consistency": 0.65,
            "trade_count": 42,
            "max_drawdown": -0.12,
        },
        "stress_metrics": {"calmar": 0.9, "cvar_95": -0.024, "max_drawdown": -0.16},
        "param_dispersion": 0.22,
        "fold_variance": 0.11,
        "stress_pass_rate": 0.8,
        "constraints": {"max_drawdown": 0.2, "min_trades": 20, "stress_pass_rate_threshold": 0.6},
    }
    first = evaluate_candidate_robustness(**kwargs)
    second = evaluate_candidate_robustness(**kwargs)
    assert first.score == second.score
    assert first.accepted == second.accepted
    assert first.explanations == second.explanations


def test_research_score_gating_blocks_fragile_candidate() -> None:
    evaluation = evaluate_candidate_robustness(
        oos_metrics={
            "calmar": 0.2,
            "cvar_95": -0.045,
            "turnover": 4.0,
            "consistency": 0.2,
            "trade_count": 4,
            "max_drawdown": -0.35,
        },
        stress_metrics={"calmar": 0.01, "cvar_95": -0.06, "max_drawdown": -0.4},
        param_dispersion=0.8,
        fold_variance=0.5,
        stress_pass_rate=0.2,
        constraints={"max_drawdown": 0.2, "min_trades": 10, "stress_pass_rate_threshold": 0.6},
    )
    assert evaluation.accepted is False
    assert evaluation.score <= -0.5
    joined = " ".join(evaluation.explanations).lower()
    assert "rejected" in joined
    assert "drawdown" in joined
    assert "stress pass rate" in joined


def test_research_job_creates_run_and_candidates() -> None:
    with _client_inline_jobs() as client:
        _import_sample(client)

        queued = _run_small_research(client)
        job = _wait_job(client, queued["job_id"], timeout=240.0)
        assert job["status"] == "SUCCEEDED"
        run_id = int((job["result_json"] or {})["run_id"])

        run_detail = client.get(f"/api/research/runs/{run_id}")
        assert run_detail.status_code == 200
        assert run_detail.json()["data"]["status"] == "SUCCEEDED"

        candidates = client.get(f"/api/research/runs/{run_id}/candidates?page=1&page_size=20")
        assert candidates.status_code == 200
        assert len(candidates.json()["data"]) >= 1


def test_research_idempotency_key_returns_same_job() -> None:
    with _client_inline_jobs() as client:
        _import_sample(client)
        first = _run_small_research(client, idempotency_key="research-idem-001")
        second = _run_small_research(client, idempotency_key="research-idem-001")
        assert first["job_id"] == second["job_id"]
        assert second.get("deduplicated") is True


def test_policy_generation_and_paper_mode_reasons() -> None:
    with _client_inline_jobs() as client:
        _reset_paper_runtime()
        _import_sample(client)
        queued = _run_small_research(client)
        research_job = _wait_job(client, queued["job_id"], timeout=240.0)
        assert research_job["status"] == "SUCCEEDED"
        run_id = int((research_job["result_json"] or {})["run_id"])

        create_policy = client.post(
            "/api/policies",
            json={"research_run_id": run_id, "name": "Auto Research Test Policy"},
        )
        assert create_policy.status_code == 200
        policy = create_policy.json()["data"]
        regime_map = policy["definition_json"].get("regime_map", {})
        trend_cfg = regime_map.get("TREND_UP", {})
        selected_template = str(trend_cfg.get("strategy_key") or "trend_breakout")
        blocked_template = (
            "squeeze_breakout" if selected_template != "squeeze_breakout" else "pullback_trend"
        )

        promote = client.post(f"/api/policies/{policy['id']}/promote-to-paper")
        assert promote.status_code == 200
        assert promote.json()["data"]["paper_mode"] == "policy"

        paper = client.post(
            "/api/paper/run-step",
            json={
                "regime": "TREND_UP",
                "signals": [
                    {
                        "symbol": "AUTO1",
                        "side": "BUY",
                        "template": selected_template,
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.9,
                    },
                    {
                        "symbol": "AUTO2",
                        "side": "BUY",
                        "template": blocked_template,
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.8,
                    },
                ],
                "mark_prices": {},
            },
        )
        assert paper.status_code == 200
        paper_job = _wait_job(client, paper.json()["data"]["job_id"])
        assert paper_job["status"] == "SUCCEEDED"
        result = paper_job["result_json"] or {}
        assert result.get("policy_mode") == "policy"
        skipped = result.get("skipped_signals", [])
        assert any(item.get("reason") == "template_blocked_by_policy" for item in skipped)

        reset = client.put(
            "/api/settings", json={"paper_mode": "strategy", "active_policy_id": None}
        )
        assert reset.status_code == 200
