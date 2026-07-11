from __future__ import annotations

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.db.models import Strategy, WalkForwardRun
from app.db.session import engine, init_db
from app.main import app
from app.services.research import _best_params
from app.services.walkforward import _promotion_consensus


def test_walkforward_rejects_zero_step_before_enqueue() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/walkforward/run",
            json={
                "symbol": "NIFTY500",
                "strategy_template": "trend_breakout",
                "config": {"step_months": 0},
            },
        )

    assert response.status_code == 422


def test_promotion_consensus_is_order_invariant_and_preserves_default_types() -> None:
    folds = [
        {"params": {"lookback": 10, "atr_stop_mult": 1.5, "direction": "long"}},
        {"params": {"lookback": 30, "atr_stop_mult": 2.5, "direction": "short"}},
        {"params": {"lookback": 20, "atr_stop_mult": 2.0, "direction": "long"}},
    ]
    defaults = {"lookback": 20, "atr_stop_mult": 2.0, "direction": "long"}

    forward = _promotion_consensus(folds, default_params=defaults)
    reverse = _promotion_consensus(list(reversed(folds)), default_params=defaults)

    assert forward == reverse
    assert forward == {"atr_stop_mult": 2.0, "direction": "long", "lookback": 20}
    assert isinstance(forward["lookback"], int)


def test_research_params_ignore_oos_fold_scores() -> None:
    promotion = {
        "method": "train_fold_consensus_median_mode_v1",
        "params": {"lookback": 20, "atr_stop_mult": 2.0},
    }
    first = {
        "promotion": promotion,
        "folds": [
            {"params": {"lookback": 5}, "oos_score": 100.0},
            {"params": {"lookback": 99}, "oos_score": -100.0},
        ],
    }
    second = {
        "promotion": promotion,
        "folds": [
            {"params": {"lookback": 5}, "oos_score": -100.0},
            {"params": {"lookback": 99}, "oos_score": 100.0},
        ],
    }

    assert _best_params(first) == _best_params(second) == promotion["params"]


def test_strategy_promotion_requires_locked_consensus() -> None:
    init_db()
    digest = "a" * 64
    with Session(engine) as session:
        run = WalkForwardRun(
            config_json={"strategy_template": "trend_breakout"},
            summary_json={
                "eligible_for_promotion": True,
                "engine_version": "test-engine",
                "data_digest": "data-digest",
                "promotion": {
                    "method": "train_fold_consensus_median_mode_v1",
                    "params": {"breakout_lookback": 20},
                    "params_digest": digest,
                },
            },
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        run_id = int(run.id or 0)

    payload = {
        "strategy_name": "locked-consensus",
        "template": "trend_breakout",
        "params_json": {"breakout_lookback": 20},
        "walkforward_run_id": run_id,
        "promotion_method": "train_fold_consensus_median_mode_v1",
        "promotion_params_digest": digest,
    }
    with TestClient(app) as client:
        tampered = client.post(
            "/api/strategies/promote",
            json={**payload, "params_json": {"breakout_lookback": 99}},
        )
        accepted = client.post("/api/strategies/promote", json=payload)

    assert tampered.status_code == 409
    assert accepted.status_code == 200
    with Session(engine) as session:
        strategy = session.get(Strategy, int(accepted.json()["data"]["strategy_id"]))
        assert strategy is not None
        assert strategy.params_json["breakout_lookback"] == 20
        assert strategy.params_json["_atlas_promotion"]["walkforward_run_id"] == run_id
