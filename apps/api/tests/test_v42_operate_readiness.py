from __future__ import annotations

import os
from datetime import date, datetime, timezone
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import (
    DailyConfidenceAggregate,
    DataQualityReport,
    DatasetBundle,
    OperateEvent,
    PaperRun,
    PaperState,
    Policy,
    ProviderUpdateRun,
)
from app.db.session import engine, init_db
from app.main import app
from app.services.paper import get_or_create_paper_state


@pytest.fixture(autouse=True)
def preserve_operator_state():
    init_db()
    with Session(engine) as session:
        original_state = session.get(PaperState, 1)
        original_state_dump = original_state.model_dump() if original_state is not None else None
        original_events = [row.model_dump() for row in session.exec(select(OperateEvent)).all()]

    yield

    with Session(engine) as session:
        readiness_bundle_ids = [
            int(row.id)
            for row in session.exec(
                select(DatasetBundle).where(DatasetBundle.name.startswith("readiness-"))
            ).all()
            if row.id is not None
        ]
        for bundle_id in readiness_bundle_ids:
            for row in session.exec(
                select(DailyConfidenceAggregate).where(
                    DailyConfidenceAggregate.bundle_id == bundle_id
                )
            ).all():
                session.delete(row)
            for row in session.exec(
                select(ProviderUpdateRun).where(ProviderUpdateRun.bundle_id == bundle_id)
            ).all():
                session.delete(row)
            for row in session.exec(
                select(DataQualityReport).where(DataQualityReport.bundle_id == bundle_id)
            ).all():
                session.delete(row)
            for row in session.exec(
                select(PaperRun).where(PaperRun.bundle_id == bundle_id)
            ).all():
                session.delete(row)
        for row in session.exec(
            select(Policy).where(Policy.name.startswith("readiness-policy-"))
        ).all():
            session.delete(row)
        for row in session.exec(
            select(DatasetBundle).where(DatasetBundle.name.startswith("readiness-"))
        ).all():
            session.delete(row)

        for row in session.exec(select(OperateEvent)).all():
            session.delete(row)
        for item in original_events:
            session.add(OperateEvent(**item))

        current_state = session.get(PaperState, 1)
        if original_state_dump is None:
            if current_state is not None:
                session.delete(current_state)
        else:
            if current_state is None:
                current_state = PaperState(**original_state_dump)
            else:
                for key, value in original_state_dump.items():
                    setattr(current_state, key, value)
            session.add(current_state)
        session.commit()


def _client() -> TestClient:
    os.environ["ATLAS_JOBS_INLINE"] = "true"
    os.environ.pop("ATLAS_FAST_MODE", None)
    os.environ.pop("ATLAS_E2E_FAST", None)
    get_settings.cache_clear()
    return TestClient(app)


def _seed_ready_context() -> tuple[int, int]:
    init_db()
    settings = get_settings()
    suffix = uuid4().hex[:8]
    now = datetime.now(timezone.utc)
    with Session(engine) as session:
        for row in session.exec(select(OperateEvent)).all():
            session.delete(row)

        bundle = DatasetBundle(
            name=f"readiness-ready-{suffix}",
            provider="test",
            symbols_json=[f"READY_{suffix.upper()}"],
            supported_timeframes_json=["1d"],
        )
        policy = Policy(
            name=f"readiness-policy-{suffix}",
            definition_json={"regime_map": {"TREND_UP": {"strategy_key": "trend_breakout"}}},
        )
        session.add(bundle)
        session.add(policy)
        session.commit()
        session.refresh(bundle)
        session.refresh(policy)

        state = get_or_create_paper_state(session, settings)
        state.equity = 1_000_000.0
        state.cash = 1_000_000.0
        state.peak_equity = 1_000_000.0
        state.drawdown = 0.0
        state.kill_switch_active = False
        state.cooldown_days_left = 0
        state.settings_json = {
            **(state.settings_json or {}),
            "active_bundle_id": int(bundle.id or 0),
            "paper_mode": "policy",
            "active_policy_id": int(policy.id or 0),
            "active_ensemble_id": None,
            "operate_safe_mode_on_fail": True,
            "operate_safe_mode_action": "shadow_only",
            "risk_per_trade": 0.005,
            "max_positions": 3,
            "kill_switch_dd": 0.08,
            "commission_bps": 5.0,
            "slippage_base_bps": 2.0,
            "cost_model_enabled": True,
            "operate_auto_run_enabled": True,
            "operate_auto_run_shadow_only": True,
            "operate_auto_eval_auto_switch": False,
            "operate_auto_eval_shadow_only_gate": True,
            "data_updates_provider_enabled": True,
        }
        session.add(state)
        session.add(
            DataQualityReport(
                bundle_id=int(bundle.id or 0),
                timeframe="1d",
                status="OK",
                last_bar_ts=now,
                coverage_pct=100.0,
                checked_symbols=1,
                total_symbols=1,
            )
        )
        session.add(
            ProviderUpdateRun(
                bundle_id=int(bundle.id or 0),
                timeframe="1d",
                provider_kind="MOCK",
                status="SUCCEEDED",
                symbols_attempted=1,
                symbols_succeeded=1,
                symbols_failed=0,
                coverage_before_pct=100.0,
                coverage_after_pct=100.0,
                ended_at=now,
            )
        )
        session.add(
            DailyConfidenceAggregate(
                bundle_id=int(bundle.id or 0),
                timeframe="1d",
                trading_date=date.today(),
                eligible_symbols_count=1,
                avg_confidence=95.0,
                pct_low_confidence=0.0,
                provider_mix_json={"MOCK": 1},
                gate_decision="PASS",
                confidence_risk_scale=1.0,
            )
        )
        session.add(
            PaperRun(
                bundle_id=int(bundle.id or 0),
                policy_id=int(policy.id or 0),
                asof_ts=now,
                mode="SHADOW",
                regime="TREND_UP",
                summary_json={"timeframes": ["1d"], "quality_status": "PASS"},
            )
        )
        session.commit()
        return int(bundle.id or 0), int(policy.id or 0)


def test_operate_readiness_blocks_real_money_by_design() -> None:
    with _client() as client:
        response = client.get("/api/operate/readiness")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["target"] == "PRODUCTION_PAPER"
    assert payload["real_money"]["verdict"] == "BLOCKED"
    assert "broker execution adapter" in payload["real_money"]["required_controls"][0].lower()


def test_operate_readiness_can_be_ready_for_production_paper() -> None:
    bundle_id, _policy_id = _seed_ready_context()

    with _client() as client:
        response = client.get(f"/api/operate/readiness?bundle_id={bundle_id}&timeframe=1d")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["verdict"] == "READY"
    assert payload["can_run_production_paper"] is True
    assert payload["summary"]["fail"] == 0
    assert payload["summary"]["warn"] == 0


def test_operate_readiness_blocks_unsafe_scheduler_and_risk() -> None:
    bundle_id, _policy_id = _seed_ready_context()
    with Session(engine) as session:
        state = session.get(PaperState, 1)
        assert state is not None
        state.settings_json = {
            **(state.settings_json or {}),
            "active_bundle_id": bundle_id,
            "operate_auto_run_enabled": True,
            "operate_auto_run_shadow_only": False,
            "risk_per_trade": 0.02,
            "max_positions": 10,
            "kill_switch_dd": 0.2,
        }
        session.add(state)
        session.commit()

    with _client() as client:
        response = client.get(f"/api/operate/readiness?bundle_id={bundle_id}&timeframe=1d")

    assert response.status_code == 200
    payload = response.json()["data"]
    checks = {item["id"]: item for item in payload["checks"]}
    assert payload["verdict"] == "BLOCKED"
    assert checks["shadow_scheduler"]["status"] == "FAIL"
    assert checks["risk_limits"]["status"] == "FAIL"
