from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import (
    AutoEvalRun,
    DataQualityReport,
    DatasetBundle,
    PaperRun,
    Policy,
    PolicySwitchEvent,
)
from app.db.session import engine, init_db
from app.services.auto_evaluation import ACTION_SHADOW_ONLY, ACTION_SWITCH, execute_auto_evaluation
from app.services.data_store import DataStore
from app.services.paper import get_or_create_paper_state
from app.services.policy_health import DEGRADED, HEALTHY


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _clear_auto_eval_tables(session: Session) -> None:
    for row in session.exec(select(AutoEvalRun)).all():
        session.delete(row)
    for row in session.exec(select(PolicySwitchEvent)).all():
        session.delete(row)
    for row in session.exec(select(DataQualityReport)).all():
        session.delete(row)
    session.commit()


def _seed_context(session: Session) -> tuple[int, int, int]:
    settings = get_settings()
    _clear_auto_eval_tables(session)
    unique = uuid4().hex[:8]
    bundle = DatasetBundle(
        name=f"bundle-v25-{unique}",
        provider="test",
        description="v2.5 auto eval test bundle",
        symbols_json=[f"SYM_{unique[:3].upper()}"],
        supported_timeframes_json=["1d"],
    )
    active = Policy(
        name=f"active-{unique}",
        definition_json={
            "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 5},
            "timeframes": ["1d"],
            "baseline": {"max_drawdown": -0.08, "cvar_95": -0.03},
            "regime_map": {"TREND_UP": {"strategy_key": "trend_breakout"}},
        },
    )
    challenger = Policy(
        name=f"challenger-{unique}",
        definition_json={
            "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 5},
            "timeframes": ["1d"],
            "regime_map": {"TREND_UP": {"strategy_key": "pullback_trend"}},
        },
    )
    session.add(bundle)
    session.add(active)
    session.add(challenger)
    session.commit()
    session.refresh(bundle)
    session.refresh(active)
    session.refresh(challenger)
    assert bundle.id is not None
    assert active.id is not None
    assert challenger.id is not None

    active.definition_json = {
        **(active.definition_json or {}),
        "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 5},
    }
    challenger.definition_json = {
        **(challenger.definition_json or {}),
        "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 5},
    }
    session.add(active)
    session.add(challenger)
    session.commit()

    state = get_or_create_paper_state(session, settings)
    state.settings_json = {
        **(state.settings_json or {}),
        "paper_mode": "policy",
        "active_policy_id": int(active.id),
        "active_policy_name": active.name,
        "operate_mode": "live",
        "operate_auto_eval_auto_switch": False,
        "operate_auto_eval_shadow_only_gate": True,
        "operate_auto_eval_cooldown_trading_days": 10,
        "operate_auto_eval_max_switches_per_30d": 2,
        "operate_auto_eval_lookback_trading_days": 20,
        "operate_auto_eval_min_trades": 8,
        "evaluations_score_margin": 0.05,
        "evaluations_max_dd_multiplier": 1.1,
    }
    session.add(state)
    session.commit()

    asof = date(2026, 2, 13)
    equity = 1_000_000.0
    for idx in range(20):
        day = asof - timedelta(days=idx)
        net_pnl = 1_000.0 if idx % 2 == 0 else -900.0
        gross_pnl = net_pnl + 120.0
        session.add(
            PaperRun(
                bundle_id=int(bundle.id),
                policy_id=int(active.id),
                asof_ts=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                mode="LIVE",
                regime="TREND_UP",
                summary_json={
                    "equity_before": equity,
                    "equity_after": equity + net_pnl,
                    "net_pnl": net_pnl,
                    "gross_pnl": gross_pnl,
                    "total_cost": 120.0,
                    "trade_count": 1,
                    "turnover": 0.2,
                    "exposure": 0.4,
                    "avg_holding_days": 3.0,
                },
                cost_summary_json={"total_cost": 120.0},
            )
        )
        equity += net_pnl
    session.commit()
    return int(bundle.id), int(active.id), int(challenger.id)


def _mock_simulate(*, policy_id: int, challenger_id: int) -> dict[str, object]:
    if policy_id == challenger_id:
        return {
            "metrics": {
                "calmar": 8.0,
                "max_drawdown": -0.0005,
                "profit_factor": 4.5,
                "cvar_95": -0.005,
                "turnover": 0.05,
            },
            "symbol_rows": [{"trade_count": 20}],
            "engine_version": "atlas-sim-v2.5-test",
            "data_digest": "digest-challenger",
        }
    return {
        "metrics": {
            "calmar": 0.5,
            "max_drawdown": -0.08,
            "profit_factor": 1.1,
            "cvar_95": -0.04,
            "turnover": 0.4,
        },
        "symbol_rows": [{"trade_count": 5}],
        "engine_version": "atlas-sim-v2.5-test",
        "data_digest": "digest-active",
    }


def test_auto_eval_min_trades_gate_blocks_switch(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        bundle_id, _active_id, challenger_id = _seed_context(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_eval_min_trades": 999,
        }
        session.add(state)
        session.commit()

        monkeypatch.setattr(
            "app.services.auto_evaluation.simulate_policy_on_bundle",
            lambda **kwargs: _mock_simulate(
                policy_id=int(kwargs["policy"].id),
                challenger_id=challenger_id,
            ),
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=HEALTHY, reasons_json=[]),
        )

        result = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={"bundle_id": bundle_id, "asof_date": "2026-02-13", "seed": 7},
        )
        assert result["recommended_action"] == "KEEP"
        assert any("insufficient" in str(reason).lower() for reason in result["reasons"])


def test_auto_eval_degraded_policy_recommends_shadow_only(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        bundle_id, _active_id, challenger_id = _seed_context(session)
        monkeypatch.setattr(
            "app.services.auto_evaluation.simulate_policy_on_bundle",
            lambda **kwargs: _mock_simulate(
                policy_id=int(kwargs["policy"].id),
                challenger_id=challenger_id,
            ),
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=DEGRADED, reasons_json=["degraded"]),
        )

        result = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={"bundle_id": bundle_id, "asof_date": "2026-02-13", "seed": 7},
        )
        assert result["recommended_action"] == ACTION_SHADOW_ONLY
        assert any("degraded" in str(reason).lower() for reason in result["reasons"])


def test_auto_eval_auto_switch_false_keeps_active_policy(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        bundle_id, active_id, challenger_id = _seed_context(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_eval_auto_switch": False,
        }
        session.add(state)
        session.commit()

        monkeypatch.setattr(
            "app.services.auto_evaluation.simulate_policy_on_bundle",
            lambda **kwargs: _mock_simulate(
                policy_id=int(kwargs["policy"].id),
                challenger_id=challenger_id,
            ),
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=HEALTHY, reasons_json=[]),
        )

        result = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={"bundle_id": bundle_id, "asof_date": "2026-02-13", "seed": 7},
        )
        assert result["recommended_action"] == ACTION_SWITCH
        assert result["auto_switch_applied"] is False

        refreshed = get_or_create_paper_state(session, settings)
        assert int((refreshed.settings_json or {}).get("active_policy_id") or 0) == active_id


def test_auto_eval_auto_switch_true_switches_under_gates(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        bundle_id, _active_id, challenger_id = _seed_context(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_eval_auto_switch": True,
            "operate_mode": "live",
        }
        session.add(state)
        session.commit()

        monkeypatch.setattr(
            "app.services.auto_evaluation.simulate_policy_on_bundle",
            lambda **kwargs: _mock_simulate(
                policy_id=int(kwargs["policy"].id),
                challenger_id=challenger_id,
            ),
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=HEALTHY, reasons_json=[]),
        )

        result = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={"bundle_id": bundle_id, "asof_date": "2026-02-13", "seed": 7},
        )
        assert result["recommended_action"] == ACTION_SWITCH
        assert result["auto_switch_applied"] is True
        assert int(result["switched_to_policy_id"] or 0) == challenger_id
        refreshed = get_or_create_paper_state(session, settings)
        assert int((refreshed.settings_json or {}).get("active_policy_id") or 0) == challenger_id


def test_auto_eval_cooldown_and_max_switch_gates(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        bundle_id, active_id, challenger_id = _seed_context(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_eval_auto_switch": True,
            "operate_mode": "live",
            "operate_auto_eval_cooldown_trading_days": 10,
            "operate_auto_eval_max_switches_per_30d": 1,
        }
        session.add(state)
        session.add(
            PolicySwitchEvent(
                from_policy_id=active_id,
                to_policy_id=challenger_id,
                reason="recent-switch",
                auto_eval_id=None,
                cooldown_state_json={},
                mode="MANUAL",
                ts=datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc),
            )
        )
        session.commit()

        monkeypatch.setattr(
            "app.services.auto_evaluation.simulate_policy_on_bundle",
            lambda **kwargs: _mock_simulate(
                policy_id=int(kwargs["policy"].id),
                challenger_id=challenger_id,
            ),
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=HEALTHY, reasons_json=[]),
        )

        result = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={"bundle_id": bundle_id, "asof_date": "2026-02-13", "seed": 7},
        )
        assert result["recommended_action"] == ACTION_SWITCH
        assert result["auto_switch_applied"] is False
        assert any("cooldown" in str(reason).lower() for reason in result["reasons"])
        assert any("max switches" in str(reason).lower() for reason in result["reasons"])


def test_auto_eval_is_deterministic_for_same_seed(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    with Session(engine) as session:
        bundle_id, _active_id, challenger_id = _seed_context(session)
        monkeypatch.setattr(
            "app.services.auto_evaluation.simulate_policy_on_bundle",
            lambda **kwargs: _mock_simulate(
                policy_id=int(kwargs["policy"].id),
                challenger_id=challenger_id,
            ),
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=HEALTHY, reasons_json=[]),
        )

        first = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={"bundle_id": bundle_id, "asof_date": "2026-02-13", "seed": 19},
        )
        second = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={"bundle_id": bundle_id, "asof_date": "2026-02-13", "seed": 19},
        )
        assert first["recommended_action"] == second["recommended_action"]
        assert first["digest"] == second["digest"]
