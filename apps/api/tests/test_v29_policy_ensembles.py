from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

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
    PolicyEnsemble,
    PolicyEnsembleMember,
    PolicySwitchEvent,
    Symbol,
)
from app.db.session import engine, init_db
from app.engine.signal_engine import SignalGenerationResult
from app.services.auto_evaluation import ACTION_SWITCH, execute_auto_evaluation
from app.services.data_store import DataStore
from app.services.paper import get_or_create_paper_state, run_paper_step
from app.services.policy_health import HEALTHY


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(start: float = 100.0, rows: int = 260) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 5_000_000),
        }
    )


def _clear_runtime_tables(session: Session) -> None:
    for row in session.exec(select(PaperPosition)).all():
        session.delete(row)
    for row in session.exec(select(PaperOrder)).all():
        session.delete(row)
    for row in session.exec(select(PaperRun)).all():
        session.delete(row)
    for row in session.exec(select(PolicySwitchEvent)).all():
        session.delete(row)
    session.commit()


def _seed_bundle_with_symbols(*, provider: str, bundle_name: str) -> tuple[int, DataStore]:
    store = _store()
    with Session(engine) as session:
        store.save_ohlcv(
            session=session,
            symbol="ENS_A",
            timeframe="1d",
            frame=_frame(start=120.0),
            provider=provider,
            bundle_name=bundle_name,
        )
        store.save_ohlcv(
            session=session,
            symbol="ENS_B",
            timeframe="1d",
            frame=_frame(start=180.0),
            provider=provider,
            bundle_name=bundle_name,
        )
        symbol_a = session.exec(select(Symbol).where(Symbol.symbol == "ENS_A")).first()
        symbol_b = session.exec(select(Symbol).where(Symbol.symbol == "ENS_B")).first()
        if symbol_a is not None:
            symbol_a.sector = "TECH"
            session.add(symbol_a)
        if symbol_b is not None:
            symbol_b.sector = "ENERGY"
            session.add(symbol_b)
        session.commit()
        bundle = session.exec(
            select(DatasetBundle).where(DatasetBundle.name == bundle_name)
        ).first()
        assert bundle is not None and bundle.id is not None
        return int(bundle.id), store


def _seed_ensemble_context(
    session: Session,
    *,
    bundle_id: int,
    risk_scale: float = 0.1,
) -> tuple[int, int, int]:
    unique = uuid4().hex[:8]
    policy_a = Policy(
        name=f"ens-policy-a-{unique}",
        definition_json={
            "universe": {"bundle_id": bundle_id, "symbol_scope": "all", "max_symbols_scan": 20},
            "timeframes": ["1d"],
            "regime_map": {
                "TREND_UP": {
                    "strategy_key": "trend_breakout",
                    "risk_scale": risk_scale,
                    "max_positions_scale": 1.0,
                }
            },
        },
    )
    policy_b = Policy(
        name=f"ens-policy-b-{unique}",
        definition_json={
            "universe": {"bundle_id": bundle_id, "symbol_scope": "all", "max_symbols_scan": 20},
            "timeframes": ["1d"],
            "regime_map": {
                "TREND_UP": {
                    "strategy_key": "pullback_trend",
                    "risk_scale": risk_scale,
                    "max_positions_scale": 1.0,
                }
            },
        },
    )
    ensemble = PolicyEnsemble(
        name=f"ens-main-{unique}",
        bundle_id=bundle_id,
        is_active=True,
    )
    session.add(policy_a)
    session.add(policy_b)
    session.add(ensemble)
    session.commit()
    session.refresh(policy_a)
    session.refresh(policy_b)
    session.refresh(ensemble)
    assert policy_a.id is not None
    assert policy_b.id is not None
    assert ensemble.id is not None
    session.add(
        PolicyEnsembleMember(
            ensemble_id=int(ensemble.id),
            policy_id=int(policy_a.id),
            weight=0.55,
            enabled=True,
        )
    )
    session.add(
        PolicyEnsembleMember(
            ensemble_id=int(ensemble.id),
            policy_id=int(policy_b.id),
            weight=0.45,
            enabled=True,
        )
    )
    session.commit()
    return int(ensemble.id), int(policy_a.id), int(policy_b.id)


def _seed_active_policy_runs(
    session: Session,
    *,
    bundle_id: int,
    policy_id: int,
    asof: date,
    days: int = 20,
) -> None:
    equity = 1_000_000.0
    for idx in range(days):
        day = asof - timedelta(days=idx)
        net_pnl = 900.0 if idx % 2 == 0 else -500.0
        session.add(
            PaperRun(
                bundle_id=bundle_id,
                policy_id=policy_id,
                asof_ts=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                mode="LIVE",
                regime="TREND_UP",
                signals_source="generated",
                generated_signals_count=4,
                selected_signals_count=1,
                skipped_signals_count=3,
                scanned_symbols=20,
                evaluated_candidates=8,
                scan_truncated=False,
                summary_json={
                    "equity_before": equity,
                    "equity_after": equity + net_pnl,
                    "net_pnl": net_pnl,
                    "gross_pnl": net_pnl + 50.0,
                    "total_cost": 50.0,
                    "trade_count": 1,
                    "turnover": 0.15,
                    "exposure": 0.4,
                    "avg_holding_days": 3.0,
                },
                cost_summary_json={"total_cost": 50.0},
            )
        )
        equity += net_pnl
    session.commit()


def _prepare_state_for_ensemble(
    session: Session,
    *,
    settings,
    ensemble_id: int,
    max_positions: int = 3,
) -> None:
    state = get_or_create_paper_state(session, settings)
    state.equity = 1_000_000.0
    state.cash = 1_000_000.0
    state.peak_equity = 1_000_000.0
    state.drawdown = 0.0
    state.kill_switch_active = False
    state.cooldown_days_left = 0
    state.settings_json = {
        **(state.settings_json or {}),
        "paper_mode": "policy",
        "active_policy_id": None,
        "active_policy_name": None,
        "active_ensemble_id": int(ensemble_id),
        "allowed_sides": ["BUY"],
        "max_positions": int(max_positions),
        "operate_mode": "offline",
        "data_quality_stale_severity": "WARN",
        "data_quality_stale_severity_override": True,
    }
    session.add(state)
    session.commit()


def _fake_signal_generator_factory() -> callable:
    def _fake_generate_signals_for_policy(**kwargs) -> SignalGenerationResult:
        allowed_templates = kwargs.get("allowed_templates") or []
        template = str(allowed_templates[0] if allowed_templates else "trend_breakout")
        if template == "pullback_trend":
            symbol = "ENS_B"
            strength = 0.72
        else:
            symbol = "ENS_A"
            strength = 0.81
        signal = {
            "symbol": symbol,
            "underlying_symbol": symbol,
            "side": "BUY",
            "template": template,
            "timeframe": "1d",
            "instrument_kind": "EQUITY_CASH",
            "price": 220.0 if symbol == "ENS_A" else 180.0,
            "stop_distance": 5.0,
            "target_price": None,
            "signal_strength": strength,
            "adv": 10_000_000_000.0,
            "vol_scale": 0.0,
            "explanation": f"{template} deterministic signal",
        }
        return SignalGenerationResult(
            signals=[signal],
            scan_truncated=False,
            scanned_symbols=2,
            evaluated_candidates=1,
            total_symbols=2,
        )

    return _fake_generate_signals_for_policy


def test_ensemble_allocation_is_deterministic_for_same_seed(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    provider = f"v29-det-{uuid4().hex[:8]}"
    bundle_name = f"bundle-v29-det-{provider}"
    bundle_id, store = _seed_bundle_with_symbols(provider=provider, bundle_name=bundle_name)

    with Session(engine) as session:
        _clear_runtime_tables(session)
        ensemble_id, _, _ = _seed_ensemble_context(session, bundle_id=bundle_id)
        _prepare_state_for_ensemble(session, settings=settings, ensemble_id=ensemble_id, max_positions=3)
        monkeypatch.setattr(
            "app.services.paper.generate_signals_for_policy",
            _fake_signal_generator_factory(),
        )

        payload = {
            "regime": "TREND_UP",
            "signals": [],
            "auto_generate_signals": True,
            "bundle_id": bundle_id,
            "timeframes": ["1d"],
            "seed": 17,
        }
        first = run_paper_step(session, settings, payload=payload, store=store)

        _clear_runtime_tables(session)
        _prepare_state_for_ensemble(session, settings=settings, ensemble_id=ensemble_id, max_positions=3)
        second = run_paper_step(session, settings, payload=payload, store=store)

        first_ensemble = first.get("ensemble", {}) if isinstance(first.get("ensemble", {}), dict) else {}
        second_ensemble = second.get("ensemble", {}) if isinstance(second.get("ensemble", {}), dict) else {}
        assert first_ensemble.get("risk_budget_by_policy") == second_ensemble.get("risk_budget_by_policy")
        assert first_ensemble.get("selected_counts_by_policy") == second_ensemble.get("selected_counts_by_policy")
        first_order = [int(item.get("source_policy_id") or 0) for item in first.get("selected_signals", [])]
        second_order = [int(item.get("source_policy_id") or 0) for item in second.get("selected_signals", [])]
        assert first_order == second_order


def test_ensemble_global_caps_apply_across_members(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    provider = f"v29-cap-{uuid4().hex[:8]}"
    bundle_name = f"bundle-v29-cap-{provider}"
    bundle_id, store = _seed_bundle_with_symbols(provider=provider, bundle_name=bundle_name)

    with Session(engine) as session:
        _clear_runtime_tables(session)
        ensemble_id, _, _ = _seed_ensemble_context(session, bundle_id=bundle_id)
        _prepare_state_for_ensemble(session, settings=settings, ensemble_id=ensemble_id, max_positions=1)
        monkeypatch.setattr(
            "app.services.paper.generate_signals_for_policy",
            _fake_signal_generator_factory(),
        )

        result = run_paper_step(
            session,
            settings,
            payload={
                "regime": "TREND_UP",
                "signals": [],
                "auto_generate_signals": True,
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "seed": 21,
            },
            store=store,
        )
        assert int(result.get("selected_signals_count", 0)) == 1
        assert any(
            str(item.get("reason")) == "max_positions_reached"
            for item in (result.get("skipped_signals") or [])
            if isinstance(item, dict)
        )


def test_auto_eval_recommends_ensemble_when_score_improves(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    asof_day = date(2026, 2, 13)
    unique = uuid4().hex[:8]

    with Session(engine) as session:
        _clear_runtime_tables(session)
        bundle = DatasetBundle(
            name=f"bundle-v29-eval-{unique}",
            provider="test",
            symbols_json=["ENS_A", "ENS_B"],
            supported_timeframes_json=["1d"],
        )
        active = Policy(
            name=f"active-v29-{unique}",
            definition_json={
                "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "baseline": {"max_drawdown": -0.08, "cvar_95": -0.03},
                "regime_map": {"TREND_UP": {"strategy_key": "trend_breakout"}},
            },
        )
        chal_a = Policy(
            name=f"challenger-a-v29-{unique}",
            definition_json={
                "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "regime_map": {"TREND_UP": {"strategy_key": "pullback_trend"}},
            },
        )
        chal_b = Policy(
            name=f"challenger-b-v29-{unique}",
            definition_json={
                "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "regime_map": {"TREND_UP": {"strategy_key": "squeeze_breakout"}},
            },
        )
        session.add(bundle)
        session.add(active)
        session.add(chal_a)
        session.add(chal_b)
        session.commit()
        session.refresh(bundle)
        session.refresh(active)
        session.refresh(chal_a)
        session.refresh(chal_b)
        assert bundle.id is not None
        assert active.id is not None
        assert chal_a.id is not None
        assert chal_b.id is not None

        active.definition_json = {
            **(active.definition_json or {}),
            "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 10},
        }
        chal_a.definition_json = {
            **(chal_a.definition_json or {}),
            "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 10},
        }
        chal_b.definition_json = {
            **(chal_b.definition_json or {}),
            "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 10},
        }
        session.add(active)
        session.add(chal_a)
        session.add(chal_b)
        session.commit()

        ensemble = PolicyEnsemble(
            name=f"ensemble-v29-{unique}",
            bundle_id=int(bundle.id),
            is_active=False,
        )
        session.add(ensemble)
        session.commit()
        session.refresh(ensemble)
        assert ensemble.id is not None
        session.add(
            PolicyEnsembleMember(
                ensemble_id=int(ensemble.id),
                policy_id=int(chal_a.id),
                weight=0.5,
                enabled=True,
            )
        )
        session.add(
            PolicyEnsembleMember(
                ensemble_id=int(ensemble.id),
                policy_id=int(chal_b.id),
                weight=0.5,
                enabled=True,
            )
        )
        session.commit()

        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "paper_mode": "policy",
            "active_policy_id": int(active.id),
            "active_policy_name": active.name,
            "active_ensemble_id": None,
            "operate_mode": "live",
            "operate_auto_eval_auto_switch": False,
            "operate_auto_eval_shadow_only_gate": True,
            "operate_auto_eval_cooldown_trading_days": 10,
            "operate_auto_eval_max_switches_per_30d": 2,
            "operate_auto_eval_lookback_trading_days": 20,
            "operate_auto_eval_min_trades": 8,
            "evaluations_score_margin": 0.0,
            "evaluations_max_dd_multiplier": 1.1,
        }
        session.add(state)
        session.commit()

        _seed_active_policy_runs(
            session,
            bundle_id=int(bundle.id),
            policy_id=int(active.id),
            asof=asof_day,
            days=20,
        )

        def _fake_simulate(**kwargs):
            policy = kwargs["policy"]
            policy_id = int(policy.id)
            if policy_id == int(active.id):
                return {
                    "metrics": {
                        "calmar": 0.5,
                        "max_drawdown": -0.08,
                        "profit_factor": 1.1,
                        "cvar_95": -0.04,
                        "turnover": 0.4,
                    },
                    "symbol_rows": [{"trade_count": 20}],
                    "engine_version": "atlas-sim-v2.9-test",
                    "data_digest": f"digest-{policy_id}",
                }
            return {
                "metrics": {
                    "calmar": 5.0,
                    "max_drawdown": -0.02,
                    "profit_factor": 2.8,
                    "cvar_95": -0.01,
                    "turnover": 0.12,
                },
                "symbol_rows": [{"trade_count": 18}],
                "engine_version": "atlas-sim-v2.9-test",
                "data_digest": f"digest-{policy_id}",
            }

        monkeypatch.setattr("app.services.auto_evaluation.simulate_policy_on_bundle", _fake_simulate)
        monkeypatch.setattr(
            "app.services.auto_evaluation.compute_health_metrics",
            lambda runs, window_days: {
                "calmar": 0.1,
                "max_drawdown": -0.2,
                "profit_factor": 0.9,
                "cvar_95": -0.02,
                "turnover": 0.5,
                "trade_count": 20,
            },
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=HEALTHY, reasons_json=[]),
        )

        result = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={
                "bundle_id": int(bundle.id),
                "asof_date": asof_day.isoformat(),
                "seed": 7,
                "challenger_ensemble_ids": [int(ensemble.id)],
            },
        )
        assert result["recommended_action"] == ACTION_SWITCH
        assert result["recommended_entity_type"] == "ensemble"
        assert int(result["recommended_ensemble_id"] or 0) == int(ensemble.id)


def test_auto_eval_ensemble_switch_respects_cooldown(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store()
    asof_day = date(2026, 2, 13)
    unique = uuid4().hex[:8]

    with Session(engine) as session:
        _clear_runtime_tables(session)
        bundle = DatasetBundle(
            name=f"bundle-v29-cooldown-{unique}",
            provider="test",
            symbols_json=["ENS_A", "ENS_B"],
            supported_timeframes_json=["1d"],
        )
        active = Policy(
            name=f"active-cooldown-{unique}",
            definition_json={
                "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "baseline": {"max_drawdown": -0.08, "cvar_95": -0.03},
                "regime_map": {"TREND_UP": {"strategy_key": "trend_breakout"}},
            },
        )
        switch_from = Policy(
            name=f"switch-from-{unique}",
            definition_json={"regime_map": {"TREND_UP": {"strategy_key": "trend_breakout"}}},
        )
        switch_to = Policy(
            name=f"switch-to-{unique}",
            definition_json={"regime_map": {"TREND_UP": {"strategy_key": "trend_breakout"}}},
        )
        chal_a = Policy(
            name=f"challenger-cooldown-a-{unique}",
            definition_json={
                "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "regime_map": {"TREND_UP": {"strategy_key": "pullback_trend"}},
            },
        )
        chal_b = Policy(
            name=f"challenger-cooldown-b-{unique}",
            definition_json={
                "universe": {"bundle_id": None, "symbol_scope": "all", "max_symbols_scan": 10},
                "timeframes": ["1d"],
                "regime_map": {"TREND_UP": {"strategy_key": "squeeze_breakout"}},
            },
        )
        session.add(bundle)
        session.add(active)
        session.add(switch_from)
        session.add(switch_to)
        session.add(chal_a)
        session.add(chal_b)
        session.commit()
        session.refresh(bundle)
        session.refresh(active)
        session.refresh(switch_from)
        session.refresh(switch_to)
        session.refresh(chal_a)
        session.refresh(chal_b)
        assert bundle.id is not None
        assert active.id is not None
        assert switch_from.id is not None
        assert switch_to.id is not None
        assert chal_a.id is not None
        assert chal_b.id is not None

        active.definition_json = {
            **(active.definition_json or {}),
            "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 10},
        }
        chal_a.definition_json = {
            **(chal_a.definition_json or {}),
            "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 10},
        }
        chal_b.definition_json = {
            **(chal_b.definition_json or {}),
            "universe": {"bundle_id": int(bundle.id), "symbol_scope": "all", "max_symbols_scan": 10},
        }
        session.add(active)
        session.add(chal_a)
        session.add(chal_b)
        session.commit()

        ensemble = PolicyEnsemble(
            name=f"ensemble-cooldown-{unique}",
            bundle_id=int(bundle.id),
            is_active=False,
        )
        session.add(ensemble)
        session.commit()
        session.refresh(ensemble)
        assert ensemble.id is not None
        session.add(
            PolicyEnsembleMember(
                ensemble_id=int(ensemble.id),
                policy_id=int(chal_a.id),
                weight=0.6,
                enabled=True,
            )
        )
        session.add(
            PolicyEnsembleMember(
                ensemble_id=int(ensemble.id),
                policy_id=int(chal_b.id),
                weight=0.4,
                enabled=True,
            )
        )
        session.add(
            PolicySwitchEvent(
                from_policy_id=int(switch_from.id),
                to_policy_id=int(switch_to.id),
                reason="recent-switch",
                auto_eval_id=None,
                cooldown_state_json={},
                mode="MANUAL",
                ts=datetime(2026, 2, 12, 12, 0, tzinfo=timezone.utc),
            )
        )
        session.commit()

        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "paper_mode": "policy",
            "active_policy_id": int(active.id),
            "active_policy_name": active.name,
            "active_ensemble_id": None,
            "operate_mode": "live",
            "operate_auto_eval_auto_switch": True,
            "operate_auto_eval_shadow_only_gate": True,
            "operate_auto_eval_cooldown_trading_days": 10,
            "operate_auto_eval_max_switches_per_30d": 2,
            "operate_auto_eval_lookback_trading_days": 20,
            "operate_auto_eval_min_trades": 8,
            "evaluations_score_margin": 0.0,
            "evaluations_max_dd_multiplier": 1.1,
        }
        session.add(state)
        session.commit()

        _seed_active_policy_runs(
            session,
            bundle_id=int(bundle.id),
            policy_id=int(active.id),
            asof=asof_day,
            days=20,
        )

        def _fake_simulate(**kwargs):
            policy = kwargs["policy"]
            policy_id = int(policy.id)
            if policy_id == int(active.id):
                return {
                    "metrics": {
                        "calmar": 0.5,
                        "max_drawdown": -0.08,
                        "profit_factor": 1.1,
                        "cvar_95": -0.04,
                        "turnover": 0.4,
                    },
                    "symbol_rows": [{"trade_count": 20}],
                    "engine_version": "atlas-sim-v2.9-test",
                    "data_digest": f"digest-{policy_id}",
                }
            return {
                "metrics": {
                    "calmar": 4.5,
                    "max_drawdown": -0.02,
                    "profit_factor": 2.4,
                    "cvar_95": -0.01,
                    "turnover": 0.12,
                },
                "symbol_rows": [{"trade_count": 18}],
                "engine_version": "atlas-sim-v2.9-test",
                "data_digest": f"digest-{policy_id}",
            }

        monkeypatch.setattr("app.services.auto_evaluation.simulate_policy_on_bundle", _fake_simulate)
        monkeypatch.setattr(
            "app.services.auto_evaluation.compute_health_metrics",
            lambda runs, window_days: {
                "calmar": 0.1,
                "max_drawdown": -0.2,
                "profit_factor": 0.9,
                "cvar_95": -0.02,
                "turnover": 0.5,
                "trade_count": 20,
            },
        )
        monkeypatch.setattr(
            "app.services.auto_evaluation.get_policy_health_snapshot",
            lambda *args, **kwargs: SimpleNamespace(status=HEALTHY, reasons_json=[]),
        )

        result = execute_auto_evaluation(
            session=session,
            store=store,
            settings=settings,
            payload={
                "bundle_id": int(bundle.id),
                "asof_date": asof_day.isoformat(),
                "seed": 7,
                "challenger_ensemble_ids": [int(ensemble.id)],
                "auto_switch": True,
            },
        )
        assert result["recommended_action"] == ACTION_SWITCH
        assert result["recommended_entity_type"] == "ensemble"
        assert int(result["recommended_ensemble_id"] or 0) == int(ensemble.id)
        assert result["auto_switch_applied"] is False
        assert any("cooldown" in str(reason).lower() for reason in result["reasons"])
