from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd
import pytest
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import Dataset, PaperOrder, PaperPosition, PaperRun, PaperState, Policy, PolicyEnsemble, PolicyEnsembleMember
from app.db.session import engine, init_db
from app.engine.signal_engine import SignalGenerationResult
from app.services.data_store import DataStore
from app.services.ensembles import upsert_policy_ensemble_regime_weights
from app.services.no_trade import evaluate_no_trade_gate
from app.services.paper import get_or_create_paper_state, run_paper_step


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _frame(start: float = 100.0, rows: int = 280) -> pd.DataFrame:
    idx = pd.date_range("2022-01-01", periods=rows, freq="D", tz="UTC")
    close = np.linspace(start, start + rows - 1, rows)
    return pd.DataFrame(
        {
            "datetime": idx,
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.full(rows, 4_000_000),
        }
    )


def _clear_runtime(session: Session) -> None:
    for row in session.exec(select(PaperPosition)).all():
        session.delete(row)
    for row in session.exec(select(PaperOrder)).all():
        session.delete(row)
    for row in session.exec(select(PaperRun)).all():
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
            "operate_mode": "offline",
            "data_quality_stale_severity": "WARN",
            "data_quality_stale_severity_override": True,
            "no_trade_enabled": False,
        }
        session.add(state)
    session.commit()


def _seed_bundle(*, provider: str, bundle_name: str) -> tuple[int, DataStore]:
    store = _store()
    with Session(engine) as session:
        for symbol, start in (("V30_A", 120.0), ("V30_B", 180.0)):
            store.save_ohlcv(
                session=session,
                symbol=symbol,
                timeframe="1d",
                frame=_frame(start=start),
                provider=provider,
                bundle_name=bundle_name,
            )
        bundle = session.exec(select(Dataset).where(Dataset.provider == provider)).first()
        assert bundle is not None and bundle.bundle_id is not None
        return int(bundle.bundle_id), store


def _seed_ensemble(session: Session, *, bundle_id: int) -> tuple[int, int, int]:
    token = uuid4().hex[:8]
    pol_a = Policy(
        name=f"v30-policy-a-{token}",
        definition_json={
            "universe": {"bundle_id": bundle_id, "symbol_scope": "all", "max_symbols_scan": 20},
            "timeframes": ["1d"],
            "regime_map": {"TREND_UP": {"strategy_key": "trend_breakout", "risk_scale": 1.0}},
        },
    )
    pol_b = Policy(
        name=f"v30-policy-b-{token}",
        definition_json={
            "universe": {"bundle_id": bundle_id, "symbol_scope": "all", "max_symbols_scan": 20},
            "timeframes": ["1d"],
            "regime_map": {"TREND_UP": {"strategy_key": "pullback_trend", "risk_scale": 1.0}},
        },
    )
    ensemble = PolicyEnsemble(name=f"v30-ensemble-{token}", bundle_id=bundle_id, is_active=True)
    session.add(pol_a)
    session.add(pol_b)
    session.add(ensemble)
    session.commit()
    session.refresh(pol_a)
    session.refresh(pol_b)
    session.refresh(ensemble)
    assert pol_a.id is not None and pol_b.id is not None and ensemble.id is not None
    session.add(
        PolicyEnsembleMember(
            ensemble_id=int(ensemble.id),
            policy_id=int(pol_a.id),
            weight=0.5,
            enabled=True,
        )
    )
    session.add(
        PolicyEnsembleMember(
            ensemble_id=int(ensemble.id),
            policy_id=int(pol_b.id),
            weight=0.5,
            enabled=True,
        )
    )
    session.commit()
    return int(ensemble.id), int(pol_a.id), int(pol_b.id)


def _fake_signal_generator() -> callable:
    def _inner(**kwargs) -> SignalGenerationResult:
        allowed = kwargs.get("allowed_templates") or []
        template = str(allowed[0] if allowed else "trend_breakout")
        symbol = "V30_B" if template == "pullback_trend" else "V30_A"
        signal = {
            "symbol": symbol,
            "underlying_symbol": symbol,
            "side": "BUY",
            "template": template,
            "timeframe": "1d",
            "instrument_kind": "EQUITY_CASH",
            "price": 220.0,
            "stop_distance": 5.0,
            "target_price": None,
            "signal_strength": 0.8 if symbol == "V30_B" else 0.6,
            "adv": 20_000_000_000.0,
            "vol_scale": 0.0,
            "explanation": "v30 deterministic",
        }
        return SignalGenerationResult(
            signals=[signal],
            scan_truncated=False,
            scanned_symbols=2,
            evaluated_candidates=1,
            total_symbols=2,
        )

    return _inner


def test_regime_weights_applied_deterministically(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    bundle_id, store = _seed_bundle(
        provider=f"v30-regime-{uuid4().hex[:8]}",
        bundle_name=f"v30-regime-bundle-{uuid4().hex[:6]}",
    )
    with Session(engine) as session:
        _clear_runtime(session)
        ensemble_id, policy_a_id, policy_b_id = _seed_ensemble(session, bundle_id=bundle_id)
        upsert_policy_ensemble_regime_weights(
            session,
            ensemble_id=ensemble_id,
            payload={
                "TREND_UP": {
                    str(policy_a_id): 0.2,
                    str(policy_b_id): 0.8,
                }
            },
        )
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "paper_mode": "policy",
            "active_policy_id": None,
            "active_ensemble_id": ensemble_id,
            "no_trade_enabled": False,
            "max_positions": 2,
        }
        session.add(state)
        session.commit()

        monkeypatch.setattr("app.services.paper.generate_signals_for_policy", _fake_signal_generator())
        result = run_paper_step(
            session,
            settings,
            payload={
                "regime": "TREND_UP",
                "asof": "2022-10-05T10:00:00+05:30",
                "signals": [],
                "auto_generate_signals": True,
                "bundle_id": bundle_id,
                "timeframes": ["1d"],
                "seed": 31,
            },
            store=store,
        )
        ensemble = result.get("ensemble", {})
        assert isinstance(ensemble, dict)
        assert ensemble.get("weights_source") == "regime"
        budgets = ensemble.get("risk_budget_by_policy", {})
        assert isinstance(budgets, dict)
        a_budget = float(budgets.get(str(policy_a_id), 0.0))
        b_budget = float(budgets.get(str(policy_b_id), 0.0))
        assert b_budget > a_budget > 0.0
        assert b_budget == pytest.approx(a_budget * 4.0, rel=1e-2)


def test_no_trade_gate_blocks_entries_but_allows_exits() -> None:
    init_db()
    settings = get_settings()
    with Session(engine) as session:
        _clear_runtime(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "allowed_sides": ["BUY", "SELL"],
            "paper_short_squareoff_time": "15:20",
            "no_trade_enabled": False,
        }
        session.add(state)
        session.commit()

        open_result = run_paper_step(
            session,
            settings,
            payload={
                "regime": "TREND_UP",
                "asof": "2026-02-16T10:00:00+05:30",
                "signals": [
                    {
                        "symbol": "V30_SHORT",
                        "side": "SELL",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 100.0,
                        "stop_distance": 5.0,
                        "signal_strength": 0.9,
                        "adv": 10_000_000_000.0,
                        "vol_scale": 0.0,
                    }
                ],
            },
            store=_store(),
        )
        assert int(open_result.get("selected_signals_count", 0)) == 1
        assert any(
            row.get("side") == "SELL" and row.get("must_exit_by_eod") is True
            for row in (open_result.get("positions") or [])
        )

        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "no_trade_enabled": True,
            "no_trade_regimes": ["TREND_UP"],
            "no_trade_cooldown_trading_days": 2,
            "no_trade_max_realized_vol_annual": 1.0,
            "no_trade_min_breadth_pct": 0.0,
            "no_trade_min_trend_strength": 0.0,
        }
        session.add(state)
        session.commit()

        close_result = run_paper_step(
            session,
            settings,
            payload={
                "regime": "TREND_UP",
                "asof": "2026-02-16T15:25:00+05:30",
                "signals": [
                    {
                        "symbol": "V30_NEW_ENTRY",
                        "side": "BUY",
                        "template": "trend_breakout",
                        "instrument_kind": "EQUITY_CASH",
                        "price": 101.0,
                        "stop_distance": 4.0,
                        "signal_strength": 0.7,
                        "adv": 10_000_000_000.0,
                        "vol_scale": 0.0,
                    }
                ],
                "mark_prices": {"V30_SHORT": 97.0},
            },
            store=_store(),
        )
        assert bool((close_result.get("no_trade") or {}).get("triggered", False)) is True
        assert int(close_result.get("selected_signals_count", 0)) == 0
        skipped = close_result.get("skipped_signals") or []
        assert any(item.get("reason") == "no_trade_gate_triggered" for item in skipped)
        orders = close_result.get("orders") or []
        assert any(item.get("reason") == "EOD_SQUARE_OFF" for item in orders)
        assert len(close_result.get("positions") or []) == 0


def test_no_trade_cooldown_is_deterministic() -> None:
    init_db()
    settings = get_settings()
    provider = f"v30-cooldown-{uuid4().hex[:8]}"
    bundle_name = f"v30-cooldown-{uuid4().hex[:6]}"
    bundle_id, store = _seed_bundle(provider=provider, bundle_name=bundle_name)
    with Session(engine) as session:
        first = evaluate_no_trade_gate(
            session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            asof_ts=datetime.fromisoformat("2026-02-16T10:00:00+00:00"),
            regime="TREND_UP",
            overrides={
                "no_trade_enabled": True,
                "no_trade_regimes": ["TREND_UP"],
                "no_trade_cooldown_trading_days": 2,
                "no_trade_max_realized_vol_annual": 1.0,
                "no_trade_min_breadth_pct": 0.0,
                "no_trade_min_trend_strength": 0.0,
            },
        )
        assert bool(first.get("triggered", False)) is True
        assert int(first.get("cooldown_remaining", 0)) >= 2

        second = evaluate_no_trade_gate(
            session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            asof_ts=datetime.fromisoformat("2026-02-17T10:00:00+00:00"),
            regime="RANGE",
            overrides={
                "no_trade_enabled": True,
                "no_trade_regimes": ["HIGH_VOL"],
                "no_trade_cooldown_trading_days": 2,
                "no_trade_max_realized_vol_annual": 1.0,
                "no_trade_min_breadth_pct": 0.0,
                "no_trade_min_trend_strength": 0.0,
            },
        )
        assert bool(second.get("triggered", False)) is True
        assert "cooldown_active" in list(second.get("reasons", []))

        third = evaluate_no_trade_gate(
            session,
            settings=settings,
            store=store,
            bundle_id=bundle_id,
            timeframe="1d",
            asof_ts=datetime.fromisoformat("2026-02-19T10:00:00+00:00"),
            regime="RANGE",
            overrides={
                "no_trade_enabled": True,
                "no_trade_regimes": ["HIGH_VOL"],
                "no_trade_cooldown_trading_days": 2,
                "no_trade_max_realized_vol_annual": 1.0,
                "no_trade_min_breadth_pct": 0.0,
                "no_trade_min_trend_strength": 0.0,
            },
        )
        assert bool(third.get("triggered", False)) is False
        assert int(third.get("cooldown_remaining", 0)) == 0
