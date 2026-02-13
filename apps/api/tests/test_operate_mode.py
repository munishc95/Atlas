from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from uuid import uuid4

from sqlmodel import Session

from app.core.config import get_settings
from app.db.models import DatasetBundle, PaperRun, Policy, PolicyHealthSnapshot
from app.db.session import engine, init_db
from app.services.policy_health import (
    DEGRADED,
    PAUSED,
    WARNING,
    apply_policy_health_actions,
    get_policy_health_snapshot,
    select_fallback_policy,
)
from app.services.reports import generate_daily_report


def _dt(day_offset: int) -> datetime:
    return datetime.now(timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0) + timedelta(
        days=day_offset
    )


def _policy(name: str, *, status: str = "ACTIVE", score: float = 0.0, regime_key: str = "TREND_UP") -> Policy:
    return Policy(
        name=name,
        definition_json={
            "status": status,
            "baseline": {
                "max_drawdown": 0.1,
                "win_rate": 0.55,
                "period_return": 0.2,
                "oos_score": score,
            },
            "regime_map": {
                regime_key: {
                    "strategy_key": "trend_breakout",
                    "risk_scale": 1.0,
                    "max_positions_scale": 1.0,
                }
            },
        },
    )


def test_policy_health_snapshot_is_deterministic_for_fixed_runs() -> None:
    init_db()
    settings = get_settings()
    unique = uuid4().hex[:8]
    report_date = datetime.now(timezone.utc).date()

    with Session(engine) as session:
        policy = _policy(name=f"health-deterministic-{unique}", score=1.0)
        session.add(policy)
        session.commit()
        session.refresh(policy)
        assert policy.id is not None

        for idx in range(25):
            day = report_date - timedelta(days=24 - idx)
            equity_before = 1_000_000.0 + (idx * 500.0)
            net_pnl = 1_000.0 if idx % 3 else -250.0
            row = PaperRun(
                bundle_id=None,
                policy_id=policy.id,
                asof_ts=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                regime="TREND_UP",
                signals_source="generated",
                generated_signals_count=4,
                selected_signals_count=1,
                skipped_signals_count=3,
                scanned_symbols=40,
                evaluated_candidates=12,
                scan_truncated=False,
                summary_json={
                    "equity_before": equity_before,
                    "equity_after": equity_before + net_pnl,
                    "net_pnl": net_pnl,
                    "gross_pnl": net_pnl + 40.0,
                    "total_cost": 40.0,
                    "trade_count": 1,
                    "avg_holding_days": 2.0,
                    "turnover": 0.12,
                    "exposure": 0.35,
                },
                cost_summary_json={"total_cost": 40.0},
            )
            session.add(row)
        session.commit()

        first = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=policy,
            window_days=20,
            asof_date=report_date,
            refresh=True,
        )
        second = get_policy_health_snapshot(
            session,
            settings=settings,
            policy=policy,
            window_days=20,
            asof_date=report_date,
            refresh=True,
        )

        assert first.status == second.status
        assert first.metrics_json == second.metrics_json
        assert first.reasons_json == second.reasons_json


def test_drift_actions_apply_warning_scale_and_pause_policy() -> None:
    init_db()
    settings = get_settings()
    unique = uuid4().hex[:8]

    with Session(engine) as session:
        policy = _policy(name=f"health-actions-{unique}", score=0.4)
        session.add(policy)
        session.commit()
        session.refresh(policy)
        assert policy.id is not None

        warning_snapshot = PolicyHealthSnapshot(
            policy_id=policy.id,
            asof_date=date.today(),
            window_days=20,
            metrics_json={"max_drawdown": -0.14},
            status=WARNING,
            reasons_json=["warning condition"],
        )
        session.add(warning_snapshot)
        session.commit()
        session.refresh(warning_snapshot)

        warning_action = apply_policy_health_actions(
            session,
            settings=settings,
            policy=policy,
            snapshot=warning_snapshot,
            overrides={"drift_warning_risk_scale": 0.7},
        )
        assert warning_action["action"] == "RISK_SCALE_WARNING"
        assert float(warning_action["risk_scale_override"]) == 0.7

        degraded_snapshot = PolicyHealthSnapshot(
            policy_id=policy.id,
            asof_date=date.today(),
            window_days=60,
            metrics_json={"max_drawdown": -0.3},
            status=DEGRADED,
            reasons_json=["degraded condition"],
        )
        session.add(degraded_snapshot)
        session.commit()
        session.refresh(degraded_snapshot)

        degraded_action = apply_policy_health_actions(
            session,
            settings=settings,
            policy=policy,
            snapshot=degraded_snapshot,
            overrides={"drift_degraded_action": "PAUSE", "drift_degraded_risk_scale": 0.2},
        )
        session.refresh(policy)
        definition = policy.definition_json if isinstance(policy.definition_json, dict) else {}

        assert str(degraded_action["policy_status"]).upper() == PAUSED
        assert str(definition.get("status", "")).upper() == PAUSED
        assert float(degraded_action["risk_scale_override"]) == 0.2


def test_daily_report_generation_is_bundle_policy_scoped() -> None:
    init_db()
    settings = get_settings()
    unique = uuid4().hex[:8]
    report_date = datetime.now(timezone.utc).date()

    with Session(engine) as session:
        bundle_a = DatasetBundle(
            name=f"bundle-a-{unique}",
            provider="test",
            symbols_json=["AAA"],
            supported_timeframes_json=["1d"],
        )
        bundle_b = DatasetBundle(
            name=f"bundle-b-{unique}",
            provider="test",
            symbols_json=["BBB"],
            supported_timeframes_json=["1d"],
        )
        policy_a = _policy(name=f"report-policy-a-{unique}", score=1.0)
        policy_b = _policy(name=f"report-policy-b-{unique}", score=0.8)
        session.add(bundle_a)
        session.add(bundle_b)
        session.add(policy_a)
        session.add(policy_b)
        session.commit()
        session.refresh(bundle_a)
        session.refresh(bundle_b)
        session.refresh(policy_a)
        session.refresh(policy_b)
        assert bundle_a.id is not None
        assert bundle_b.id is not None
        assert policy_a.id is not None
        assert policy_b.id is not None

        rows = [
            (bundle_a.id, policy_a.id, 1200.0),
            (bundle_a.id, policy_a.id, -300.0),
            (bundle_b.id, policy_b.id, 900.0),
        ]
        for idx, (bundle_id, policy_id, net_pnl) in enumerate(rows):
            ts = datetime.combine(report_date, datetime.min.time(), tzinfo=timezone.utc) + timedelta(
                minutes=idx * 10
            )
            session.add(
                PaperRun(
                    bundle_id=bundle_id,
                    policy_id=policy_id,
                    asof_ts=ts,
                    regime="TREND_UP",
                    signals_source="generated",
                    generated_signals_count=2,
                    selected_signals_count=1,
                    skipped_signals_count=1,
                    scanned_symbols=20,
                    evaluated_candidates=8,
                    scan_truncated=False,
                    summary_json={
                        "equity_before": 1_000_000.0,
                        "equity_after": 1_000_000.0 + net_pnl,
                        "net_pnl": net_pnl,
                        "gross_pnl": net_pnl + 30.0,
                        "total_cost": 30.0,
                        "positions_before": 0,
                        "positions_after": 1,
                        "positions_opened": 1,
                        "positions_closed": 0,
                        "selected_reason_histogram": {"provided": 1},
                        "skipped_reason_histogram": {"max_positions_reached": 1},
                    },
                    cost_summary_json={"total_cost": 30.0},
                )
            )
        session.commit()

        report = generate_daily_report(
            session=session,
            settings=settings,
            report_date=report_date,
            bundle_id=bundle_a.id,
            policy_id=policy_a.id,
            overwrite=True,
        )
        content = report.content_json

        assert report.bundle_id == bundle_a.id
        assert report.policy_id == policy_a.id
        assert int(content["summary"]["runs"]) == 2
        assert float(content["summary"]["net_pnl"]) == 900.0
        assert int(content["explainability"]["selected_reason_histogram"]["provided"]) == 2


def test_regime_fallback_prefers_regime_match_then_risk_off() -> None:
    init_db()
    unique = uuid4().hex[:8]
    regime_key = f"TREND_{unique.upper()}"

    with Session(engine) as session:
        current = _policy(name=f"fallback-current-{unique}", status=PAUSED, score=1.0)
        candidate_low = _policy(name=f"fallback-low-{unique}", score=700.0, regime_key=regime_key)
        candidate_high = _policy(name=f"fallback-high-{unique}", score=900.0, regime_key=regime_key)
        candidate_risk_off = _policy(
            name=f"fallback-riskoff-{unique}",
            score=9_999.0,
            regime_key="RISK_OFF",
        )

        session.add(current)
        session.add(candidate_low)
        session.add(candidate_high)
        session.add(candidate_risk_off)
        session.commit()
        session.refresh(current)
        session.refresh(candidate_high)
        session.refresh(candidate_risk_off)
        assert current.id is not None
        assert candidate_high.id is not None
        assert candidate_risk_off.id is not None

        selected = select_fallback_policy(
            session,
            current_policy_id=current.id,
            regime=regime_key,
        )
        assert selected is not None
        assert selected.id == candidate_high.id

        # Mark regime-specific policies as paused to force RISK_OFF fallback.
        candidate_high.definition_json = {
            **(candidate_high.definition_json or {}),
            "status": PAUSED,
        }
        candidate_low.definition_json = {
            **(candidate_low.definition_json or {}),
            "status": PAUSED,
        }
        session.add(candidate_high)
        session.add(candidate_low)
        session.commit()

        risk_off_selected = select_fallback_policy(
            session,
            current_policy_id=current.id,
            regime=regime_key,
        )
        assert risk_off_selected is not None
        assert risk_off_selected.id == candidate_risk_off.id
