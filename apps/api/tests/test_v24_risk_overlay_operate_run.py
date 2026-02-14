from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
from sqlmodel import Session, select

from app.core.config import get_settings
from app.db.models import PaperRun, PaperState
from app.db.session import engine, init_db
from app.engine.simulator import SimulationConfig, simulate_portfolio_step
from app.main import app
from app.services.portfolio_risk import create_portfolio_risk_snapshot


DATA_FILE = Path("data/sample/NIFTY500_1d.csv")


def _client_inline_jobs() -> TestClient:
    import os

    os.environ["ATLAS_JOBS_INLINE"] = "true"
    get_settings.cache_clear()
    return TestClient(app)


def _wait_job(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    started = time.time()
    while time.time() - started < timeout:
        payload = client.get(f"/api/jobs/{job_id}").json()["data"]
        if payload["status"] in {"SUCCEEDED", "FAILED", "DONE"}:
            return payload
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} timed out")


def _import_sample(client: TestClient) -> None:
    with DATA_FILE.open("rb") as fh:
        response = client.post(
            "/api/data/import",
            data={"symbol": "NIFTY500", "timeframe": "1d", "provider": "csv"},
            files={"file": (DATA_FILE.name, fh, "text/csv")},
        )
    assert response.status_code == 200
    job = _wait_job(client, response.json()["data"]["job_id"], timeout=120.0)
    assert job["status"] == "SUCCEEDED"


def test_risk_overlay_scale_is_deterministic_for_fixed_returns() -> None:
    init_db()
    settings = get_settings()
    asof = datetime(2026, 2, 14, 9, 30, tzinfo=timezone.utc)

    with Session(engine) as session:
        for row in session.exec(select(PaperRun)).all():
            session.delete(row)
        session.commit()

        equity = 1_000_000.0
        returns = [0.01, -0.006, 0.004, -0.002, 0.007, -0.003, 0.005]
        for idx, ret in enumerate(returns):
            session.add(
                PaperRun(
                    bundle_id=1,
                    policy_id=1,
                    asof_ts=asof - timedelta(days=len(returns) - idx),
                    mode="LIVE",
                    regime="TREND_UP",
                    summary_json={
                        "equity_before": equity,
                        "net_pnl": equity * ret,
                    },
                    cost_summary_json={},
                )
            )
        session.commit()

        overrides = {
            "operate_mode": "live",
            "risk_overlay_enabled": True,
            "risk_overlay_target_vol_annual": 0.18,
            "risk_overlay_lookback_days": 20,
            "risk_overlay_min_scale": 0.25,
            "risk_overlay_max_scale": 1.25,
        }
        snapshot = create_portfolio_risk_snapshot(
            session=session,
            settings=settings,
            bundle_id=1,
            policy_id=1,
            overrides=overrides,
            asof=asof,
        )
        expected_vol = float(np.std(np.array(returns, dtype=float), ddof=0) * np.sqrt(252.0))
        expected_scale = float(
            max(0.25, min(1.25, 0.18 / max(expected_vol, 1e-9)))
        )

        assert abs(float(snapshot["realized_vol"]) - expected_vol) < 1e-12
        assert abs(float(snapshot["risk_scale"]) - expected_scale) < 1e-12


def test_simulator_risk_overlay_caps_produce_explicit_skip_reasons() -> None:
    asof = datetime(2026, 2, 14, 10, 0, tzinfo=timezone.utc)
    base_signal = {
        "symbol": "ABC",
        "underlying_symbol": "ABC",
        "sector": "TECH",
        "side": "BUY",
        "template": "trend_breakout",
        "instrument_kind": "EQUITY_CASH",
        "price": 100.0,
        "stop_distance": 2.0,
        "signal_strength": 0.9,
        "adv": 1_000_000_000.0,
        "vol_scale": 0.01,
    }

    gross_cfg = SimulationConfig(
        initial_equity=100_000.0,
        risk_per_trade=0.01,
        max_positions=5,
        risk_overlay_enabled=True,
        risk_overlay_scale=1.0,
        risk_overlay_max_gross_exposure_pct=0.05,
        risk_overlay_max_single_name_exposure_pct=1.0,
        risk_overlay_max_sector_exposure_pct=1.0,
        seed=11,
    )
    gross_result = simulate_portfolio_step(
        signals=[base_signal],
        open_positions=[],
        mark_prices={},
        asof=pd.Timestamp(asof),
        cash=100_000.0,
        equity_reference=100_000.0,
        config=gross_cfg,
    )
    assert any(
        str(item.get("reason")) == "risk_overlay_gross_exposure_cap"
        for item in gross_result.skipped_signals
    )

    single_cfg = SimulationConfig(
        initial_equity=100_000.0,
        risk_per_trade=0.01,
        max_positions=5,
        risk_overlay_enabled=True,
        risk_overlay_scale=1.0,
        risk_overlay_max_gross_exposure_pct=2.0,
        risk_overlay_max_single_name_exposure_pct=0.03,
        risk_overlay_max_sector_exposure_pct=1.0,
        seed=11,
    )
    single_result = simulate_portfolio_step(
        signals=[base_signal],
        open_positions=[],
        mark_prices={},
        asof=pd.Timestamp(asof),
        cash=100_000.0,
        equity_reference=100_000.0,
        config=single_cfg,
    )
    assert any(
        str(item.get("reason")) == "risk_overlay_single_name_cap"
        for item in single_result.skipped_signals
    )

    corr_cfg = SimulationConfig(
        initial_equity=100_000.0,
        risk_per_trade=0.01,
        max_positions=5,
        risk_overlay_enabled=True,
        risk_overlay_scale=1.0,
        risk_overlay_max_gross_exposure_pct=2.0,
        risk_overlay_max_single_name_exposure_pct=1.0,
        risk_overlay_max_sector_exposure_pct=1.0,
        risk_overlay_corr_clamp_enabled=True,
        risk_overlay_corr_threshold=0.65,
        risk_overlay_corr_reduce_factor=0.0,
        seed=11,
    )
    corr_result = simulate_portfolio_step(
        signals=[{**base_signal, "correlations": {"OPEN1": 0.9}}],
        open_positions=[
            {
                "id": 1,
                "symbol": "OPEN1",
                "side": "BUY",
                "instrument_kind": "EQUITY_CASH",
                "qty": 10,
                "avg_price": 100.0,
                "stop_price": 90.0,
                "target_price": None,
                "lot_size": 1,
                "qty_lots": 1,
                "margin_reserved": 0.0,
                "must_exit_by_eod": False,
                "opened_at": asof.isoformat(),
                "metadata_json": {"underlying_symbol": "OPEN1", "sector": "TECH"},
            }
        ],
        mark_prices={"OPEN1": 100.0},
        asof=pd.Timestamp(asof),
        cash=100_000.0,
        equity_reference=100_000.0,
        config=corr_cfg,
    )
    assert any(
        str(item.get("reason")) == "risk_overlay_corr_clamp"
        for item in corr_result.skipped_signals
    )


def test_operate_run_executes_in_order_and_persists_report_artifacts() -> None:
    init_db()
    with _client_inline_jobs() as client:
        _import_sample(client)

        universes = client.get("/api/universes")
        assert universes.status_code == 200
        rows = universes.json()["data"]
        assert rows
        bundle_id = int(rows[0]["id"])

        settings_res = client.put(
            "/api/settings",
            json={
                "paper_mode": "strategy",
                "active_policy_id": None,
                "operate_auto_run_include_data_updates": True,
                "reports_auto_generate_daily": False,
                "risk_overlay_enabled": True,
                "operate_mode": "live",
            },
        )
        assert settings_res.status_code == 200

        run = client.post(
            "/api/operate/run",
            json={"bundle_id": bundle_id, "timeframe": "1d"},
        )
        assert run.status_code == 200
        job_id = run.json()["data"]["job_id"]
        job = _wait_job(client, job_id, timeout=180.0)
        assert job["status"] == "SUCCEEDED"
        result = job["result_json"] or {}
        summary = result.get("summary", {})
        assert summary.get("step_order") == [
            "data_updates",
            "data_quality",
            "paper_step",
            "daily_report",
        ]
        steps = summary.get("steps", [])
        assert [step.get("name") for step in steps] == [
            "data_updates",
            "data_quality",
            "paper_step",
            "daily_report",
        ]
        report_id = int((summary.get("daily_report") or {}).get("id") or 0)
        assert report_id > 0

        report_detail = client.get(f"/api/reports/daily/{report_id}")
        assert report_detail.status_code == 200
        pdf = client.get(f"/api/reports/daily/{report_id}/export.pdf")
        assert pdf.status_code == 200
        assert pdf.headers["content-type"].startswith("application/pdf")
