from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import (
    DataUpdateRun,
    DatasetBundle,
    OperateEvent,
    UpstoxNotifierEvent,
    UpstoxNotifierPingEvent,
    UpstoxTokenRequestRun,
)
from app.db.session import engine, init_db
from app.main import app
from app.services import upstox_token_request
from app.services.jobs import create_job
from app.services.paper import get_or_create_paper_state


def _settings(tmp_path: Path, **kwargs: object) -> Settings:
    base = {
        "redis_url": "redis://127.0.0.1:1/0",
        "cred_key_path": str(tmp_path / "atlas_cred.key"),
        "upstox_client_id": "client-v35",
        "upstox_client_secret": "secret-v35",
        "upstox_notifier_base_url": "http://127.0.0.1:8000",
        "jobs_inline": True,
    }
    base.update(kwargs)
    return Settings(**base)


def _reset_upstox_state(session: Session) -> None:
    for row in session.exec(select(UpstoxNotifierEvent)).all():
        session.delete(row)
    for row in session.exec(select(UpstoxNotifierPingEvent)).all():
        session.delete(row)
    for row in session.exec(select(UpstoxTokenRequestRun)).all():
        session.delete(row)
    for row in session.exec(select(OperateEvent)).all():
        session.delete(row)
    for row in session.exec(select(DataUpdateRun)).all():
        session.delete(row)
    session.commit()


def test_token_renew_reuses_existing_pending_run(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    init_db()
    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    os.environ["ATLAS_UPSTOX_CLIENT_ID"] = "client-v35"
    os.environ["ATLAS_UPSTOX_CLIENT_SECRET"] = "secret-v35"
    get_settings.cache_clear()

    def _fake_request(
        *, settings: Settings, client_id: str, client_secret: str
    ) -> dict[str, object]:
        assert client_id == "client-v35"
        assert client_secret == "secret-v35"
        return {
            "status": "success",
            "data": {
                "authorization_expiry": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            },
        }

    monkeypatch.setattr(upstox_token_request, "_request_upstox_token", _fake_request)

    with Session(engine) as session:
        _reset_upstox_state(session)

    with TestClient(app) as client:
        first = client.post("/api/providers/upstox/token/renew", json={"source": "test_v35"})
        assert first.status_code == 200
        first_data = first.json()["data"]
        first_run = first_data["run"]
        assert first_data["reused"] is False
        assert str(first_data["approval_instructions"]["status"]) == "new_pending"

        second = client.post("/api/providers/upstox/token/renew", json={"source": "test_v35"})
        assert second.status_code == 200
        second_data = second.json()["data"]
        second_run = second_data["run"]
        assert second_data["reused"] is True
        assert str(second_data["approval_instructions"]["status"]) == "reused_pending"
        assert str(second_run["id"]) == str(first_run["id"])


def test_ping_lifecycle_create_receive_status(tmp_path: Path) -> None:
    init_db()
    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    os.environ["ATLAS_UPSTOX_CLIENT_ID"] = "client-v35"
    os.environ["ATLAS_UPSTOX_CLIENT_SECRET"] = "secret-v35"
    get_settings.cache_clear()

    with Session(engine) as session:
        _reset_upstox_state(session)

    with TestClient(app) as client:
        create_res = client.post("/api/providers/upstox/notifier/ping?source=settings", json={})
        assert create_res.status_code == 200
        created = create_res.json()["data"]
        ping_id = str(created["ping_id"])
        assert ping_id
        assert str(created["status"]) == "SENT"

        before = client.get(f"/api/providers/upstox/notifier/ping/{ping_id}/status")
        assert before.status_code == 200
        assert str(before.json()["data"]["status"]) == "SENT"

        receive = client.get(f"/api/providers/upstox/notifier/ping/{ping_id}")
        assert receive.status_code == 200
        receive_data = receive.json()["data"]
        assert receive_data["ok"] is True
        assert str(receive_data["status"]) == "RECEIVED"

        after = client.get(f"/api/providers/upstox/notifier/ping/{ping_id}/status")
        assert after.status_code == 200
        assert str(after.json()["data"]["status"]) == "RECEIVED"


def test_notifier_rate_limit_warn_event_emitted(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    with Session(engine) as session:
        _reset_upstox_state(session)
        for _ in range(35):
            upstox_token_request.check_notifier_rate_limit(
                session,
                ip="127.0.0.1",
                source="webhook_secret",
                correlation_id=None,
            )
        rows = session.exec(
            select(OperateEvent)
            .where(OperateEvent.message == "upstox_notifier_rate_limited")
            .order_by(OperateEvent.ts.desc())
        ).all()
        assert len(rows) >= 1
        details = rows[0].details_json if isinstance(rows[0].details_json, dict) else {}
        assert str(details.get("source")) == "webhook_secret"
        assert int(details.get("limit_per_minute", 0)) == 30
        _ = settings  # keep settings construction for deterministic key path initialization


def test_operate_run_skips_provider_stage_on_token_invalid_and_continues(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    init_db()
    settings = _settings(
        tmp_path,
        data_updates_provider_enabled=True,
        data_updates_provider_kind="UPSTOX",
        operate_provider_stage_on_token_invalid="SKIP",
    )
    with Session(engine) as session:
        _reset_upstox_state(session)
        bundle = DatasetBundle(
            name=f"v35-bundle-{datetime.now(UTC).timestamp()}",
            provider="local",
            description="v35 operate skip fallback",
            symbols_json=["NIFTY500"],
            supported_timeframes_json=["1d"],
        )
        session.add(bundle)
        session.commit()
        session.refresh(bundle)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "operate_auto_run_include_data_updates": True,
            "data_updates_provider_enabled": True,
            "data_updates_provider_kind": "UPSTOX",
            "data_updates_provider_timeframes": ["1d"],
            "operate_provider_stage_on_token_invalid": "SKIP",
        }
        session.add(state)
        session.commit()

        from app.jobs import tasks as job_tasks
        from app.services.data_store import DataStore

        monkeypatch.setattr(
            job_tasks,
            "run_data_updates",
            lambda **kwargs: SimpleNamespace(
                status="SUCCEEDED",
                id=1,
                rows_ingested=7,
                processed_files=1,
                skipped_files=0,
            ),
        )
        monkeypatch.setattr(
            job_tasks,
            "run_data_quality_report",
            lambda **kwargs: SimpleNamespace(status="OK", id=1, coverage_pct=100.0, issues_json=[]),
        )
        monkeypatch.setattr(
            job_tasks,
            "run_paper_step",
            lambda **kwargs: {
                "status": "ok",
                "paper_run_id": 1,
                "execution_mode": "LIVE",
                "selected_signals_count": 0,
                "generated_signals_count": 0,
                "safe_mode": {"active": False},
                "scan_truncated": False,
                "risk_overlay": {},
            },
        )
        monkeypatch.setattr(
            job_tasks,
            "generate_daily_report",
            lambda **kwargs: SimpleNamespace(id=1, date=date.today()),
        )

        job = create_job(session, "operate_run")
        store = DataStore(
            parquet_root=settings.parquet_root,
            duckdb_path=settings.duckdb_path,
            feature_cache_root=settings.feature_cache_root,
        )
        out = job_tasks._operate_run_result(
            session=session,
            settings=settings,
            store=store,
            payload={
                "bundle_id": int(bundle.id or 0),
                "timeframe": "1d",
                "include_data_updates": True,
            },
            job_id=job.id,
        )
        summary = out["summary"]
        assert str(summary["provider_stage_status"]) == "SKIPPED_TOKEN_INVALID"
        assert str(summary["data_updates"]["status"]) == "SUCCEEDED"
        assert str(summary["data_quality"]["status"]) == "OK"
        assert str(summary["daily_report"]["status"]) == "SUCCEEDED"

