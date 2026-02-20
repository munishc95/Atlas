from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import OperateEvent, UpstoxNotifierEvent, UpstoxTokenRequestRun
from app.db.session import engine, init_db
from app.main import app
from app.services import upstox_token_request


def _settings(tmp_path: Path, **kwargs: object) -> Settings:
    base = {
        "redis_url": "redis://127.0.0.1:1/0",
        "cred_key_path": str(tmp_path / "atlas_cred.key"),
        "upstox_client_id": "client-v34",
        "upstox_client_secret": "secret-v34",
        "upstox_notifier_base_url": "http://127.0.0.1:8000",
    }
    base.update(kwargs)
    return Settings(**base)


def _reset_state(session: Session) -> None:
    for row in session.exec(select(UpstoxNotifierEvent)).all():
        session.delete(row)
    for row in session.exec(select(UpstoxTokenRequestRun)).all():
        session.delete(row)
    for row in session.exec(select(OperateEvent)).all():
        session.delete(row)
    session.commit()


def test_secret_route_mismatch_returns_200_and_records_warning(tmp_path: Path) -> None:
    init_db()
    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    os.environ["ATLAS_UPSTOX_CLIENT_ID"] = "client-v34"
    os.environ["ATLAS_UPSTOX_CLIENT_SECRET"] = "secret-v34"
    get_settings.cache_clear()

    with Session(engine) as session:
        _reset_state(session)

    with TestClient(app) as client:
        status_res = client.get("/api/providers/upstox/notifier/status")
        assert status_res.status_code == 200
        notifier_url = str(status_res.json()["data"]["recommended_notifier_url"])
        secret = notifier_url.rsplit("/", 1)[-1]
        bad_secret = f"{secret}-bad"
        res = client.post(
            f"/api/providers/upstox/notifier/{bad_secret}",
            json={
                "client_id": "client-v34",
                "user_id": "USER123",
                "access_token": "token-bad",
                "message_type": "access_token",
            },
        )
        assert res.status_code == 200
        data = res.json()["data"]
        assert data["acknowledged"] is True
        assert data["accepted"] is False
        assert data["reason"] == "secret_mismatch"

    with Session(engine) as session:
        events = session.exec(select(UpstoxNotifierEvent)).all()
        assert len(events) >= 1
        warnings = session.exec(
            select(OperateEvent)
            .where(OperateEvent.message == "upstox_notifier_invalid_secret")
            .order_by(OperateEvent.ts.desc())
        ).all()
        assert len(warnings) >= 1


def test_digest_dedup_prevents_duplicate_events(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    init_db()
    settings = _settings(tmp_path)

    def _fake_request(
        *, settings: Settings, client_id: str, client_secret: str
    ) -> dict[str, object]:
        return {
            "status": "success",
            "data": {
                "authorization_expiry": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            },
        }

    monkeypatch.setattr(upstox_token_request, "_request_upstox_token", _fake_request)

    with Session(engine) as session:
        _reset_state(session)
        run, _ = upstox_token_request.request_token_run(session, settings=settings, source="test")
        payload = {
            "client_id": "client-v34",
            "user_id": "U123",
            "access_token": "token-abc",
            "issued_at": datetime.now(UTC).isoformat(),
            "expires_at": (datetime.now(UTC) + timedelta(hours=8)).isoformat(),
            "message_type": "access_token",
        }
        first = upstox_token_request.process_notifier_payload(
            session,
            settings=settings,
            payload=payload,
            nonce=run.correlation_nonce,
            source="test",
        )
        second = upstox_token_request.process_notifier_payload(
            session,
            settings=settings,
            payload=payload,
            nonce=run.correlation_nonce,
            source="test",
        )
        assert first["accepted"] is True
        assert second["deduplicated"] is True
        rows = session.exec(select(UpstoxNotifierEvent)).all()
        assert len(rows) == 1


def test_pending_run_transitions_to_approved(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    init_db()
    settings = _settings(tmp_path)

    def _fake_request(
        *, settings: Settings, client_id: str, client_secret: str
    ) -> dict[str, object]:
        return {
            "status": "success",
            "data": {
                "authorization_expiry": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
            },
        }

    monkeypatch.setattr(upstox_token_request, "_request_upstox_token", _fake_request)

    with Session(engine) as session:
        _reset_state(session)
        run, _ = upstox_token_request.request_token_run(session, settings=settings, source="test")
        out = upstox_token_request.process_notifier_payload(
            session,
            settings=settings,
            payload={
                "client_id": "client-v34",
                "user_id": "USER1",
                "access_token": "token-approved",
                "issued_at": datetime.now(UTC).isoformat(),
                "expires_at": (datetime.now(UTC) + timedelta(hours=9)).isoformat(),
                "message_type": "access_token",
            },
            nonce=run.correlation_nonce,
            source="test",
        )
        assert out["accepted"] is True
        refreshed = session.get(UpstoxTokenRequestRun, run.id)
        assert refreshed is not None
        assert refreshed.status == upstox_token_request.STATUS_APPROVED
        assert refreshed.resolved_at is not None
        assert refreshed.resolution_reason == upstox_token_request.REASON_NOTIFIER_RECEIVED


def test_sweeper_marks_pending_as_expired(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)

    with Session(engine) as session:
        _reset_state(session)
        now = datetime.now(UTC)
        row = UpstoxTokenRequestRun(
            provider_kind="UPSTOX",
            status="PENDING",
            requested_at=now - timedelta(hours=5),
            authorization_expiry=now - timedelta(minutes=1),
            client_id="client-v34",
            correlation_nonce="nonce-expired",
        )
        session.add(row)
        session.commit()
        session.refresh(row)

        changed = upstox_token_request.sweep_expired_request_runs(session, settings=settings)
        assert changed == 1
        refreshed = session.get(UpstoxTokenRequestRun, row.id)
        assert refreshed is not None
        assert refreshed.status == upstox_token_request.STATUS_EXPIRED
        assert refreshed.resolution_reason == upstox_token_request.REASON_EXPIRED_NO_NOTIFIER


def test_notifier_test_endpoint_creates_event_and_health(tmp_path: Path) -> None:
    init_db()
    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    os.environ["ATLAS_UPSTOX_CLIENT_ID"] = "client-v34"
    os.environ["ATLAS_UPSTOX_CLIENT_SECRET"] = "secret-v34"
    get_settings.cache_clear()

    with Session(engine) as session:
        _reset_state(session)

    with TestClient(app) as client:
        res = client.post("/api/providers/upstox/notifier/test", json={})
        assert res.status_code == 200
        data = res.json()["data"]
        assert data.get("created_event_id") is not None
        health = data.get("webhook_health_after") or {}
        assert str(health.get("status")) in {"OK", "FAILING", "STALE", "NEVER_RECEIVED"}
