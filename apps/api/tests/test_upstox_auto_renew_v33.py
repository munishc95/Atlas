from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.db.models import OperateEvent, ProviderCredential, UpstoxTokenRequestRun
from app.db.session import engine, init_db
from app.main import app
from app.services.operate_scheduler import run_auto_operate_once
from app.services.paper import get_or_create_paper_state
from app.services.upstox_auth import (
    build_fake_access_token,
    decrypt_token,
    save_provider_credential,
)
from app.services import upstox_token_request


class _FakeQueue:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def enqueue(self, task_path: str, *args: object, **kwargs: object) -> object:
        self.calls.append((task_path, args, kwargs))
        return {"task_path": task_path}


def _reset_upstox_request_state(session: Session) -> None:
    for row in session.exec(select(UpstoxTokenRequestRun)).all():
        session.delete(row)
    session.commit()


def _settings(tmp_path: Path, **kwargs: object) -> Settings:
    base = {
        "redis_url": "redis://127.0.0.1:1/0",
        "cred_key_path": str(tmp_path / "atlas_cred.key"),
        "upstox_client_id": "client-123",
        "upstox_client_secret": "secret-123",
        "upstox_notifier_base_url": "http://127.0.0.1:8000",
    }
    base.update(kwargs)
    return Settings(**base)


def test_token_request_dedupe_returns_existing_pending(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    init_db()
    settings = _settings(tmp_path)

    def _fake_request(
        *, settings: Settings, client_id: str, client_secret: str
    ) -> dict[str, object]:
        assert client_id == "client-123"
        assert client_secret == "secret-123"
        return {
            "status": "success",
            "data": {
                "authorization_expiry": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                "notifier_url": "http://127.0.0.1:8000/api/providers/upstox/notifier",
            },
        }

    monkeypatch.setattr(upstox_token_request, "_request_upstox_token", _fake_request)

    with Session(engine) as session:
        _reset_upstox_request_state(session)
        first, deduped_first = upstox_token_request.request_token_run(
            session,
            settings=settings,
            source="test",
        )
        assert deduped_first is False
        assert first.status == upstox_token_request.STATUS_REQUESTED

        second, deduped_second = upstox_token_request.request_token_run(
            session,
            settings=settings,
            source="test",
        )
        assert deduped_second is True
        assert first.id == second.id


def test_notifier_accepts_matching_nonce_and_stores_encrypted_credential(
    tmp_path: Path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
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
        _reset_upstox_request_state(session)
        run, _ = upstox_token_request.request_token_run(
            session,
            settings=settings,
            source="test",
        )
        token_value = f"token-{uuid4().hex}"
        result = upstox_token_request.process_notifier_payload(
            session,
            settings=settings,
            payload={
                "client_id": "client-123",
                "user_id": "USR1",
                "access_token": token_value,
                "issued_at": datetime.now(UTC).isoformat(),
                "expires_at": (datetime.now(UTC) + timedelta(hours=10)).isoformat(),
            },
            nonce=run.correlation_nonce,
            source="test",
        )
        assert result["accepted"] is True

        credential = session.exec(select(ProviderCredential)).first()
        assert credential is not None
        assert token_value not in credential.access_token_encrypted
        decrypted = decrypt_token(settings=settings, ciphertext=credential.access_token_encrypted)
        assert decrypted == token_value

        refreshed_run = session.get(type(run), run.id)
        assert refreshed_run is not None
        assert refreshed_run.status == upstox_token_request.STATUS_APPROVED


def test_notifier_unmatched_returns_200_and_warn_event(tmp_path: Path) -> None:
    import os

    init_db()
    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.post(
            "/api/providers/upstox/notifier?nonce=bad",
            json={
                "client_id": "missing",
                "access_token": "missing",
            },
        )
        assert response.status_code == 200
        body = response.json()["data"]
        assert body["acknowledged"] is True
        assert body["accepted"] is False

    with Session(engine) as session:
        events = session.exec(
            select(OperateEvent)
            .where(OperateEvent.message == "upstox_notifier_unmatched")
            .order_by(OperateEvent.ts.desc())
        ).all()
        assert len(events) >= 1


def test_scheduler_enqueues_auto_renew_when_token_expiring_soon(tmp_path: Path) -> None:
    init_db()
    settings = _settings(
        tmp_path,
        upstox_auto_renew_enabled=True,
        upstox_auto_renew_time_ist="06:30",
        upstox_auto_renew_if_expires_within_hours=12,
        data_updates_provider_enabled=True,
    )

    with Session(engine) as session:
        _reset_upstox_request_state(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "upstox_auto_renew_enabled": True,
            "upstox_auto_renew_time_ist": "06:30",
            "upstox_auto_renew_if_expires_within_hours": 12,
            "upstox_auto_renew_only_when_provider_enabled": True,
            "operate_last_upstox_auto_renew_date": None,
            "data_updates_provider_enabled": True,
            "operate_auto_run_enabled": False,
        }
        session.add(state)
        session.commit()

        scheduler_now_ist = datetime(2026, 2, 19, 7, 0, tzinfo=upstox_token_request.IST_ZONE)
        expiring_soon = scheduler_now_ist.astimezone(UTC) + timedelta(hours=2)
        save_provider_credential(
            session,
            settings=settings,
            access_token=build_fake_access_token(),
            expires_at=expiring_soon,
        )

        queue = _FakeQueue()
        triggered = run_auto_operate_once(
            session=session,
            queue=queue,  # type: ignore[arg-type]
            settings=settings,
            now_ist=scheduler_now_ist,
        )
        assert triggered is True
        assert any(call[0] == "app.jobs.tasks.run_upstox_token_request_job" for call in queue.calls)


def test_token_status_includes_auto_renew_metadata(tmp_path: Path) -> None:
    init_db()
    settings = _settings(
        tmp_path,
        upstox_auto_renew_enabled=True,
        upstox_auto_renew_time_ist="06:30",
        upstox_auto_renew_if_expires_within_hours=8,
    )

    with Session(engine) as session:
        _reset_upstox_request_state(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **(state.settings_json or {}),
            "upstox_auto_renew_enabled": True,
            "upstox_auto_renew_time_ist": "06:30",
            "upstox_auto_renew_if_expires_within_hours": 8,
            "operate_last_upstox_auto_renew_date": "2026-02-12",
        }
        session.add(state)
        session.commit()

    import os

    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    get_settings.cache_clear()

    with TestClient(app) as client:
        response = client.get("/api/providers/upstox/token/status")
        assert response.status_code == 200
        payload = response.json()["data"]
        auto = payload.get("auto_renew") or {}
        assert bool(auto.get("enabled")) is True
        assert str(auto.get("time_ist")) == "06:30"
        assert int(auto.get("if_expires_within_hours")) == 8
        assert "next_scheduled_run_ist" in auto


def test_token_request_history_endpoints(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import os

    init_db()
    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    os.environ["ATLAS_UPSTOX_CLIENT_ID"] = "client-hist"
    os.environ["ATLAS_UPSTOX_CLIENT_SECRET"] = "secret-hist"
    get_settings.cache_clear()

    def _fake_request(
        *, settings: Settings, client_id: str, client_secret: str
    ) -> dict[str, object]:
        return {
            "status": "success",
            "data": {
                "authorization_expiry": (datetime.now(UTC) + timedelta(hours=4)).isoformat(),
            },
        }

    monkeypatch.setattr(upstox_token_request, "_request_upstox_token", _fake_request)

    with TestClient(app) as client:
        create_res = client.post("/api/providers/upstox/token/request", json={"source": "test"})
        assert create_res.status_code == 200

        latest_res = client.get("/api/providers/upstox/token/requests/latest")
        assert latest_res.status_code == 200
        latest_data = latest_res.json()["data"]
        assert str(latest_data.get("status")) in {"REQUESTED", "APPROVED"}

        history_res = client.get("/api/providers/upstox/token/requests/history?page=1&page_size=10")
        assert history_res.status_code == 200
        history_data = history_res.json()["data"]
        assert isinstance(history_data, list)
        assert len(history_data) >= 1
