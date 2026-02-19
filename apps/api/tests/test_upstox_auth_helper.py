from __future__ import annotations

import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib import error
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.core.exceptions import APIError
from app.db.models import OAuthState
from app.db.session import engine, init_db
from app.main import app
from app.services import upstox_auth


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        redis_url="redis://127.0.0.1:1/0",
        cred_key_path=str(tmp_path / "atlas_cred.key"),
    )


def test_build_authorization_url_has_expected_query() -> None:
    url = upstox_auth.build_authorization_url(
        client_id="client-id",
        redirect_uri="http://localhost:3000/providers/upstox/callback",
        state="abc123",
        base_url="https://api.upstox.com",
    )
    assert url.startswith("https://api.upstox.com/v2/login/authorization/dialog?")
    assert "response_type=code" in url
    assert "client_id=client-id" in url
    assert "state=abc123" in url


def test_store_and_validate_oauth_state_with_db_ttl(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    with Session(engine) as session:
        state_token = f"state-{uuid4().hex}"
        expires = upstox_auth.store_oauth_state(
            session,
            settings=settings,
            state=state_token,
            redirect_uri="http://localhost:3000/providers/upstox/callback",
            ttl_seconds=120,
        )
        assert expires > datetime.now(UTC)
        upstox_auth.validate_oauth_state(
            session,
            settings=settings,
            state=state_token,
            redirect_uri="http://localhost:3000/providers/upstox/callback",
        )
        with pytest.raises(APIError) as exc:
            upstox_auth.validate_oauth_state(
                session,
                settings=settings,
                state=state_token,
                redirect_uri="http://localhost:3000/providers/upstox/callback",
            )
        assert exc.value.code == "oauth_state_consumed"


def test_validate_oauth_state_expired(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    with Session(engine) as session:
        state_token = f"state-expired-{uuid4().hex}"
        upstox_auth.store_oauth_state(
            session,
            settings=settings,
            state=state_token,
            redirect_uri="http://localhost:3000/providers/upstox/callback",
            ttl_seconds=120,
        )
        row = session.exec(
            select(OAuthState).where(OAuthState.state == state_token)
        ).first()
        assert row is not None
        row.expires_at = datetime.now(UTC) - timedelta(minutes=1)
        session.add(row)
        session.commit()
        with pytest.raises(APIError) as exc:
            upstox_auth.validate_oauth_state(
                session,
                settings=settings,
                state=state_token,
                redirect_uri="http://localhost:3000/providers/upstox/callback",
            )
        assert exc.value.code == "oauth_state_expired"


def test_save_provider_credential_stores_encrypted(tmp_path: Path) -> None:
    init_db()
    settings = _settings(tmp_path)
    raw_token = "raw-token-value-123"
    with Session(engine) as session:
        row = upstox_auth.save_provider_credential(
            session,
            settings=settings,
            access_token=raw_token,
            issued_at=datetime.now(UTC),
            expires_at=datetime.now(UTC) + timedelta(hours=12),
        )
        assert row.access_token_encrypted
        assert raw_token not in row.access_token_encrypted
        token = upstox_auth.get_provider_access_token(
            session,
            settings=settings,
            allow_env_fallback=False,
        )
        assert token == raw_token


def test_token_status_and_disconnect_endpoints(tmp_path: Path) -> None:
    import os

    init_db()
    os.environ["ATLAS_REDIS_URL"] = "redis://127.0.0.1:1/0"
    os.environ["ATLAS_CRED_KEY_PATH"] = str(tmp_path / "atlas_cred.key")
    os.environ["ATLAS_UPSTOX_ACCESS_TOKEN"] = ""
    get_settings.cache_clear()

    with Session(engine) as session:
        upstox_auth.save_provider_credential(
            session,
            settings=get_settings(),
            access_token=upstox_auth.build_fake_access_token(),
        )
    with TestClient(app) as client:
        status_res = client.get("/api/providers/upstox/token/status")
        assert status_res.status_code == 200
        status_data = status_res.json()["data"]
        assert bool(status_data["connected"]) is True
        assert status_data["token_source"] == "encrypted_store"
        assert status_data["is_expired"] is False

        disconnect_res = client.post("/api/providers/upstox/disconnect", json={})
        assert disconnect_res.status_code == 200
        assert bool(disconnect_res.json()["data"]["disconnected"]) is True

        status_after = client.get("/api/providers/upstox/token/status").json()["data"]
        assert bool(status_after["connected"]) is False


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None


def test_exchange_authorization_code_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(req, timeout):  # noqa: ANN001
        assert req.full_url.endswith("/v2/login/authorization/token")
        assert timeout == 5.0
        return _FakeResponse({"status": "success", "data": {"access_token": "abc"}})

    monkeypatch.setattr(upstox_auth.request, "urlopen", _fake_urlopen)
    out = upstox_auth.exchange_authorization_code(
        code="code-1",
        client_id="cid",
        client_secret="csecret",
        redirect_uri="http://localhost/callback",
        base_url="https://api.upstox.com",
        timeout_seconds=5.0,
    )
    assert out["status"] == "success"


def test_exchange_authorization_code_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_urlopen(req, timeout):  # noqa: ANN001
        payload = io.BytesIO(
            json.dumps({"status": "error", "errors": [{"message": "bad code"}]}).encode("utf-8")
        )
        raise error.HTTPError(req.full_url, 400, "bad request", {}, payload)

    monkeypatch.setattr(upstox_auth.request, "urlopen", _fake_urlopen)
    with pytest.raises(APIError) as exc:
        upstox_auth.exchange_authorization_code(
            code="bad",
            client_id="cid",
            client_secret="csecret",
            redirect_uri="http://localhost/callback",
            base_url="https://api.upstox.com",
            timeout_seconds=5.0,
        )
    assert exc.value.code == "upstox_token_exchange_failed"
