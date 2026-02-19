from __future__ import annotations

import io
import json
from pathlib import Path
from urllib import error

import pytest

from app.core.exceptions import APIError
from app.services import upstox_auth


def test_build_authorization_url_has_expected_query() -> None:
    url = upstox_auth.build_authorization_url(
        client_id="client-id",
        redirect_uri="http://localhost:3000/callback",
        state="abc123",
        base_url="https://api.upstox.com",
    )
    assert url.startswith("https://api.upstox.com/v2/login/authorization/dialog?")
    assert "response_type=code" in url
    assert "client_id=client-id" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A3000%2Fcallback" in url
    assert "state=abc123" in url


def test_extract_access_token_from_nested_and_root() -> None:
    assert (
        upstox_auth.extract_access_token({"status": "success", "data": {"access_token": "token-1"}})
        == "token-1"
    )
    assert upstox_auth.extract_access_token({"access_token": "token-2"}) == "token-2"


def test_extract_access_token_raises_when_missing() -> None:
    with pytest.raises(APIError):
        upstox_auth.extract_access_token({"status": "success", "data": {}})


def test_persist_access_token_upserts_env_lines(tmp_path: Path) -> None:
    target = tmp_path / ".env"
    target.write_text("FOO=bar\nATLAS_UPSTOX_ACCESS_TOKEN=old\n", encoding="utf-8")
    written = upstox_auth.persist_access_token(access_token="new-token", paths=[target])
    assert str(target) in written
    content = target.read_text(encoding="utf-8")
    assert "FOO=bar" in content
    assert "ATLAS_UPSTOX_ACCESS_TOKEN=new-token" in content
    assert "old" not in content


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
