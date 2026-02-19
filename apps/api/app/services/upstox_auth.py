from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from app.core.config import Settings
from app.core.exceptions import APIError


def resolve_client_id(settings: Settings) -> str:
    client_id = str(
        settings.upstox_client_id or settings.upstox_api_key or ""
    ).strip()
    if not client_id:
        raise APIError(
            code="upstox_client_id_missing",
            message="Set ATLAS_UPSTOX_CLIENT_ID (or ATLAS_UPSTOX_API_KEY).",
            status_code=400,
        )
    return client_id


def resolve_client_secret(settings: Settings) -> str:
    client_secret = str(
        settings.upstox_client_secret or settings.upstox_api_secret or ""
    ).strip()
    if not client_secret:
        raise APIError(
            code="upstox_client_secret_missing",
            message="Set ATLAS_UPSTOX_CLIENT_SECRET (or ATLAS_UPSTOX_API_SECRET).",
            status_code=400,
        )
    return client_secret


def resolve_redirect_uri(*, settings: Settings, redirect_uri: str | None = None) -> str:
    resolved = str(redirect_uri or settings.upstox_redirect_uri or "").strip()
    if not resolved:
        raise APIError(
            code="upstox_redirect_uri_missing",
            message="Provide redirect_uri or set ATLAS_UPSTOX_REDIRECT_URI.",
            status_code=400,
        )
    return resolved


def build_authorization_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    base_url: str,
) -> str:
    query = parse.urlencode(
        {
            "response_type": "code",
            "client_id": str(client_id).strip(),
            "redirect_uri": str(redirect_uri).strip(),
            "state": str(state).strip(),
        }
    )
    return f"{str(base_url).rstrip('/')}/v2/login/authorization/dialog?{query}"


def _token_request_payload(
    *,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
) -> bytes:
    encoded = parse.urlencode(
        {
            "code": str(code).strip(),
            "client_id": str(client_id).strip(),
            "client_secret": str(client_secret).strip(),
            "redirect_uri": str(redirect_uri).strip(),
            "grant_type": "authorization_code",
        }
    )
    return encoded.encode("utf-8")


def exchange_authorization_code(
    *,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not str(code).strip():
        raise APIError(code="missing_code", message="Authorization code is required.")
    url = f"{str(base_url).rstrip('/')}/v2/login/authorization/token"
    payload = _token_request_payload(
        code=code,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
    )
    req = request.Request(
        url=url,
        data=payload,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "Api-Version": "2.0",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=float(timeout_seconds)) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(raw)
        except json.JSONDecodeError:
            details = {"raw": raw}
        raise APIError(
            code="upstox_token_exchange_failed",
            message=f"Token exchange failed with status {exc.code}.",
            status_code=400,
            details=details,
        ) from exc
    except (error.URLError, TimeoutError) as exc:
        raise APIError(
            code="upstox_token_exchange_failed",
            message="Token exchange request failed.",
            status_code=400,
            details={"error": str(exc)},
        ) from exc

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        raise APIError(
            code="upstox_token_exchange_failed",
            message="Token exchange returned non-JSON response.",
            status_code=400,
        ) from exc
    return parsed


def extract_access_token(payload: dict[str, Any]) -> str:
    data = payload.get("data")
    if isinstance(data, dict):
        token = str(data.get("access_token", "")).strip()
        if token:
            return token
    token = str(payload.get("access_token", "")).strip()
    if token:
        return token
    raise APIError(
        code="upstox_token_exchange_failed",
        message="Token exchange response did not include access_token.",
        status_code=400,
        details=payload,
    )


def verify_access_token(
    *,
    access_token: str,
    base_url: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    token = str(access_token).strip()
    if not token:
        raise APIError(code="missing_token", message="Access token is required.")
    url = f"{str(base_url).rstrip('/')}/v2/user/profile"
    req = request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        },
        method="GET",
    )
    try:
        with request.urlopen(req, timeout=float(timeout_seconds)) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            if not isinstance(parsed, dict):
                return {"valid": True}
            parsed["valid"] = True
            return parsed
    except error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            details = json.loads(raw)
        except json.JSONDecodeError:
            details = {"raw": raw}
        return {
            "valid": False,
            "status_code": int(exc.code),
            "error": details,
        }
    except (error.URLError, TimeoutError) as exc:
        return {"valid": False, "error": str(exc)}


def mask_token(value: str) -> str:
    token = str(value or "").strip()
    if len(token) <= 12:
        return "***"
    return f"{token[:8]}...{token[-6:]}"


def _upsert_env_var(*, text: str, key: str, value: str) -> str:
    lines = text.splitlines()
    target = f"{key}={value}"
    replaced = False
    out: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            out.append(target)
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(target)
    cleaned = "\n".join(out).strip("\n")
    return f"{cleaned}\n"


def persist_access_token(
    *,
    access_token: str,
    paths: list[Path] | None = None,
) -> list[str]:
    token = str(access_token).strip()
    if not token:
        raise APIError(code="missing_token", message="Access token is required.")
    targets = paths or [Path(".env"), Path("apps/api/.env")]
    written: list[str] = []
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if target.exists():
            existing = target.read_text(encoding="utf-8")
        updated = _upsert_env_var(
            text=existing,
            key="ATLAS_UPSTOX_ACCESS_TOKEN",
            value=token,
        )
        target.write_text(updated, encoding="utf-8", newline="\n")
        written.append(str(target))
    return written
