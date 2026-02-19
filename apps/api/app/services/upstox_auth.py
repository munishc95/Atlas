from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib import error, parse, request

from cryptography.fernet import Fernet, InvalidToken
from redis import Redis
from redis.exceptions import RedisError
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.exceptions import APIError
from app.db.models import OAuthState, ProviderCredential

PROVIDER_KIND_UPSTOX = "UPSTOX"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def resolve_client_id(settings: Settings) -> str:
    client_id = str(settings.upstox_client_id or settings.upstox_api_key or "").strip()
    if not client_id:
        raise APIError(
            code="upstox_client_id_missing",
            message="Set ATLAS_UPSTOX_CLIENT_ID (or ATLAS_UPSTOX_API_KEY).",
            status_code=400,
        )
    return client_id


def resolve_client_secret(settings: Settings) -> str:
    client_secret = str(settings.upstox_client_secret or settings.upstox_api_secret or "").strip()
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


def _jwt_claims(token: str) -> dict[str, Any]:
    parts = str(token).split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padding = "=" * ((4 - (len(payload) % 4)) % 4)
    raw = base64.urlsafe_b64decode(payload + padding)
    parsed = json.loads(raw.decode("utf-8"))
    if isinstance(parsed, dict):
        return parsed
    return {}


def _to_utc_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=UTC)
        except (OSError, OverflowError, ValueError):
            return None
    if isinstance(value, str):
        token = value.strip()
        if not token:
            return None
        try:
            parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    return None


def token_timestamps_from_payload(
    *,
    token: str,
    payload: dict[str, Any] | None = None,
) -> tuple[datetime | None, datetime | None]:
    data = payload.get("data") if isinstance(payload, dict) else None
    values: dict[str, Any] = data if isinstance(data, dict) else {}

    issued_at = _to_utc_datetime(values.get("issued_at"))
    expires_at = _to_utc_datetime(values.get("expires_at"))
    if issued_at is not None and expires_at is not None:
        return issued_at, expires_at

    claims = _jwt_claims(token)
    issued_from_jwt = _to_utc_datetime(claims.get("iat"))
    expires_from_jwt = _to_utc_datetime(claims.get("exp"))
    return (issued_at or issued_from_jwt), (expires_at or expires_from_jwt)


def build_fake_access_token(*, issued_at: datetime | None = None, expires_in_seconds: int = 43_200) -> str:
    now = _to_utc_datetime(issued_at) or _utc_now()
    exp_ts = int((now + timedelta(seconds=max(300, int(expires_in_seconds)))).timestamp())
    payload = {"sub": "ATLAS_E2E", "iat": int(now.timestamp()), "exp": exp_ts}
    header = {"typ": "JWT", "alg": "HS256"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).decode("utf-8").rstrip("=")
    payload_b64 = (
        base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")
    )
    return f"{header_b64}.{payload_b64}.atlas-e2e-signature"


def _resolve_cred_key(settings: Settings) -> bytes:
    env_key = str(settings.cred_key or "").strip()
    if env_key:
        return env_key.encode("utf-8")
    key_file = Path(settings.cred_key_path)
    if key_file.exists():
        return key_file.read_bytes().strip()
    generated = Fernet.generate_key()
    key_file.parent.mkdir(parents=True, exist_ok=True)
    key_file.write_bytes(generated + b"\n")
    return generated


def _fernet(settings: Settings) -> Fernet:
    try:
        return Fernet(_resolve_cred_key(settings))
    except Exception as exc:  # noqa: BLE001
        raise APIError(
            code="credential_key_invalid",
            message="Credential encryption key is invalid.",
            details={"error": str(exc)},
            status_code=500,
        ) from exc


def encrypt_token(*, settings: Settings, token: str) -> str:
    cipher = _fernet(settings)
    return cipher.encrypt(str(token).encode("utf-8")).decode("utf-8")


def decrypt_token(*, settings: Settings, ciphertext: str) -> str:
    cipher = _fernet(settings)
    try:
        raw = cipher.decrypt(str(ciphertext).encode("utf-8"))
        return raw.decode("utf-8")
    except InvalidToken as exc:
        raise APIError(
            code="credential_decrypt_failed",
            message="Stored provider credential could not be decrypted.",
            status_code=500,
        ) from exc


def get_provider_credential(
    session: Session,
    *,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
) -> ProviderCredential | None:
    return session.exec(
        select(ProviderCredential)
        .where(ProviderCredential.provider_kind == str(provider_kind).upper())
        .order_by(ProviderCredential.updated_at.desc(), ProviderCredential.id.desc())
    ).first()


def save_provider_credential(
    session: Session,
    *,
    settings: Settings,
    access_token: str,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
    user_id: str | None = None,
    issued_at: datetime | None = None,
    expires_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProviderCredential:
    token = str(access_token).strip()
    if not token:
        raise APIError(code="missing_token", message="Access token is required.")
    now = _utc_now()
    row = get_provider_credential(session, provider_kind=provider_kind)
    if row is None:
        row = ProviderCredential(provider_kind=str(provider_kind).upper())
    row.access_token_encrypted = encrypt_token(settings=settings, token=token)
    row.user_id = str(user_id).strip() if user_id else None
    row.issued_at = _to_utc_datetime(issued_at)
    row.expires_at = _to_utc_datetime(expires_at)
    row.updated_at = now
    if row.created_at is None:
        row.created_at = now
    if metadata:
        row.metadata_json = dict(metadata)
    session.add(row)
    session.commit()
    session.refresh(row)
    return row


def delete_provider_credential(
    session: Session,
    *,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
) -> bool:
    row = get_provider_credential(session, provider_kind=provider_kind)
    if row is None:
        return False
    session.delete(row)
    session.commit()
    return True


def get_provider_access_token(
    session: Session,
    *,
    settings: Settings,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
    allow_env_fallback: bool = True,
) -> str | None:
    row = get_provider_credential(session, provider_kind=provider_kind)
    if row is not None:
        token = decrypt_token(settings=settings, ciphertext=row.access_token_encrypted)
        return str(token).strip() or None
    if allow_env_fallback:
        fallback = str(settings.upstox_access_token or "").strip()
        if fallback:
            return fallback
    return None


def token_status(
    session: Session,
    *,
    settings: Settings,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
    allow_env_fallback: bool = True,
) -> dict[str, Any]:
    now = _utc_now()
    expires_soon_seconds = max(60, int(settings.upstox_expires_soon_seconds))
    row = get_provider_credential(session, provider_kind=provider_kind)
    source = "none"
    token: str | None = None
    issued_at: datetime | None = None
    expires_at: datetime | None = None
    last_verified_at: datetime | None = None
    user_id: str | None = None
    if row is not None:
        source = "encrypted_store"
        token = decrypt_token(settings=settings, ciphertext=row.access_token_encrypted)
        issued_at = _to_utc_datetime(row.issued_at)
        expires_at = _to_utc_datetime(row.expires_at)
        last_verified_at = _to_utc_datetime(row.last_verified_at)
        user_id = row.user_id
    elif allow_env_fallback:
        fallback = str(settings.upstox_access_token or "").strip()
        if fallback:
            source = "env"
            token = fallback

    connected = bool(token)
    if token and expires_at is None:
        issued_at_claim, expires_at_claim = token_timestamps_from_payload(token=token, payload=None)
        issued_at = issued_at or issued_at_claim
        expires_at = expires_at or expires_at_claim
    is_expired = bool(expires_at is not None and expires_at <= now)
    expires_soon = bool(
        connected
        and not is_expired
        and expires_at is not None
        and expires_at <= (now + timedelta(seconds=expires_soon_seconds))
    )
    return {
        "provider_kind": str(provider_kind).upper(),
        "connected": connected,
        "token_masked": mask_token(token or "") if connected else None,
        "token_source": source,
        "issued_at": issued_at.isoformat() if issued_at is not None else None,
        "expires_at": expires_at.isoformat() if expires_at is not None else None,
        "is_expired": is_expired,
        "expires_soon": expires_soon,
        "last_verified_at": last_verified_at.isoformat() if last_verified_at is not None else None,
        "user_id": user_id,
    }


def _redis_client(settings: Settings) -> Redis | None:
    try:
        client = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=0.4,
            socket_timeout=0.4,
            retry_on_timeout=False,
        )
        client.ping()
        return client
    except RedisError:
        return None


def _state_key(*, provider_kind: str, state: str) -> str:
    return f"atlas:oauth-state:{str(provider_kind).upper()}:{state}"


def store_oauth_state(
    session: Session,
    *,
    settings: Settings,
    state: str,
    redirect_uri: str,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
    ttl_seconds: int | None = None,
) -> datetime:
    token = str(state).strip()
    if not token:
        raise APIError(code="oauth_state_missing", message="OAuth state is required.")
    ttl = max(60, int(ttl_seconds or settings.upstox_oauth_state_ttl_seconds))
    now = _utc_now()
    expires_at = now + timedelta(seconds=ttl)

    payload = {"redirect_uri": str(redirect_uri).strip(), "expires_at": expires_at.isoformat()}
    redis_client = _redis_client(settings)
    if redis_client is not None:
        try:
            redis_client.setex(
                _state_key(provider_kind=provider_kind, state=token),
                ttl,
                json.dumps(payload),
            )
            return expires_at
        except RedisError:
            pass

    row = OAuthState(
        provider_kind=str(provider_kind).upper(),
        state=token,
        redirect_uri=str(redirect_uri).strip(),
        expires_at=expires_at,
    )
    session.add(row)
    session.commit()
    return expires_at


def validate_oauth_state(
    session: Session,
    *,
    settings: Settings,
    state: str,
    redirect_uri: str,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
    consume: bool = True,
) -> None:
    token = str(state).strip()
    if not token:
        raise APIError(
            code="oauth_state_missing",
            message="OAuth state is required for token exchange.",
            status_code=400,
        )
    redirect = str(redirect_uri).strip()
    now = _utc_now()

    redis_client = _redis_client(settings)
    if redis_client is not None:
        key = _state_key(provider_kind=provider_kind, state=token)
        try:
            raw = redis_client.get(key)
            if raw is not None:
                parsed = json.loads(raw.decode("utf-8"))
                stored_redirect = str(parsed.get("redirect_uri", "")).strip()
                expires_at = _to_utc_datetime(parsed.get("expires_at"))
                if stored_redirect and stored_redirect != redirect:
                    raise APIError(
                        code="oauth_state_mismatch",
                        message="OAuth state validation failed (redirect mismatch).",
                        status_code=400,
                    )
                if expires_at is not None and expires_at <= now:
                    raise APIError(
                        code="oauth_state_expired",
                        message="OAuth state expired. Restart connect flow.",
                        status_code=400,
                    )
                if consume:
                    redis_client.delete(key)
                return
        except APIError:
            raise
        except (RedisError, json.JSONDecodeError):
            pass

    row = session.exec(
        select(OAuthState)
        .where(OAuthState.provider_kind == str(provider_kind).upper())
        .where(OAuthState.state == token)
        .order_by(OAuthState.created_at.desc(), OAuthState.id.desc())
    ).first()
    if row is None:
        raise APIError(
            code="oauth_state_not_found",
            message="OAuth state not found. Restart connect flow.",
            status_code=400,
        )
    if row.consumed_at is not None:
        raise APIError(
            code="oauth_state_consumed",
            message="OAuth state already consumed. Restart connect flow.",
            status_code=400,
        )
    expires_at_row = _to_utc_datetime(row.expires_at)
    if expires_at_row is None or expires_at_row <= now:
        raise APIError(
            code="oauth_state_expired",
            message="OAuth state expired. Restart connect flow.",
            status_code=400,
        )
    if str(row.redirect_uri).strip() != redirect:
        raise APIError(
            code="oauth_state_mismatch",
            message="OAuth state validation failed (redirect mismatch).",
            status_code=400,
        )
    if consume:
        row.consumed_at = now
        session.add(row)
        session.commit()


def mark_verified_now(
    session: Session,
    *,
    settings: Settings,
    provider_kind: str = PROVIDER_KIND_UPSTOX,
) -> None:
    row = get_provider_credential(session, provider_kind=provider_kind)
    if row is None:
        return
    row.last_verified_at = _utc_now()
    row.updated_at = _utc_now()
    session.add(row)
    session.commit()
