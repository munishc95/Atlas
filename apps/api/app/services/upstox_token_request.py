from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from urllib import error, parse, request
from zoneinfo import ZoneInfo

from sqlalchemy.exc import IntegrityError, OperationalError
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import PaperState, UpstoxNotifierEvent, UpstoxTokenRequestRun
from app.services.fast_mode import fast_mode_enabled
from app.services.operate_events import emit_operate_event
from app.services.upstox_auth import (
    build_fake_access_token,
    mark_verified_now,
    resolve_client_id,
    resolve_client_secret,
    save_provider_credential,
)

PROVIDER_KIND_UPSTOX = "UPSTOX"
STATUS_PENDING = "PENDING"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"
STATUS_EXPIRED = "EXPIRED"
STATUS_ERROR = "ERROR"

# Backward-compatible aliases.
STATUS_REQUESTED = STATUS_PENDING
STATUS_FAILED = STATUS_ERROR

REASON_NOTIFIER_RECEIVED = "notifier_received"
REASON_EXPIRED_NO_NOTIFIER = "expired_no_notifier"
REASON_INVALID_PAYLOAD = "invalid_payload"
REASON_CLIENT_ID_MISMATCH = "client_id_mismatch"
REASON_NONCE_MISMATCH = "nonce_mismatch"
REASON_SECRET_MISMATCH = "secret_mismatch"
REASON_TOKEN_STORE_FAILED = "token_store_failed"

VALID_MESSAGE_TYPES = {"access_token", "token_issued"}
IST_ZONE = ZoneInfo("Asia/Kolkata")


def _utc_now() -> datetime:
    return datetime.now(UTC)


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


def _default_authorization_expiry(reference: datetime | None = None) -> datetime:
    now_utc = reference or _utc_now()
    now_ist = now_utc.astimezone(IST_ZONE)
    expiry_ist = datetime.combine(
        now_ist.date() + timedelta(days=1),
        dt_time(3, 30),
        tzinfo=IST_ZONE,
    )
    return expiry_ist.astimezone(UTC)


def _default_token_expiry(reference: datetime | None = None) -> datetime:
    return _default_authorization_expiry(reference)


def _normalize_run_status(status: str | None) -> str:
    token = str(status or "").strip().upper()
    if token in {"REQUESTED", "PENDING"}:
        return STATUS_PENDING
    if token in {"FAILED", "ERROR"}:
        return STATUS_ERROR
    if token in {STATUS_APPROVED, STATUS_REJECTED, STATUS_EXPIRED}:
        return token
    return STATUS_PENDING


def _legacy_run_status(status: str | None) -> str:
    token = _normalize_run_status(status)
    if token == STATUS_PENDING:
        return "REQUESTED"
    if token == STATUS_ERROR:
        return "FAILED"
    return token


def _canonical_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _payload_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_payload(payload).encode("utf-8")).hexdigest()


def _notifier_base_url(settings: Settings) -> str:
    base = str(settings.upstox_notifier_base_url or "").strip()
    if base:
        return base.rstrip("/")
    return "http://127.0.0.1:8000"


def _secret_file_path(settings: Settings) -> Path:
    return Path(settings.secrets_root) / "upstox_notifier_secret.txt"


def _get_or_create_state(session: Session, settings: Settings) -> PaperState:
    state = session.get(PaperState, 1)
    if state is not None:
        return state
    state = PaperState(id=1, settings_json={})
    session.add(state)
    session.commit()
    session.refresh(state)
    return state


def _persist_secret_file(settings: Settings, secret: str) -> None:
    try:
        path = _secret_file_path(settings)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{secret}\n", encoding="utf-8")
    except Exception:  # noqa: BLE001
        return


def get_notifier_secret(session: Session, *, settings: Settings) -> str:
    env_secret = str(settings.upstox_notifier_secret or "").strip()
    if env_secret:
        return env_secret

    path = _secret_file_path(settings)
    if path.exists():
        token = str(path.read_text(encoding="utf-8").strip())
        if token:
            return token

    state = _get_or_create_state(session, settings)
    merged = dict(state.settings_json or {})
    existing = str(merged.get("upstox_notifier_secret") or "").strip()
    if existing:
        return existing

    generated = secrets.token_urlsafe(24)
    _persist_secret_file(settings, generated)
    merged["upstox_notifier_secret"] = generated
    state.settings_json = merged
    try:
        session.add(state)
        session.commit()
        session.refresh(state)
    except OperationalError:
        session.rollback()
    return generated


def legacy_notifier_endpoint(*, settings: Settings, nonce: str | None = None) -> str:
    base = _notifier_base_url(settings)
    query = str(nonce or "").strip()
    if query:
        return f"{base}/api/providers/upstox/notifier?nonce={parse.quote(query)}"
    return f"{base}/api/providers/upstox/notifier"


def recommended_notifier_endpoint(
    *,
    settings: Settings,
    session: Session | None = None,
    nonce: str | None = None,
    include_nonce_query: bool = False,
) -> str:
    secret = str(settings.upstox_notifier_secret or "").strip()
    if not secret and session is not None:
        secret = get_notifier_secret(session, settings=settings)
    if not secret:
        return legacy_notifier_endpoint(
            settings=settings,
            nonce=(nonce if include_nonce_query else None),
        )
    endpoint = f"{_notifier_base_url(settings)}/api/providers/upstox/notifier/{parse.quote(secret)}"
    if include_nonce_query and str(nonce or "").strip():
        endpoint = f"{endpoint}?nonce={parse.quote(str(nonce).strip())}"
    return endpoint


def _extract_response_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


def _parse_authorization_expiry(payload: dict[str, Any]) -> datetime | None:
    data = _extract_response_data(payload)
    for key in ("authorization_expiry", "authorizationExpiry", "expires_at", "expiry", "expiresAt"):
        parsed = _to_utc_datetime(data.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_notifier_url(payload: dict[str, Any]) -> str | None:
    data = _extract_response_data(payload)
    for key in ("notifier_url", "notifierUrl", "webhook_url", "webhookUrl"):
        token = str(data.get(key, "")).strip()
        if token:
            return token
    return None


def _parse_user_id(payload: dict[str, Any]) -> str | None:
    data = _extract_response_data(payload)
    for key in ("user_id", "userId"):
        token = str(data.get(key, "")).strip()
        if token:
            return token
    return None


def _nonce_for_request(*, settings: Settings, client_id: str) -> str:
    if fast_mode_enabled(settings):
        seed = f"{client_id}:{_utc_now().date().isoformat()}:{settings.fast_mode_seed}"
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]
    return secrets.token_urlsafe(18)


def _request_upstox_token(
    *, settings: Settings, client_id: str, client_secret: str
) -> dict[str, Any]:
    url = f"{str(settings.upstox_base_url).rstrip('/')}/v3/login/auth/token/request/{client_id}"
    payload = json.dumps({"client_secret": client_secret}).encode("utf-8")
    req = request.Request(
        url=url,
        data=payload,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        },
    )
    with request.urlopen(req, timeout=float(settings.upstox_timeout_seconds)) as response:  # noqa: S310
        body = response.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        return {"status": "success"}
    return parsed


def serialize_request_run(row: UpstoxTokenRequestRun) -> dict[str, Any]:
    status = _normalize_run_status(row.status)
    return {
        "id": row.id,
        "provider_kind": row.provider_kind,
        "status": status,
        "status_legacy": _legacy_run_status(status),
        "requested_at": row.requested_at.isoformat() if row.requested_at is not None else None,
        "authorization_expiry": (
            row.authorization_expiry.isoformat() if row.authorization_expiry is not None else None
        ),
        "approved_at": row.approved_at.isoformat() if row.approved_at is not None else None,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at is not None else None,
        "resolution_reason": row.resolution_reason,
        "notifier_url": row.notifier_url,
        "client_id": row.client_id,
        "user_id": row.user_id,
        "correlation_nonce": row.correlation_nonce,
        "last_error": row.last_error,
        "created_at": row.created_at.isoformat() if row.created_at is not None else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at is not None else None,
    }


def _pending_status_filter():
    return UpstoxTokenRequestRun.status.in_([STATUS_PENDING, "REQUESTED"])


def sweep_expired_request_runs(
    session: Session,
    *,
    settings: Settings,
    correlation_id: str | None = None,
    now: datetime | None = None,
) -> int:
    at = now or _utc_now()
    rows = session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == PROVIDER_KIND_UPSTOX)
        .where(_pending_status_filter())
    ).all()
    changed: list[UpstoxTokenRequestRun] = []
    for row in rows:
        expiry = _to_utc_datetime(row.authorization_expiry)
        if expiry is None or expiry > at:
            continue
        row.status = STATUS_EXPIRED
        row.resolved_at = at
        row.resolution_reason = REASON_EXPIRED_NO_NOTIFIER
        row.updated_at = at
        row.last_error = row.last_error or {"reason": REASON_EXPIRED_NO_NOTIFIER}
        session.add(row)
        changed.append(row)
    if changed:
        session.commit()
        for row in changed:
            emit_operate_event(
                session,
                severity="WARN",
                category="SYSTEM",
                message="upstox_token_request_expired",
                details={"run_id": row.id, "status": row.status},
                correlation_id=correlation_id,
                commit=True,
            )
    return len(changed)


def _latest_pending_run(session: Session) -> UpstoxTokenRequestRun | None:
    return session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == PROVIDER_KIND_UPSTOX)
        .where(_pending_status_filter())
        .order_by(UpstoxTokenRequestRun.requested_at.desc(), UpstoxTokenRequestRun.id.desc())
    ).first()


def ensure_test_pending_run(
    session: Session,
    *,
    settings: Settings,
    source: str = "notifier_test",
) -> UpstoxTokenRequestRun:
    now = _utc_now()
    sweep_expired_request_runs(session, settings=settings, correlation_id=None, now=now)
    existing = _latest_pending_run(session)
    if existing is not None:
        metadata = existing.metadata_json if isinstance(existing.metadata_json, dict) else {}
        if str(metadata.get("source") or "").strip() == source:
            return existing
    client_id = str(settings.upstox_client_id or settings.upstox_api_key or "ATLAS_FASTMODE").strip()
    nonce = _nonce_for_request(settings=settings, client_id=client_id)
    run = UpstoxTokenRequestRun(
        provider_kind=PROVIDER_KIND_UPSTOX,
        status=STATUS_PENDING,
        requested_at=now,
        authorization_expiry=now + timedelta(hours=1),
        approved_at=None,
        resolved_at=None,
        resolution_reason=None,
        notifier_url=recommended_notifier_endpoint(
            settings=settings,
            session=session,
            nonce=nonce,
            include_nonce_query=False,
        ),
        client_id=client_id,
        user_id=None,
        correlation_nonce=nonce,
        last_error=None,
        metadata_json={"source": source, "mode": "local_test"},
        created_at=now,
        updated_at=now,
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def request_token_run(
    session: Session,
    *,
    settings: Settings,
    correlation_id: str | None = None,
    source: str = "manual",
) -> tuple[UpstoxTokenRequestRun, bool]:
    now = _utc_now()
    sweep_expired_request_runs(
        session,
        settings=settings,
        correlation_id=correlation_id,
        now=now,
    )
    existing = _latest_pending_run(session)
    if existing is not None:
        expiry = _to_utc_datetime(existing.authorization_expiry)
        if expiry is not None and expiry > now:
            return existing, True

    client_id = (
        str(settings.upstox_client_id or settings.upstox_api_key or "ATLAS_FASTMODE").strip()
        if fast_mode_enabled(settings)
        else resolve_client_id(settings)
    )
    nonce = _nonce_for_request(settings=settings, client_id=client_id)
    run = UpstoxTokenRequestRun(
        provider_kind=PROVIDER_KIND_UPSTOX,
        status=STATUS_PENDING,
        requested_at=now,
        authorization_expiry=None,
        approved_at=None,
        resolved_at=None,
        resolution_reason=None,
        notifier_url=recommended_notifier_endpoint(
            settings=settings,
            session=session,
            nonce=nonce,
            include_nonce_query=False,
        ),
        client_id=client_id,
        user_id=None,
        correlation_nonce=nonce,
        last_error=None,
        metadata_json={"source": source},
        created_at=now,
        updated_at=now,
    )
    session.add(run)
    session.commit()
    session.refresh(run)

    if fast_mode_enabled(settings):
        run.authorization_expiry = now + timedelta(hours=1)
        run.updated_at = _utc_now()
        session.add(run)
        session.commit()
        session.refresh(run)
        emit_operate_event(
            session,
            severity="INFO",
            category="SYSTEM",
            message="upstox_token_request_initiated",
            details={
                "run_id": run.id,
                "status": run.status,
                "authorization_expiry": run.authorization_expiry.isoformat(),
                "source": source,
                "mode": "fast",
            },
            correlation_id=correlation_id,
            commit=True,
        )
        fake_payload = {
            "client_id": run.client_id,
            "user_id": "ATLAS_E2E",
            "access_token": build_fake_access_token(issued_at=now),
            "token_type": "Bearer",
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=12)).isoformat(),
            "message_type": "access_token",
        }
        process_notifier_payload(
            session,
            settings=settings,
            payload=fake_payload,
            nonce=run.correlation_nonce,
            headers={"x-atlas-source": "fast_mode"},
            secret_valid=True,
            correlation_id=correlation_id,
            source="fast_mode",
        )
        session.refresh(run)
        return run, False

    try:
        client_secret = resolve_client_secret(settings)
        raw = _request_upstox_token(
            settings=settings,
            client_id=client_id,
            client_secret=client_secret,
        )
        run.authorization_expiry = _parse_authorization_expiry(raw) or _default_authorization_expiry(now)
        run.user_id = _parse_user_id(raw) or run.user_id
        run.status = STATUS_PENDING
        run.resolved_at = None
        run.resolution_reason = None
        run.last_error = None
        run.updated_at = _utc_now()
        run.metadata_json = {
            **(run.metadata_json or {}),
            "response_status": raw.get("status") if isinstance(raw, dict) else None,
            "upstream_notifier_url": _parse_notifier_url(raw),
        }
        session.add(run)
        session.commit()
        session.refresh(run)
        emit_operate_event(
            session,
            severity="INFO",
            category="SYSTEM",
            message="upstox_token_request_initiated",
            details={
                "run_id": run.id,
                "status": run.status,
                "authorization_expiry": run.authorization_expiry.isoformat()
                if run.authorization_expiry is not None
                else None,
                "source": source,
                "mode": "real",
            },
            correlation_id=correlation_id,
            commit=True,
        )
        return run, False
    except Exception as exc:  # noqa: BLE001
        details: dict[str, Any] = {"run_id": run.id, "reason": str(exc), "source": source}
        if isinstance(exc, error.HTTPError):
            try:
                details["http_status"] = int(exc.code)
                details["raw"] = exc.read().decode("utf-8", errors="replace")[:500]
            except Exception:  # noqa: BLE001
                details["http_status"] = int(exc.code)
        run.status = STATUS_ERROR
        run.resolved_at = _utc_now()
        run.resolution_reason = REASON_INVALID_PAYLOAD
        run.last_error = details
        run.updated_at = _utc_now()
        session.add(run)
        session.commit()
        session.refresh(run)
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_token_request_failed",
            details={
                "run_id": run.id,
                "status": run.status,
                "authorization_expiry": run.authorization_expiry.isoformat()
                if run.authorization_expiry is not None
                else None,
                "error": str(exc),
                "source": source,
            },
            correlation_id=correlation_id,
            commit=True,
        )
        return run, False


def _match_pending_run(
    session: Session,
    *,
    client_id: str | None,
) -> UpstoxTokenRequestRun | None:
    now = _utc_now()
    rows = session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == PROVIDER_KIND_UPSTOX)
        .where(_pending_status_filter())
        .order_by(UpstoxTokenRequestRun.requested_at.desc(), UpstoxTokenRequestRun.id.desc())
    ).all()
    for row in rows:
        expiry = _to_utc_datetime(row.authorization_expiry)
        if expiry is not None and expiry <= now:
            continue
        if client_id and str(row.client_id).strip() and str(row.client_id).strip() != client_id:
            continue
        return row
    return None


def _normalize_headers(headers: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(headers, dict) or not headers:
        return None
    out: dict[str, Any] = {}
    for key, value in headers.items():
        token = str(key or "").strip().lower()
        if not token or token in {"authorization", "cookie", "set-cookie"}:
            continue
        out[token] = str(value)
    return out or None


def _store_notifier_event(
    session: Session,
    *,
    payload: dict[str, Any],
    headers: dict[str, Any] | None,
    correlated_request_run_id: str | None = None,
) -> tuple[UpstoxNotifierEvent, bool]:
    digest = _payload_digest(payload)
    row = UpstoxNotifierEvent(
        client_id=str(payload.get("client_id") or "").strip(),
        user_id=str(payload.get("user_id") or "").strip() or None,
        message_type=str(payload.get("message_type") or "").strip() or None,
        issued_at=_to_utc_datetime(payload.get("issued_at")),
        expires_at=_to_utc_datetime(payload.get("expires_at")),
        payload_digest=digest,
        raw_payload_json=dict(payload),
        headers_json=_normalize_headers(headers),
        correlated_request_run_id=correlated_request_run_id,
    )
    session.add(row)
    try:
        session.commit()
        session.refresh(row)
        return row, False
    except IntegrityError:
        session.rollback()
        existing = session.exec(
            select(UpstoxNotifierEvent).where(UpstoxNotifierEvent.payload_digest == digest)
        ).first()
        if existing is not None:
            return existing, True
        raise


def _mask_user_id(value: str | None) -> str | None:
    token = str(value or "").strip()
    if not token:
        return None
    if len(token) <= 3:
        return "***"
    return f"{'*' * (len(token) - 3)}{token[-3:]}"


def _redacted_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    out = dict(payload)
    if "access_token" in out:
        out["access_token"] = "***REDACTED***"
    user_id = out.get("user_id")
    if user_id:
        out["user_id"] = _mask_user_id(str(user_id))
    return out


def serialize_notifier_event(
    row: UpstoxNotifierEvent,
    *,
    run_lookup: dict[str, UpstoxTokenRequestRun] | None = None,
) -> dict[str, Any]:
    payload = {
        "id": row.id,
        "received_at": row.received_at.isoformat() if row.received_at is not None else None,
        "client_id": row.client_id,
        "user_id": _mask_user_id(row.user_id),
        "message_type": row.message_type,
        "issued_at": row.issued_at.isoformat() if row.issued_at is not None else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at is not None else None,
        "payload_digest": row.payload_digest,
        "raw_payload_json": _redacted_payload(row.raw_payload_json),
        "headers_json": row.headers_json or {},
        "correlated_request_run_id": row.correlated_request_run_id,
        "created_at": row.created_at.isoformat() if row.created_at is not None else None,
    }
    if run_lookup and row.correlated_request_run_id:
        run = run_lookup.get(str(row.correlated_request_run_id))
        if run is not None:
            payload["correlated_request_status"] = _normalize_run_status(run.status)
            payload["correlated_resolution_reason"] = run.resolution_reason
    return payload


def list_notifier_events(
    session: Session,
    *,
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[UpstoxNotifierEvent], int]:
    safe_limit = max(1, min(int(limit), 200))
    safe_offset = max(0, int(offset))
    stmt = select(UpstoxNotifierEvent).order_by(
        UpstoxNotifierEvent.received_at.desc(),
        UpstoxNotifierEvent.id.desc(),
    )
    rows = list(session.exec(stmt.offset(safe_offset).limit(safe_limit)).all())
    total = len(session.exec(select(UpstoxNotifierEvent.id)).all())
    return rows, total


def process_notifier_payload(
    session: Session,
    *,
    settings: Settings,
    payload: dict[str, Any] | None,
    nonce: str | None,
    headers: dict[str, Any] | None = None,
    secret_valid: bool = True,
    correlation_id: str | None = None,
    source: str = "webhook",
) -> dict[str, Any]:
    body = payload if isinstance(payload, dict) else {}
    sweep_expired_request_runs(session, settings=settings, correlation_id=correlation_id)
    notifier_event, duplicated = _store_notifier_event(
        session,
        payload=body,
        headers=headers,
    )
    if duplicated:
        return {
            "matched": bool(notifier_event.correlated_request_run_id),
            "accepted": False,
            "reason": "duplicate",
            "run_id": notifier_event.correlated_request_run_id,
            "event_id": notifier_event.id,
            "deduplicated": True,
        }

    client_id = str(body.get("client_id") or "").strip() or None
    user_id = str(body.get("user_id") or "").strip() or None
    access_token = str(body.get("access_token") or "").strip()
    message_type = str(body.get("message_type") or "").strip().lower()
    incoming_nonce = str(nonce or "").strip()
    expected_client_id = str(settings.upstox_client_id or settings.upstox_api_key or "").strip()

    if not secret_valid:
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_invalid_secret",
            details={"source": source, "event_id": notifier_event.id},
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": False,
            "accepted": False,
            "reason": REASON_SECRET_MISMATCH,
            "event_id": notifier_event.id,
        }

    if not client_id:
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_invalid_payload",
            details={"source": source, "event_id": notifier_event.id, "reason": "missing_client_id"},
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": False,
            "accepted": False,
            "reason": REASON_INVALID_PAYLOAD,
            "event_id": notifier_event.id,
        }

    if expected_client_id and client_id != expected_client_id:
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_client_id_mismatch",
            details={
                "source": source,
                "event_id": notifier_event.id,
                "expected_client_id": expected_client_id,
                "incoming_client_id": client_id,
            },
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": False,
            "accepted": False,
            "reason": REASON_CLIENT_ID_MISMATCH,
            "event_id": notifier_event.id,
        }

    run = _match_pending_run(session, client_id=client_id)
    if run is None:
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_unmatched",
            details={"source": source, "client_id": client_id, "event_id": notifier_event.id},
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": False,
            "accepted": False,
            "reason": "unmatched",
            "event_id": notifier_event.id,
        }

    notifier_event.correlated_request_run_id = run.id
    session.add(notifier_event)
    session.commit()
    session.refresh(notifier_event)

    if incoming_nonce and incoming_nonce != str(run.correlation_nonce or "").strip():
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_nonce_mismatch",
            details={"run_id": run.id, "source": source, "event_id": notifier_event.id},
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": True,
            "accepted": False,
            "reason": REASON_NONCE_MISMATCH,
            "run_id": run.id,
            "event_id": notifier_event.id,
        }

    if message_type and message_type not in VALID_MESSAGE_TYPES:
        if message_type in {"rejected", "deny", "denied"}:
            run.status = STATUS_REJECTED
            run.resolved_at = _utc_now()
            run.resolution_reason = REASON_INVALID_PAYLOAD
            run.updated_at = _utc_now()
            run.last_error = {"reason": "upstox_rejected", "message_type": message_type}
            session.add(run)
            session.commit()
            session.refresh(run)
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_invalid_payload",
            details={
                "run_id": run.id,
                "source": source,
                "event_id": notifier_event.id,
                "reason": "unexpected_message_type",
                "message_type": message_type,
            },
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": True,
            "accepted": False,
            "reason": REASON_INVALID_PAYLOAD,
            "run_id": run.id,
            "event_id": notifier_event.id,
        }

    if not access_token:
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_missing_token",
            details={"run_id": run.id, "source": source, "event_id": notifier_event.id},
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": True,
            "accepted": False,
            "reason": "access_token_missing",
            "run_id": run.id,
            "event_id": notifier_event.id,
        }

    issued_at = _to_utc_datetime(body.get("issued_at")) or _utc_now()
    expires_at = _to_utc_datetime(body.get("expires_at")) or _default_token_expiry(issued_at)
    try:
        row = save_provider_credential(
            session,
            settings=settings,
            access_token=access_token,
            provider_kind=PROVIDER_KIND_UPSTOX,
            user_id=user_id,
            issued_at=issued_at,
            expires_at=expires_at,
            metadata={"source": source, "message_type": body.get("message_type")},
        )
        mark_verified_now(session, settings=settings)
    except Exception as exc:  # noqa: BLE001
        run.status = STATUS_ERROR
        run.resolved_at = _utc_now()
        run.resolution_reason = REASON_TOKEN_STORE_FAILED
        run.updated_at = _utc_now()
        run.last_error = {"reason": REASON_TOKEN_STORE_FAILED, "error": str(exc)}
        session.add(run)
        session.commit()
        session.refresh(run)
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_token_store_failed",
            details={"run_id": run.id, "source": source, "error": str(exc), "event_id": notifier_event.id},
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": True,
            "accepted": False,
            "reason": REASON_TOKEN_STORE_FAILED,
            "run_id": run.id,
            "event_id": notifier_event.id,
        }

    run.status = STATUS_APPROVED
    run.approved_at = _utc_now()
    run.resolved_at = _utc_now()
    run.resolution_reason = REASON_NOTIFIER_RECEIVED
    run.user_id = user_id or run.user_id
    run.last_error = None
    run.updated_at = _utc_now()
    if run.authorization_expiry is None:
        run.authorization_expiry = expires_at
    session.add(run)
    session.commit()
    session.refresh(run)

    emit_operate_event(
        session,
        severity="INFO",
        category="SYSTEM",
        message="upstox_token_received",
        details={
            "run_id": run.id,
            "source": source,
            "expires_at": row.expires_at.isoformat() if row.expires_at is not None else None,
            "user_id": _mask_user_id(row.user_id),
            "event_id": notifier_event.id,
        },
        correlation_id=correlation_id,
        commit=True,
    )
    return {
        "matched": True,
        "accepted": True,
        "reason": None,
        "run_id": run.id,
        "status": run.status,
        "event_id": notifier_event.id,
    }


def latest_request_run(session: Session) -> UpstoxTokenRequestRun | None:
    return session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == PROVIDER_KIND_UPSTOX)
        .order_by(UpstoxTokenRequestRun.requested_at.desc(), UpstoxTokenRequestRun.id.desc())
    ).first()


def list_request_runs(
    session: Session,
    *,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[UpstoxTokenRequestRun], int]:
    safe_page = max(1, int(page))
    safe_size = max(1, min(200, int(page_size)))
    rows = session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == PROVIDER_KIND_UPSTOX)
        .order_by(UpstoxTokenRequestRun.requested_at.desc(), UpstoxTokenRequestRun.id.desc())
    ).all()
    total = len(rows)
    start = (safe_page - 1) * safe_size
    end = start + safe_size
    return list(rows[start:end]), total


def _pending_request_summary(session: Session, *, settings: Settings) -> dict[str, Any] | None:
    run = _latest_pending_run(session)
    if run is None:
        return None
    now = _utc_now()
    waiting = max(
        0.0,
        (now - (run.requested_at if run.requested_at is not None else now)).total_seconds() / 60.0,
    )
    return {
        "id": run.id,
        "status": _normalize_run_status(run.status),
        "requested_at": run.requested_at.isoformat() if run.requested_at is not None else None,
        "authorization_expiry": (
            run.authorization_expiry.isoformat() if run.authorization_expiry is not None else None
        ),
        "minutes_waiting": round(waiting, 3),
    }


def notifier_health_payload(session: Session, *, settings: Settings) -> dict[str, Any]:
    sweep_expired_request_runs(session, settings=settings, correlation_id=None)
    latest_event = session.exec(
        select(UpstoxNotifierEvent).order_by(
            UpstoxNotifierEvent.received_at.desc(),
            UpstoxNotifierEvent.id.desc(),
        )
    ).first()
    pending_request = _pending_request_summary(session, settings=settings)
    last_received = (
        _to_utc_datetime(latest_event.received_at) if latest_event is not None else None
    )
    threshold_minutes = max(1, int(settings.upstox_notifier_pending_no_callback_minutes))
    stale_hours = max(1, int(settings.upstox_notifier_stale_hours))

    pending_no_callback = False
    if pending_request is not None:
        wait_minutes = float(pending_request.get("minutes_waiting") or 0.0)
        requested_at = _to_utc_datetime(pending_request.get("requested_at"))
        callback_after_request = bool(
            last_received is not None
            and requested_at is not None
            and last_received >= requested_at
        )
        pending_no_callback = wait_minutes >= float(threshold_minutes) and not callback_after_request

    status = "OK"
    if last_received is None:
        status = "NEVER_RECEIVED"
    if pending_no_callback:
        status = "FAILING"
    elif last_received is not None:
        age_hours = (_utc_now() - last_received).total_seconds() / 3600.0
        if age_hours >= float(stale_hours):
            status = "STALE"

    return {
        "last_notifier_received_at": (
            last_received.isoformat() if last_received is not None else None
        ),
        "status": status,
        "pending_request": pending_request,
        "pending_no_callback": pending_no_callback,
        "pending_threshold_minutes": threshold_minutes,
    }


def notifier_status_payload(session: Session, *, settings: Settings) -> dict[str, Any]:
    secret = str(settings.upstox_notifier_secret or "").strip() or get_notifier_secret(
        session,
        settings=settings,
    )
    recommended_url = recommended_notifier_endpoint(
        settings=settings,
        session=session,
        include_nonce_query=False,
    )
    legacy_url = legacy_notifier_endpoint(settings=settings, nonce=None)
    health = notifier_health_payload(session, settings=settings)
    latest = latest_request_run(session)
    suggested_actions: list[str] = []
    if not bool(health.get("last_notifier_received_at")):
        suggested_actions.append("Send test webhook from Atlas Settings.")
    if bool(health.get("pending_no_callback")):
        suggested_actions.append(
            "Pending request has no callback. Verify notifier URL in Upstox My Apps and keep tunnel active."
        )
    suggested_actions.append(
        "Use a public tunnel (ngrok/cloudflared) and configure notifier URL in Upstox My Apps."
    )
    suggested_actions.append("Request token now if token is missing or expired.")
    return {
        "recommended_notifier_url": recommended_url,
        "legacy_notifier_url": legacy_url,
        "legacy_route_security": "less_secure",
        "secret_configured": bool(secret),
        "webhook_health": health,
        "last_request_run": (serialize_request_run(latest) if latest is not None else None),
        "suggested_actions": suggested_actions,
    }


def auto_renew_meta(
    *,
    settings: Settings,
    state_settings: dict[str, Any],
    expires_at: str | None,
) -> dict[str, Any]:
    enabled = bool(
        state_settings.get("upstox_auto_renew_enabled", settings.upstox_auto_renew_enabled)
    )
    threshold_hours = max(
        1,
        int(
            state_settings.get(
                "upstox_auto_renew_if_expires_within_hours",
                settings.upstox_auto_renew_if_expires_within_hours,
            )
        ),
    )
    from app.services.operate_scheduler import compute_next_scheduled_run_ist

    next_run = compute_next_scheduled_run_ist(
        auto_run_enabled=enabled,
        auto_run_time_ist=str(
            state_settings.get("upstox_auto_renew_time_ist", settings.upstox_auto_renew_time_ist)
        ),
        last_run_date=(
            str(state_settings.get("operate_last_upstox_auto_renew_date"))
            if state_settings.get("operate_last_upstox_auto_renew_date")
            else None
        ),
        segment=str(
            state_settings.get("trading_calendar_segment", settings.trading_calendar_segment)
        ),
        settings=settings,
    )
    parsed_expiry = _to_utc_datetime(expires_at)
    expires_within_hours = None
    if parsed_expiry is not None:
        diff = parsed_expiry - _utc_now()
        expires_within_hours = round(diff.total_seconds() / 3600, 3)
    return {
        "enabled": enabled,
        "time_ist": str(
            state_settings.get("upstox_auto_renew_time_ist", settings.upstox_auto_renew_time_ist)
        ),
        "if_expires_within_hours": threshold_hours,
        "only_when_provider_enabled": bool(
            state_settings.get(
                "upstox_auto_renew_only_when_provider_enabled",
                settings.upstox_auto_renew_only_when_provider_enabled,
            )
        ),
        "last_run_date": state_settings.get("operate_last_upstox_auto_renew_date"),
        "next_scheduled_run_ist": next_run,
        "expires_within_hours": expires_within_hours,
    }
