from __future__ import annotations

import hashlib
import json
import secrets
from datetime import UTC, datetime, time as dt_time, timedelta
from typing import Any
from urllib import error, request
from zoneinfo import ZoneInfo

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import UpstoxTokenRequestRun
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
STATUS_REQUESTED = "REQUESTED"
STATUS_APPROVED = "APPROVED"
STATUS_REJECTED = "REJECTED"
STATUS_EXPIRED = "EXPIRED"
STATUS_FAILED = "FAILED"
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


def _notifier_base_url(settings: Settings) -> str:
    base = str(settings.upstox_notifier_base_url or "").strip()
    if base:
        return base.rstrip("/")
    return "http://127.0.0.1:8000"


def recommended_notifier_endpoint(*, settings: Settings, nonce: str) -> str:
    token = str(nonce or "").strip()
    return f"{_notifier_base_url(settings)}/api/providers/upstox/notifier?nonce={token}"


def _extract_response_data(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


def _parse_authorization_expiry(payload: dict[str, Any]) -> datetime | None:
    data = _extract_response_data(payload)
    for key in (
        "authorization_expiry",
        "authorizationExpiry",
        "expires_at",
        "expiry",
        "expiresAt",
    ):
        value = data.get(key)
        parsed = _to_utc_datetime(value)
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


def serialize_request_run(row: UpstoxTokenRequestRun) -> dict[str, Any]:
    return {
        "id": row.id,
        "provider_kind": row.provider_kind,
        "status": row.status,
        "requested_at": row.requested_at.isoformat() if row.requested_at is not None else None,
        "authorization_expiry": (
            row.authorization_expiry.isoformat() if row.authorization_expiry is not None else None
        ),
        "approved_at": row.approved_at.isoformat() if row.approved_at is not None else None,
        "notifier_url": row.notifier_url,
        "client_id": row.client_id,
        "user_id": row.user_id,
        "correlation_nonce": row.correlation_nonce,
        "last_error": row.last_error,
        "created_at": row.created_at.isoformat() if row.created_at is not None else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at is not None else None,
    }


def _latest_pending_run(session: Session) -> UpstoxTokenRequestRun | None:
    return session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == PROVIDER_KIND_UPSTOX)
        .where(UpstoxTokenRequestRun.status == STATUS_REQUESTED)
        .order_by(UpstoxTokenRequestRun.requested_at.desc(), UpstoxTokenRequestRun.id.desc())
    ).first()


def _expire_stale_runs(session: Session, *, now: datetime) -> None:
    rows = session.exec(
        select(UpstoxTokenRequestRun)
        .where(UpstoxTokenRequestRun.provider_kind == PROVIDER_KIND_UPSTOX)
        .where(UpstoxTokenRequestRun.status == STATUS_REQUESTED)
    ).all()
    changed = False
    for row in rows:
        expiry = _to_utc_datetime(row.authorization_expiry)
        if expiry is not None and expiry <= now:
            row.status = STATUS_EXPIRED
            row.updated_at = now
            session.add(row)
            changed = True
    if changed:
        session.commit()


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


def request_token_run(
    session: Session,
    *,
    settings: Settings,
    correlation_id: str | None = None,
    source: str = "manual",
) -> tuple[UpstoxTokenRequestRun, bool]:
    now = _utc_now()
    _expire_stale_runs(session, now=now)
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
        status=STATUS_REQUESTED,
        requested_at=now,
        authorization_expiry=None,
        approved_at=None,
        notifier_url=recommended_notifier_endpoint(settings=settings, nonce=nonce),
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
        expiry = now + timedelta(hours=1)
        run.authorization_expiry = expiry
        run.notifier_url = recommended_notifier_endpoint(
            settings=settings, nonce=run.correlation_nonce
        )
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
            "access_token": build_fake_access_token(),
            "token_type": "Bearer",
            "issued_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=12)).isoformat(),
            "message_type": "TOKEN_ISSUED",
        }
        process_notifier_payload(
            session,
            settings=settings,
            payload=fake_payload,
            nonce=run.correlation_nonce,
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
        expiry = _parse_authorization_expiry(raw) or _default_authorization_expiry(now)
        run.authorization_expiry = expiry
        run.notifier_url = _parse_notifier_url(raw) or run.notifier_url
        run.user_id = _parse_user_id(raw) or run.user_id
        run.status = STATUS_REQUESTED
        run.last_error = None
        run.updated_at = _utc_now()
        run.metadata_json = {
            **(run.metadata_json or {}),
            "response_status": raw.get("status") if isinstance(raw, dict) else None,
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
        details: dict[str, Any] = {
            "run_id": run.id,
            "reason": str(exc),
            "source": source,
        }
        if isinstance(exc, error.HTTPError):
            try:
                payload = exc.read().decode("utf-8", errors="replace")
                details["http_status"] = int(exc.code)
                details["raw"] = payload[:500]
            except Exception:  # noqa: BLE001
                details["http_status"] = int(exc.code)
        run.status = STATUS_FAILED
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
        .where(UpstoxTokenRequestRun.status == STATUS_REQUESTED)
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


def process_notifier_payload(
    session: Session,
    *,
    settings: Settings,
    payload: dict[str, Any] | None,
    nonce: str | None,
    correlation_id: str | None = None,
    source: str = "webhook",
) -> dict[str, Any]:
    body = payload if isinstance(payload, dict) else {}
    client_id = str(body.get("client_id") or "").strip() or None
    user_id = str(body.get("user_id") or "").strip() or None
    access_token = str(body.get("access_token") or "").strip()
    incoming_nonce = str(nonce or "").strip()

    run = _match_pending_run(session, client_id=client_id)
    if run is None:
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_unmatched",
            details={"source": source, "client_id": client_id},
            correlation_id=correlation_id,
            commit=True,
        )
        return {"matched": False, "accepted": False, "reason": "unmatched"}

    if incoming_nonce != str(run.correlation_nonce or "").strip():
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_nonce_mismatch",
            details={"run_id": run.id, "source": source},
            correlation_id=correlation_id,
            commit=True,
        )
        return {"matched": True, "accepted": False, "reason": "nonce_mismatch", "run_id": run.id}

    if not access_token:
        emit_operate_event(
            session,
            severity="WARN",
            category="SYSTEM",
            message="upstox_notifier_missing_token",
            details={"run_id": run.id, "source": source},
            correlation_id=correlation_id,
            commit=True,
        )
        return {
            "matched": True,
            "accepted": False,
            "reason": "access_token_missing",
            "run_id": run.id,
        }

    issued_at = _to_utc_datetime(body.get("issued_at")) or _utc_now()
    expires_at = _to_utc_datetime(body.get("expires_at")) or _default_token_expiry(issued_at)
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

    run.status = STATUS_APPROVED
    run.approved_at = _utc_now()
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
            "user_id": row.user_id,
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
