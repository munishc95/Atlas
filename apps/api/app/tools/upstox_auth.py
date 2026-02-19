from __future__ import annotations

import argparse
import json
import secrets

from sqlmodel import Session

from app.core.config import get_settings
from app.db.session import engine
from app.services.upstox_auth import (
    build_authorization_url,
    build_fake_access_token,
    get_provider_access_token,
    mark_verified_now,
    save_provider_credential,
    token_status,
    token_timestamps_from_payload,
    exchange_authorization_code,
    extract_access_token,
    mask_token,
    persist_access_token,
    resolve_client_id,
    resolve_client_secret,
    resolve_redirect_uri,
    verify_access_token,
)


def _cmd_auth_url(args: argparse.Namespace) -> int:
    settings = get_settings()
    client_id = resolve_client_id(settings)
    redirect_uri = resolve_redirect_uri(settings=settings, redirect_uri=args.redirect_uri)
    state = args.state or secrets.token_urlsafe(16)
    url = build_authorization_url(
        client_id=client_id,
        redirect_uri=redirect_uri,
        state=state,
        base_url=settings.upstox_base_url,
    )
    print(json.dumps({"auth_url": url, "state": state, "redirect_uri": redirect_uri}, indent=2))
    return 0


def _cmd_exchange(args: argparse.Namespace) -> int:
    settings = get_settings()
    client_id = resolve_client_id(settings)
    client_secret = resolve_client_secret(settings)
    redirect_uri = resolve_redirect_uri(settings=settings, redirect_uri=args.redirect_uri)
    if settings.fast_mode_enabled and str(args.code).strip() == str(settings.upstox_e2e_fake_code).strip():
        token = build_fake_access_token()
        response = {"status": "success", "data": {"access_token": token}}
    else:
        response = exchange_authorization_code(
            code=args.code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            base_url=settings.upstox_base_url,
            timeout_seconds=settings.upstox_timeout_seconds,
        )
    token = extract_access_token(response)
    issued_at, expires_at = token_timestamps_from_payload(token=token, payload=response)
    with Session(engine) as session:
        save_provider_credential(
            session,
            settings=settings,
            access_token=token,
            issued_at=issued_at,
            expires_at=expires_at,
        )
    persisted_paths: list[str] = []
    if (not args.no_save) and bool(settings.upstox_persist_env_fallback):
        persisted_paths = persist_access_token(access_token=token)
        get_settings.cache_clear()
    verification = verify_access_token(
        access_token=token,
        base_url=settings.upstox_base_url,
        timeout_seconds=settings.upstox_timeout_seconds,
    )
    if bool(verification.get("valid")):
        with Session(engine) as session:
            mark_verified_now(session, settings=settings)
    print(
        json.dumps(
            {
                "token_masked": mask_token(token),
                "persisted_paths": persisted_paths,
                "token_source": "encrypted_store",
                "verification": verification,
            },
            indent=2,
        )
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.token:
        token = args.token
    else:
        with Session(engine) as session:
            token = get_provider_access_token(session, settings=settings, allow_env_fallback=True)
    verification = verify_access_token(
        access_token=str(token or ""),
        base_url=settings.upstox_base_url,
        timeout_seconds=settings.upstox_timeout_seconds,
    )
    if bool(verification.get("valid")):
        with Session(engine) as session:
            mark_verified_now(session, settings=settings)
            status = token_status(session, settings=settings, allow_env_fallback=True)
    else:
        with Session(engine) as session:
            status = token_status(session, settings=settings, allow_env_fallback=True)
    print(
        json.dumps(
            {
                "token_masked": status.get("token_masked") or mask_token(str(token or "")),
                "expires_at": status.get("expires_at"),
                "last_verified_at": status.get("last_verified_at"),
                "verification": verification,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Upstox OAuth helper for Atlas.")
    sub = parser.add_subparsers(dest="command", required=True)

    auth_url = sub.add_parser("auth-url", help="Build Upstox authorization URL.")
    auth_url.add_argument("--redirect-uri", default=None, help="OAuth redirect URI override")
    auth_url.add_argument("--state", default=None, help="Optional OAuth state")
    auth_url.set_defaults(handler=_cmd_auth_url)

    exchange = sub.add_parser("exchange", help="Exchange authorization code for access token.")
    exchange.add_argument("--code", required=True, help="Authorization code returned by Upstox")
    exchange.add_argument("--redirect-uri", default=None, help="OAuth redirect URI override")
    exchange.add_argument(
        "--no-save",
        action="store_true",
        help="Do not persist token to .env files (only relevant when fallback enabled).",
    )
    exchange.set_defaults(handler=_cmd_exchange)

    verify = sub.add_parser("verify", help="Verify configured Upstox access token.")
    verify.add_argument("--token", default=None, help="Optional token override for verification")
    verify.set_defaults(handler=_cmd_verify)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
