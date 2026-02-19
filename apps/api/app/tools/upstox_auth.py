from __future__ import annotations

import argparse
import json
import secrets

from app.core.config import get_settings
from app.services.upstox_auth import (
    build_authorization_url,
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
    response = exchange_authorization_code(
        code=args.code,
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        base_url=settings.upstox_base_url,
        timeout_seconds=settings.upstox_timeout_seconds,
    )
    token = extract_access_token(response)
    persisted_paths: list[str] = []
    if not args.no_save:
        persisted_paths = persist_access_token(access_token=token)
        get_settings.cache_clear()
    verification = verify_access_token(
        access_token=token,
        base_url=settings.upstox_base_url,
        timeout_seconds=settings.upstox_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "token_masked": mask_token(token),
                "persisted_paths": persisted_paths,
                "verification": verification,
            },
            indent=2,
        )
    )
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    settings = get_settings()
    token = args.token or settings.upstox_access_token
    verification = verify_access_token(
        access_token=str(token or ""),
        base_url=settings.upstox_base_url,
        timeout_seconds=settings.upstox_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "token_masked": mask_token(str(token or "")),
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
        help="Do not persist token to .env files.",
    )
    exchange.set_defaults(handler=_cmd_exchange)

    verify = sub.add_parser("verify", help="Verify configured Upstox access token.")
    verify.add_argument("--token", default=None, help="Optional token override for verification")
    verify.set_defaults(handler=_cmd_verify)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
