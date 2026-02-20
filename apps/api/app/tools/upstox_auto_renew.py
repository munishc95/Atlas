from __future__ import annotations

import argparse
import json

from sqlmodel import Session

from app.core.config import get_settings
from app.db.session import engine, init_db
from app.services.paper import get_or_create_paper_state
from app.services.upstox_auth import token_status
from app.services.upstox_token_request import (
    auto_renew_meta,
    latest_request_run,
    recommended_notifier_endpoint,
    request_token_run,
    serialize_request_run,
)


def _status_command() -> int:
    settings = get_settings()
    with Session(engine) as session:
        state = get_or_create_paper_state(session, settings)
        status = token_status(session, settings=settings, allow_env_fallback=True)
        meta = auto_renew_meta(
            settings=settings,
            state_settings=dict(state.settings_json or {}),
            expires_at=str(status.get("expires_at") or "") or None,
        )
        latest = latest_request_run(session)
        print("Upstox token status:")
        print(json.dumps(status, indent=2))
        print("\nAuto-renew:")
        print(json.dumps(meta, indent=2))
        if latest is not None:
            print("\nLatest token request:")
            print(json.dumps(serialize_request_run(latest), indent=2))
            print("\nRecommended notifier URL:")
            print(recommended_notifier_endpoint(settings=settings, nonce=latest.correlation_nonce))
    return 0


def _request_command(source: str) -> int:
    settings = get_settings()
    with Session(engine) as session:
        run, deduped = request_token_run(
            session,
            settings=settings,
            correlation_id=None,
            source=source,
        )
        print("Created token request" if not deduped else "Reused pending token request")
        print(json.dumps(serialize_request_run(run), indent=2))
        print(
            "Recommended notifier URL:",
            recommended_notifier_endpoint(settings=settings, nonce=run.correlation_nonce),
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Upstox auto-renew helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("status", help="Show token status and recommended notifier URL")
    req = sub.add_parser("request", help="Request a new Upstox access token approval")
    req.add_argument("--source", default="cli", help="source tag for audit metadata")

    args = parser.parse_args()
    init_db()

    if args.cmd == "status":
        return _status_command()
    if args.cmd == "request":
        return _request_command(source=str(args.source))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
