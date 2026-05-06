from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlmodel import Session

from app.core.config import get_settings
from app.db.models import DatasetBundle, PaperRun
from app.db.session import engine, init_db
from app.jobs.tasks import _resolve_operate_context
from app.main import app
from app.services.paper import get_or_create_paper_state


def _seed_bundle_pair(session: Session) -> tuple[int, int]:
    active_bundle = DatasetBundle(
        name=f"operate-active-{uuid4().hex[:8]}",
        provider="test",
        symbols_json=["ACTIVE"],
        supported_timeframes_json=["1d"],
    )
    stale_bundle = DatasetBundle(
        name=f"operate-stale-{uuid4().hex[:8]}",
        provider="test",
        symbols_json=["STALE"],
        supported_timeframes_json=["1d"],
    )
    session.add(active_bundle)
    session.add(stale_bundle)
    session.commit()
    session.refresh(active_bundle)
    session.refresh(stale_bundle)
    assert active_bundle.id is not None
    assert stale_bundle.id is not None
    session.add(
        PaperRun(
            bundle_id=int(stale_bundle.id),
            asof_ts=datetime(2026, 5, 6, 10, 0, tzinfo=timezone.utc),
            regime="HIGH_VOL",
            summary_json={"timeframes": ["4h_ish"]},
        )
    )
    session.commit()
    return int(active_bundle.id), int(stale_bundle.id)


def test_operate_run_context_uses_configured_active_bundle() -> None:
    init_db()
    settings = get_settings()

    with Session(engine) as session:
        active_bundle_id, stale_bundle_id = _seed_bundle_pair(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {"active_bundle_id": active_bundle_id}
        session.add(state)
        session.commit()

        context = _resolve_operate_context(session=session, payload={}, settings=settings)

    assert context["bundle_id"] == active_bundle_id
    assert context["bundle_id"] != stale_bundle_id
    assert context["timeframe"] == "1d"
    assert context["regime"] == "TREND_UP"


def test_operate_status_reports_configured_active_bundle() -> None:
    init_db()
    settings = get_settings()

    with Session(engine) as session:
        active_bundle_id, stale_bundle_id = _seed_bundle_pair(session)
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {"active_bundle_id": active_bundle_id}
        session.add(state)
        session.commit()

    with TestClient(app) as client:
        response = client.get("/api/operate/status")

    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["active_bundle_id"] == active_bundle_id
    latest_run = payload.get("latest_run")
    if latest_run is not None:
        assert latest_run["bundle_id"] != stale_bundle_id
