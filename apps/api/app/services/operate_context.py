from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from app.db.models import DatasetBundle, PaperRun


def positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def bundle_exists(session: Session, bundle_id: int | None) -> bool:
    return (
        isinstance(bundle_id, int)
        and bundle_id > 0
        and session.get(DatasetBundle, bundle_id) is not None
    )


def resolve_active_bundle_id(
    session: Session,
    *,
    state_settings: dict[str, Any] | None = None,
    explicit_bundle_id: Any = None,
    latest_run: PaperRun | None = None,
    prefer_latest_run: bool = True,
) -> int | None:
    """Resolve the bundle used by operate flows.

    Order is explicit request, configured active bundle, latest paper run, newest bundle.
    The configured active bundle is the stable anchor for scheduled automation.
    """
    explicit_id = positive_int(explicit_bundle_id)
    if bundle_exists(session, explicit_id):
        return explicit_id

    active_id = positive_int((state_settings or {}).get("active_bundle_id"))
    if bundle_exists(session, active_id):
        return active_id

    if prefer_latest_run:
        run = latest_run
        if run is None:
            run = session.exec(select(PaperRun).order_by(PaperRun.created_at.desc())).first()
        run_bundle_id = positive_int(run.bundle_id if run is not None else None)
        if bundle_exists(session, run_bundle_id):
            return run_bundle_id

    latest_bundle = session.exec(
        select(DatasetBundle).order_by(DatasetBundle.created_at.desc())
    ).first()
    return positive_int(latest_bundle.id if latest_bundle is not None else None)


def latest_paper_run_for_bundle(session: Session, bundle_id: int | None) -> PaperRun | None:
    stmt = select(PaperRun)
    if isinstance(bundle_id, int) and bundle_id > 0:
        stmt = stmt.where(PaperRun.bundle_id == int(bundle_id))
    return session.exec(stmt.order_by(PaperRun.created_at.desc())).first()
