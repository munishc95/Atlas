from __future__ import annotations

from typing import Any

from sqlmodel import Session, select

from app.core.exceptions import APIError
from app.db.models import (
    DatasetBundle,
    Policy,
    PolicyEnsemble,
    PolicyEnsembleMember,
)


def _normalize_weight(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _normalize_enabled(value: Any, *, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _policy_lookup(session: Session, policy_ids: list[int]) -> dict[int, Policy]:
    valid_ids = sorted({int(item) for item in policy_ids if int(item) > 0})
    if not valid_ids:
        return {}
    rows = session.exec(select(Policy).where(Policy.id.in_(valid_ids))).all()
    return {int(row.id): row for row in rows if row.id is not None}


def list_policy_ensemble_members(
    session: Session,
    *,
    ensemble_id: int,
    enabled_only: bool = False,
) -> list[dict[str, Any]]:
    stmt = (
        select(PolicyEnsembleMember)
        .where(PolicyEnsembleMember.ensemble_id == int(ensemble_id))
        .order_by(
            PolicyEnsembleMember.policy_id.asc(),
            PolicyEnsembleMember.id.asc(),
        )
    )
    if enabled_only:
        stmt = stmt.where(PolicyEnsembleMember.enabled == True)  # noqa: E712
    rows = list(session.exec(stmt).all())
    lookup = _policy_lookup(session, [int(row.policy_id) for row in rows])
    output: list[dict[str, Any]] = []
    for row in rows:
        policy = lookup.get(int(row.policy_id))
        output.append(
            {
                "id": int(row.id) if row.id is not None else None,
                "ensemble_id": int(row.ensemble_id),
                "policy_id": int(row.policy_id),
                "policy_name": policy.name if policy is not None else None,
                "weight": float(row.weight),
                "enabled": bool(row.enabled),
                "created_at": row.created_at.isoformat(),
            }
        )
    return output


def serialize_policy_ensemble(
    session: Session,
    ensemble: PolicyEnsemble,
    *,
    include_members: bool = True,
) -> dict[str, Any]:
    payload = {
        "id": int(ensemble.id) if ensemble.id is not None else None,
        "name": ensemble.name,
        "bundle_id": int(ensemble.bundle_id),
        "is_active": bool(ensemble.is_active),
        "created_at": ensemble.created_at.isoformat(),
    }
    if include_members:
        payload["members"] = list_policy_ensemble_members(
            session,
            ensemble_id=int(ensemble.id or 0),
            enabled_only=False,
        )
    return payload


def create_policy_ensemble(
    session: Session,
    *,
    name: str,
    bundle_id: int,
    is_active: bool = False,
) -> PolicyEnsemble:
    bundle = session.get(DatasetBundle, int(bundle_id))
    if bundle is None:
        raise APIError(code="not_found", message="Bundle not found", status_code=404)
    ensemble = PolicyEnsemble(
        name=str(name).strip()[:128],
        bundle_id=int(bundle_id),
        is_active=bool(is_active),
    )
    if not ensemble.name:
        raise APIError(code="invalid_payload", message="Ensemble name is required")
    session.add(ensemble)
    session.commit()
    session.refresh(ensemble)
    if is_active:
        set_active_policy_ensemble(session, ensemble_id=int(ensemble.id))
        session.refresh(ensemble)
    return ensemble


def list_policy_ensembles(
    session: Session,
    *,
    bundle_id: int | None = None,
) -> list[PolicyEnsemble]:
    stmt = select(PolicyEnsemble).order_by(
        PolicyEnsemble.created_at.desc(),
        PolicyEnsemble.id.desc(),
    )
    if bundle_id is not None:
        stmt = stmt.where(PolicyEnsemble.bundle_id == int(bundle_id))
    return list(session.exec(stmt).all())


def get_policy_ensemble(session: Session, ensemble_id: int) -> PolicyEnsemble:
    row = session.get(PolicyEnsemble, int(ensemble_id))
    if row is None:
        raise APIError(code="not_found", message="Ensemble not found", status_code=404)
    return row


def upsert_policy_ensemble_members(
    session: Session,
    *,
    ensemble_id: int,
    members: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ensemble = get_policy_ensemble(session, ensemble_id)
    existing_rows = list(
        session.exec(
            select(PolicyEnsembleMember).where(
                PolicyEnsembleMember.ensemble_id == int(ensemble.id or 0)
            )
        ).all()
    )
    existing_by_policy = {int(row.policy_id): row for row in existing_rows}

    requested_policy_ids = sorted(
        {
            int(item.get("policy_id"))
            for item in members
            if isinstance(item, dict) and int(item.get("policy_id") or 0) > 0
        }
    )
    lookup = _policy_lookup(session, requested_policy_ids)
    missing = [item for item in requested_policy_ids if item not in lookup]
    if missing:
        raise APIError(
            code="invalid_payload",
            message="One or more policies do not exist.",
            details={"missing_policy_ids": missing},
        )

    for item in members:
        if not isinstance(item, dict):
            continue
        policy_id = int(item.get("policy_id") or 0)
        if policy_id <= 0:
            continue
        row = existing_by_policy.get(policy_id)
        if row is None:
            row = PolicyEnsembleMember(
                ensemble_id=int(ensemble.id or 0),
                policy_id=policy_id,
                weight=_normalize_weight(item.get("weight", 0.0)),
                enabled=_normalize_enabled(item.get("enabled"), default=True),
            )
        else:
            row.weight = _normalize_weight(item.get("weight", row.weight))
            row.enabled = _normalize_enabled(item.get("enabled"), default=bool(row.enabled))
        session.add(row)
    session.commit()
    return list_policy_ensemble_members(
        session,
        ensemble_id=int(ensemble.id or 0),
        enabled_only=False,
    )


def set_active_policy_ensemble(session: Session, *, ensemble_id: int) -> PolicyEnsemble:
    target = get_policy_ensemble(session, ensemble_id)
    rows = list(
        session.exec(
            select(PolicyEnsemble).where(PolicyEnsemble.bundle_id == int(target.bundle_id))
        ).all()
    )
    for row in rows:
        row.is_active = bool(row.id == target.id)
        session.add(row)
    session.commit()
    session.refresh(target)
    return target


def get_active_policy_ensemble(
    session: Session,
    *,
    bundle_id: int | None,
    preferred_ensemble_id: int | None = None,
) -> PolicyEnsemble | None:
    if preferred_ensemble_id is not None and preferred_ensemble_id > 0:
        row = session.get(PolicyEnsemble, int(preferred_ensemble_id))
        if row is not None and (bundle_id is None or int(row.bundle_id) == int(bundle_id)):
            return row
    if bundle_id is None:
        return None
    return session.exec(
        select(PolicyEnsemble)
        .where(PolicyEnsemble.bundle_id == int(bundle_id))
        .where(PolicyEnsemble.is_active == True)  # noqa: E712
        .order_by(PolicyEnsemble.created_at.desc(), PolicyEnsemble.id.desc())
    ).first()

