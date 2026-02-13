from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func
from sqlmodel import Session, select

from app.db.models import Job

TERMINAL_JOB_STATES = {"SUCCEEDED", "FAILED", "DONE"}


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda x: str(x[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, bytes):
        return {"__bytes__": hashlib.sha256(value).hexdigest()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def create_job(
    session: Session,
    job_type: str,
    *,
    idempotency_key: str | None = None,
    request_hash: str | None = None,
) -> Job:
    job = Job(
        id=uuid4().hex,
        type=job_type,
        status="QUEUED",
        progress=0,
        logs_json=[],
        idempotency_key=idempotency_key,
        request_hash=request_hash,
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def find_job_by_idempotency(
    session: Session,
    *,
    job_type: str,
    idempotency_key: str,
    request_hash: str,
) -> Job | None:
    statement = (
        select(Job)
        .where(Job.type == job_type)
        .where(Job.idempotency_key == idempotency_key)
        .where(Job.request_hash == request_hash)
        .order_by(Job.created_at.desc())
        .limit(1)
    )
    return session.exec(statement).first()


def append_job_log(session: Session, job_id: str, message: str) -> None:
    job = session.get(Job, job_id)
    if job is None:
        return
    logs = list(job.logs_json or [])
    logs.append(message)
    job.logs_json = logs
    session.add(job)
    session.commit()


def update_job(
    session: Session,
    job_id: str,
    *,
    status: str | None = None,
    progress: int | None = None,
    result: dict[str, object] | None = None,
) -> Job | None:
    job = session.get(Job, job_id)
    if job is None:
        return None

    if status is not None:
        job.status = status
        if status == "RUNNING" and job.started_at is None:
            job.started_at = _utc_now()
        if status in TERMINAL_JOB_STATES:
            job.ended_at = _utc_now()
    if progress is not None:
        job.progress = progress
    if result is not None:
        job.result_json = result

    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def get_job(session: Session, job_id: str) -> Job | None:
    return session.get(Job, job_id)


def list_recent_jobs(session: Session, page: int = 1, page_size: int = 20) -> tuple[list[Job], int]:
    total = int(session.exec(select(func.count()).select_from(Job)).one())
    offset = max(0, (page - 1) * page_size)
    statement = (
        select(Job).order_by(Job.created_at.desc(), Job.id.desc()).offset(offset).limit(page_size)
    )
    return list(session.exec(statement).all()), total


async def job_event_stream(session_factory, job_id: str, poll_seconds: float = 1.0):
    last_log_len = 0
    while True:
        with session_factory() as session:
            job = session.get(Job, job_id)
            if job is None:
                yield 'event: error\ndata: {"code":"not_found","message":"Job not found"}\n\n'
                return

            logs = job.logs_json or []
            for line in logs[last_log_len:]:
                yield f"event: log\ndata: {line}\n\n"
            last_log_len = len(logs)

            payload = {
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                "result": job.result_json,
            }
            yield f"event: progress\ndata: {json.dumps(payload)}\n\n"
            yield "event: heartbeat\ndata: {}\n\n"

            if job.status in TERMINAL_JOB_STATES:
                return

        await asyncio.sleep(poll_seconds)
