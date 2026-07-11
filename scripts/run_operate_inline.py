from __future__ import annotations

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from sqlmodel import Session

from app.core.config import get_settings
from app.db.bootstrap import seed_defaults
from app.db.models import Job, PaperState
from app.db.session import engine, init_db
from app.jobs.tasks import run_operate_run_job
from app.services.jobs import create_job
from app.services.operate_context import resolve_active_bundle_id

IST_ZONE = ZoneInfo("Asia/Kolkata")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Atlas operate pipeline inline without Redis/RQ."
    )
    parser.add_argument(
        "--bundle-id",
        type=int,
        default=None,
        help="Universe bundle id. Defaults to the configured active bundle, then latest bundle.",
    )
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--regime", default="TREND_UP")
    parser.add_argument("--date", default=None)
    parser.add_argument("--source", default="windows_scheduled_task")
    parser.add_argument("--max-runtime-seconds", type=int, default=10_800)
    parser.add_argument(
        "--include-data-updates",
        action="store_true",
        help="Also run provider/inbox data updates inside operate_run.",
    )
    parser.add_argument(
        "--shadow-only",
        action="store_true",
        help="Run operate in shadow mode without mutating live paper cash/positions/orders.",
    )
    parser.add_argument(
        "--mark-auto-run-date",
        action="store_true",
        help="Set operate_last_auto_run_date after a successful operate run.",
    )
    parser.add_argument(
        "--skip-if-auto-run-date-marked",
        action="store_true",
        help="Exit successfully without running if operate_last_auto_run_date already matches --date.",
    )
    return parser.parse_args()


def _auto_run_date_marked(session: Session, run_date: str) -> bool:
    state = session.get(PaperState, 1)
    if state is None:
        return False
    settings = dict(state.settings_json or {})
    return str(settings.get("operate_last_auto_run_date") or "") == str(run_date)


def _mark_auto_run_date(session: Session, run_date: str) -> None:
    state = session.get(PaperState, 1)
    if state is None:
        return
    settings = dict(state.settings_json or {})
    settings["operate_last_auto_run_date"] = run_date
    state.settings_json = settings
    session.add(state)
    session.commit()


def _resolve_required_bundle_id(session: Session, explicit_bundle_id: int | None) -> int:
    state = session.get(PaperState, 1)
    state_settings = dict(state.settings_json or {}) if state is not None else {}
    bundle_id = resolve_active_bundle_id(
        session,
        state_settings=state_settings,
        explicit_bundle_id=explicit_bundle_id,
    )
    if bundle_id is None:
        raise RuntimeError(
            "No dataset bundle found. Create or import a universe bundle before running operate."
        )
    return int(bundle_id)


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    init_db()
    with Session(engine) as session:
        seed_defaults(session, settings)

    now_ist = datetime.now(IST_ZONE)
    run_date = str(args.date or now_ist.date().isoformat())
    with Session(engine) as session:
        bundle_id = _resolve_required_bundle_id(session, args.bundle_id)
        if bool(args.skip_if_auto_run_date_marked) and _auto_run_date_marked(session, run_date):
            print(
                {
                    "event": "operate_inline_skipped",
                    "reason": "auto_run_date_already_marked",
                    "date": run_date,
                }
            )
            return 0
        job = create_job(session, "operate_run")
        job_id = str(job.id)
    payload = {
        "date": run_date,
        "bundle_id": bundle_id,
        "timeframe": str(args.timeframe),
        "regime": str(args.regime),
        "include_data_updates": bool(args.include_data_updates),
        "shadow_only": bool(args.shadow_only),
        "asof": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
    }
    print({"event": "operate_inline_started", "job_id": job_id, "payload": payload})

    run_operate_run_job(
        job_id,
        payload,
        max_runtime_seconds=max(1, int(args.max_runtime_seconds)),
    )

    with Session(engine) as session:
        job = session.get(Job, job_id)
        result = job.result_json if job is not None and isinstance(job.result_json, dict) else {}
        summary = result.get("summary", {}) if isinstance(result, dict) else {}
        status = str(job.status if job is not None else "MISSING")
        output = {
            "event": "operate_inline_finished",
            "job_id": job_id,
            "status": status,
            "bundle_id": summary.get("bundle_id"),
            "timeframe": summary.get("timeframe"),
            "quality_status": summary.get("quality_status"),
            "paper": summary.get("paper"),
            "forward_journal": summary.get("forward_journal"),
            "daily_report": summary.get("daily_report"),
        }
        print(output)
        if status != "SUCCEEDED":
            return 1
        if bool(args.mark_auto_run_date):
            _mark_auto_run_date(session, run_date)
            print({"event": "operate_last_auto_run_date_marked", "date": run_date})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
