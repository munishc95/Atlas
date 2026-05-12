from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import engine, init_db  # noqa: E402
from app.services.data_quality_gap_audit import generate_data_quality_gap_audit  # noqa: E402
from sqlmodel import Session  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Atlas daily data-quality gaps against NSE bhavcopy and membership history."
    )
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--report-id", type=int, default=None)
    parser.add_argument("--output-dir", default="data/reports/data-quality")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Build and summarize the audit without writing JSON/Markdown files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    init_db()
    with Session(engine) as session:
        payload = generate_data_quality_gap_audit(
            session=session,
            settings=settings,
            bundle_id=int(args.bundle_id),
            timeframe=str(args.timeframe),
            report_id=args.report_id,
            write_files=not bool(args.no_write),
            output_dir=args.output_dir,
        )

    summary = payload.get("summary", {})
    print(
        {
            "status": "ok",
            "report_id": payload.get("report", {}).get("id"),
            "bundle_id": payload.get("report", {}).get("bundle_id"),
            "timeframe": payload.get("report", {}).get("timeframe"),
            "symbols_audited": summary.get("symbols_audited"),
            "gap_events_total": summary.get("gap_events_total"),
            "missing_dates_total": summary.get("missing_dates_total"),
            "classification_counts": summary.get("classification_counts"),
            "backfillable_missing_dates": summary.get("backfillable_missing_dates"),
            "backfillable_symbols": summary.get("backfillable_symbols"),
            "files": payload.get("files", {}),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
