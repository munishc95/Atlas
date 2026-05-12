from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import engine, init_db  # noqa: E402
from app.services.data_quality_remediation import (  # noqa: E402
    generate_data_quality_remediation_report,
)
from sqlmodel import Session  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a remediation report from the latest Atlas data-quality report."
    )
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--report-id", type=int, default=None)
    parser.add_argument("--output-dir", default="data/reports/data-quality")
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Build and summarize the report without writing JSON/Markdown files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    init_db()
    with Session(engine) as session:
        payload = generate_data_quality_remediation_report(
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
            "symbols_with_issues": summary.get("symbols_with_issues"),
            "blocked_by_symbol_gate_count": summary.get("blocked_by_symbol_gate_count"),
            "priority_counts": summary.get("priority_counts"),
            "action_counts": summary.get("action_counts"),
            "files": payload.get("files", {}),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
