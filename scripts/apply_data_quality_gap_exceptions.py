from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import engine, init_db  # noqa: E402
from app.services.data_quality_exceptions import (  # noqa: E402
    upsert_no_trade_exceptions_from_gap_audit,
)
from app.services.data_quality_gap_audit import generate_data_quality_gap_audit  # noqa: E402
from sqlmodel import Session  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create active no-trade data-quality exceptions from a gap audit."
    )
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--timeframe", default="1d")
    parser.add_argument("--report-id", type=int, default=None)
    parser.add_argument("--output-dir", default="data/reports/data-quality")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-write-audit",
        action="store_true",
        help="Do not write the audit JSON/Markdown while applying exceptions.",
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
            write_files=not bool(args.no_write_audit),
            output_dir=args.output_dir,
        )
        result = upsert_no_trade_exceptions_from_gap_audit(
            session,
            payload=payload,
            dry_run=bool(args.dry_run),
        )
        result["audit_summary"] = payload.get("summary", {})
        result["audit_files"] = payload.get("files", {})

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
