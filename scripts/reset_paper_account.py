from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.models import (  # noqa: E402
    AuditLog,
    ForwardSignalJournal,
    PaperOrder,
    PaperPosition,
    PaperState,
    ShadowPaperState,
)
from app.db.session import engine, init_db  # noqa: E402
from sqlmodel import Session, select  # noqa: E402


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Reset Atlas paper-trading account equity and clear simulated open state."
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=float(settings.paper_starting_equity),
        help="New paper account equity and cash balance.",
    )
    parser.add_argument(
        "--keep-shadow-state",
        action="store_true",
        help="Keep shadow paper state rows instead of clearing them.",
    )
    parser.add_argument(
        "--keep-open-state",
        action="store_true",
        help="Keep paper orders and positions instead of clearing them.",
    )
    parser.add_argument(
        "--skip-forward-journal-reprice",
        action="store_true",
        help="Do not recalculate forward-journal planned quantities from the new equity.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = get_settings()
    equity = max(0.0, float(args.equity))
    init_db()
    with Session(engine) as session:
        state = session.get(PaperState, 1)
        if state is None:
            state = PaperState(id=1)
        state.equity = equity
        state.cash = equity
        state.peak_equity = equity
        state.drawdown = 0.0
        state.kill_switch_active = False
        state.cooldown_days_left = 0
        settings_json = dict(state.settings_json or {})
        settings_json["paper_starting_equity"] = equity
        state.settings_json = settings_json
        session.add(state)

        risk_per_trade = max(0.0, float(settings_json.get("risk_per_trade", settings.risk_per_trade)))
        risk_budget = equity * risk_per_trade
        deleted_positions = 0
        deleted_orders = 0
        deleted_shadow_states = 0
        repriced_journal_rows = 0
        if not bool(args.keep_open_state):
            for row in session.exec(select(PaperPosition)).all():
                session.delete(row)
                deleted_positions += 1
            for row in session.exec(select(PaperOrder)).all():
                session.delete(row)
                deleted_orders += 1
        if not bool(args.keep_shadow_state):
            for row in session.exec(select(ShadowPaperState)).all():
                session.delete(row)
                deleted_shadow_states += 1
        if not bool(args.skip_forward_journal_reprice):
            for row in session.exec(select(ForwardSignalJournal)).all():
                risk_per_share = float(row.risk_per_share or 0.0)
                if risk_per_share <= 0 and row.entry_price > 0 and row.stop_price > 0:
                    risk_per_share = abs(float(row.entry_price) - float(row.stop_price))
                qty = int(math.floor(risk_budget / risk_per_share)) if risk_per_share > 0 else 0
                planned_risk = float(qty * risk_per_share) if risk_per_share > 0 else 0.0
                planned_value = float(qty * row.entry_price) if row.entry_price > 0 else 0.0
                if (
                    int(row.planned_qty or 0) == qty
                    and abs(float(row.planned_risk_amount or 0.0) - planned_risk) < 1e-9
                    and abs(float(row.planned_position_value or 0.0) - planned_value) < 1e-9
                ):
                    continue
                row.planned_qty = qty
                row.planned_risk_amount = planned_risk
                row.planned_position_value = planned_value
                signal_json = dict(row.signal_json or {})
                signal_json["risk_budget"] = float(risk_budget)
                signal_json["risk_per_share"] = float(risk_per_share)
                signal_json["planned_qty"] = qty
                signal_json["planned_risk_amount"] = planned_risk
                signal_json["planned_position_value"] = planned_value
                signal_json["position_size_status"] = (
                    "OK" if qty > 0 else "ZERO_QTY_RISK_CAP"
                )
                row.signal_json = signal_json
                session.add(row)
                repriced_journal_rows += 1

        session.add(
            AuditLog(
                type="paper_account_reset",
                payload_json={
                    "equity": equity,
                    "cash": equity,
                    "deleted_positions": deleted_positions,
                    "deleted_orders": deleted_orders,
                    "deleted_shadow_states": deleted_shadow_states,
                    "repriced_journal_rows": repriced_journal_rows,
                },
            )
        )
        session.commit()

    print(
        {
            "status": "ok",
            "equity": equity,
            "cash": equity,
            "deleted_positions": deleted_positions,
            "deleted_orders": deleted_orders,
            "deleted_shadow_states": deleted_shadow_states,
            "repriced_journal_rows": repriced_journal_rows,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
