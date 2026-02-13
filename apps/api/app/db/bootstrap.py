from __future__ import annotations

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import PaperState, Strategy, Symbol


def seed_defaults(session: Session, settings: Settings) -> None:
    if session.get(PaperState, 1) is None:
        session.add(
            PaperState(
                id=1,
                equity=1_000_000.0,
                cash=1_000_000.0,
                peak_equity=1_000_000.0,
                drawdown=0.0,
                kill_switch_active=False,
                cooldown_days_left=0,
                settings_json={
                    "risk_per_trade": settings.risk_per_trade,
                    "max_positions": settings.max_positions,
                    "kill_switch_dd": settings.kill_switch_drawdown,
                    "max_position_value_pct_adv": settings.max_position_value_pct_adv,
                    "diversification_corr_threshold": settings.diversification_corr_threshold,
                    "paper_mode": "strategy",
                    "active_policy_id": None,
                },
            )
        )

    if session.exec(select(Symbol).where(Symbol.symbol == "NIFTY500")).first() is None:
        session.add(Symbol(symbol="NIFTY500", name="NIFTY 500 Index Proxy", sector="INDEX"))

    if session.exec(select(Strategy)).first() is None:
        session.add(
            Strategy(
                name="Trend Breakout Default",
                template="trend_breakout",
                params_json={"trend_period": 200, "breakout_lookback": 20},
                enabled=True,
            )
        )

    session.commit()
