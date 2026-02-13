from __future__ import annotations

from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import DatasetBundle, Instrument, PaperState, Strategy, Symbol


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
                    "allowed_sides": settings.allowed_sides,
                    "paper_short_squareoff_time": settings.paper_short_squareoff_time,
                    "autopilot_max_symbols_scan": settings.autopilot_max_symbols_scan,
                    "autopilot_max_runtime_seconds": settings.autopilot_max_runtime_seconds,
                    "cost_model_enabled": settings.cost_model_enabled,
                    "cost_mode": settings.cost_mode,
                    "brokerage_bps": settings.brokerage_bps,
                    "stt_delivery_buy_bps": settings.stt_delivery_buy_bps,
                    "stt_delivery_sell_bps": settings.stt_delivery_sell_bps,
                    "stt_intraday_buy_bps": settings.stt_intraday_buy_bps,
                    "stt_intraday_sell_bps": settings.stt_intraday_sell_bps,
                    "exchange_txn_bps": settings.exchange_txn_bps,
                    "sebi_bps": settings.sebi_bps,
                    "stamp_delivery_buy_bps": settings.stamp_delivery_buy_bps,
                    "stamp_intraday_buy_bps": settings.stamp_intraday_buy_bps,
                    "gst_rate": settings.gst_rate,
                    "futures_brokerage_bps": settings.futures_brokerage_bps,
                    "futures_stt_sell_bps": settings.futures_stt_sell_bps,
                    "futures_exchange_txn_bps": settings.futures_exchange_txn_bps,
                    "futures_stamp_buy_bps": settings.futures_stamp_buy_bps,
                    "paper_mode": "strategy",
                    "active_policy_id": None,
                },
            )
        )

    if session.exec(select(Symbol).where(Symbol.symbol == "NIFTY500")).first() is None:
        session.add(Symbol(symbol="NIFTY500", name="NIFTY 500 Index Proxy", sector="INDEX"))
    if (
        session.exec(
            select(Instrument)
            .where(Instrument.symbol == "NIFTY500")
            .where(Instrument.kind == "EQUITY_CASH")
        ).first()
        is None
    ):
        session.add(
            Instrument(
                symbol="NIFTY500",
                kind="EQUITY_CASH",
                underlying="NIFTY500",
                lot_size=1,
                tick_size=0.05,
            )
        )

    if (
        session.exec(select(DatasetBundle).where(DatasetBundle.name == "Default Universe")).first()
        is None
    ):
        session.add(
            DatasetBundle(
                name="Default Universe",
                provider="local",
                description="Compatibility default bundle for legacy imports.",
                symbols_json=["NIFTY500"],
                supported_timeframes_json=["1d", "4h_ish"],
            )
        )

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
