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
                    "reports_auto_generate_daily": settings.reports_auto_generate_daily,
                    "health_window_days_short": settings.health_window_days_short,
                    "health_window_days_long": settings.health_window_days_long,
                    "drift_maxdd_multiplier": settings.drift_maxdd_multiplier,
                    "drift_negative_return_cost_ratio_threshold": settings.drift_negative_return_cost_ratio_threshold,
                    "drift_win_rate_drop_pct": settings.drift_win_rate_drop_pct,
                    "drift_return_delta_threshold": settings.drift_return_delta_threshold,
                    "drift_warning_risk_scale": settings.drift_warning_risk_scale,
                    "drift_degraded_risk_scale": settings.drift_degraded_risk_scale,
                    "drift_degraded_action": settings.drift_degraded_action,
                    "evaluations_auto_promote_enabled": settings.evaluations_auto_promote_enabled,
                    "evaluations_min_window_days": settings.evaluations_min_window_days,
                    "evaluations_score_margin": settings.evaluations_score_margin,
                    "evaluations_max_dd_multiplier": settings.evaluations_max_dd_multiplier,
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
                    "futures_initial_margin_pct": settings.futures_initial_margin_pct,
                    "futures_symbol_mapping_strategy": settings.futures_symbol_mapping_strategy,
                    "paper_use_simulator_engine": settings.paper_use_simulator_engine,
                    "trading_calendar_segment": settings.trading_calendar_segment,
                    "operate_safe_mode_on_fail": settings.operate_safe_mode_on_fail,
                    "operate_safe_mode_action": settings.operate_safe_mode_action,
                    "operate_mode": settings.operate_mode,
                    "data_quality_stale_severity": (
                        "FAIL"
                        if str(settings.operate_mode).strip().lower() == "live"
                        else settings.data_quality_stale_severity
                    ),
                    "data_quality_stale_severity_override": False,
                    "data_quality_max_stale_minutes_1d": settings.data_quality_max_stale_minutes_1d,
                    "data_quality_max_stale_minutes_intraday": settings.data_quality_max_stale_minutes_intraday,
                    "operate_auto_run_enabled": settings.operate_auto_run_enabled,
                    "operate_auto_run_time_ist": settings.operate_auto_run_time_ist,
                    "operate_auto_run_include_data_updates": settings.operate_auto_run_include_data_updates,
                    "operate_last_auto_run_date": None,
                    "operate_max_stale_minutes_1d": settings.operate_max_stale_minutes_1d,
                    "operate_max_stale_minutes_4h_ish": settings.operate_max_stale_minutes_4h_ish,
                    "operate_max_gap_bars": settings.operate_max_gap_bars,
                    "operate_outlier_zscore": settings.operate_outlier_zscore,
                    "operate_cost_ratio_spike_threshold": settings.operate_cost_ratio_spike_threshold,
                    "operate_cost_ratio_spike_days": settings.operate_cost_ratio_spike_days,
                    "operate_cost_spike_risk_scale": settings.operate_cost_spike_risk_scale,
                    "operate_scan_truncated_warn_days": settings.operate_scan_truncated_warn_days,
                    "operate_scan_truncated_reduce_to": settings.operate_scan_truncated_reduce_to,
                    "data_updates_inbox_enabled": settings.data_updates_inbox_enabled,
                    "data_updates_max_files_per_run": settings.data_updates_max_files_per_run,
                    "coverage_missing_latest_warn_pct": settings.coverage_missing_latest_warn_pct,
                    "coverage_missing_latest_fail_pct": settings.coverage_missing_latest_fail_pct,
                    "coverage_inactive_after_missing_days": settings.coverage_inactive_after_missing_days,
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
