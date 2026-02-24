from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_prefix="ATLAS_", env_file=".env", extra="ignore")

    app_name: str = "Atlas API"
    environment: str = "local"
    database_url: str = "sqlite:///./apps/api/.atlas/atlas.db"
    duckdb_path: str = "apps/api/.atlas/ohlcv.duckdb"
    parquet_root: str = "data/parquet"
    feature_cache_root: str = "data/features"
    calendar_data_root: str = "data/calendars"
    data_inbox_root: str = "data/inbox"
    sample_data_root: str = "data/sample"
    secrets_root: str = "data/secrets"
    redis_url: str = "redis://localhost:6379/0"
    rq_queue_name: str = "atlas"
    jobs_inline: bool = False
    job_default_timeout_seconds: int = 10_800
    job_retry_max: int = 2
    job_retry_backoff_seconds: int = 5
    fast_mode: bool = False
    e2e_fast: bool = False
    fast_mode_max_symbols_scan: int = 10
    fast_mode_max_optuna_trials: int = 5
    fast_mode_seed: int = 7
    fast_mode_job_timeout_seconds: int = 1_200
    fast_mode_job_poll_seconds: float = 0.25
    fast_mode_provider_intraday_max_symbols: int = 3
    fast_mode_provider_intraday_max_days: int = 2

    risk_per_trade: float = 0.005
    max_positions: int = 3
    kill_switch_drawdown: float = 0.08
    kill_switch_cooldown_days: int = 10

    commission_bps: float = 5.0
    slippage_base_bps: float = 2.0
    slippage_vol_factor: float = 15.0
    cost_model_enabled: bool = False
    cost_mode: str = "delivery"
    brokerage_bps: float = 0.0
    stt_delivery_buy_bps: float = 0.0
    stt_delivery_sell_bps: float = 10.0
    stt_intraday_buy_bps: float = 0.0
    stt_intraday_sell_bps: float = 2.5
    exchange_txn_bps: float = 0.297
    sebi_bps: float = 0.001
    stamp_delivery_buy_bps: float = 1.5
    stamp_intraday_buy_bps: float = 0.3
    gst_rate: float = 0.18
    futures_brokerage_bps: float = 0.0
    futures_stt_sell_bps: float = 1.0
    futures_exchange_txn_bps: float = 0.19
    futures_stamp_buy_bps: float = 0.0
    futures_initial_margin_pct: float = 0.18
    futures_symbol_mapping_strategy: str = "underlying_lookup"
    paper_use_simulator_engine: bool = True
    trading_calendar_segment: str = "EQUITIES"
    nse_equities_open_time_ist: str = "09:15"
    nse_equities_close_time_ist: str = "15:30"
    operate_safe_mode_on_fail: bool = True
    operate_safe_mode_action: str = "exits_only"
    operate_mode: str = "offline"
    data_quality_stale_severity: str = "WARN"
    data_quality_max_stale_minutes_1d: int = 1_440
    data_quality_max_stale_minutes_intraday: int = 60
    operate_auto_run_enabled: bool = False
    operate_auto_run_time_ist: str = "15:35"
    operate_auto_run_include_data_updates: bool = True
    operate_auto_eval_enabled: bool = True
    operate_auto_eval_frequency: str = "WEEKLY"
    operate_auto_eval_day_of_week: int = 0
    operate_auto_eval_time_ist: str = "16:00"
    operate_auto_eval_lookback_trading_days: int = 20
    operate_auto_eval_min_trades: int = 8
    operate_auto_eval_cooldown_trading_days: int = 10
    operate_auto_eval_max_switches_per_30d: int = 2
    operate_auto_eval_auto_switch: bool = False
    operate_auto_eval_shadow_only_gate: bool = True
    operate_max_stale_minutes_1d: int = 2880
    operate_max_stale_minutes_4h_ish: int = 720
    operate_max_gap_bars: int = 3
    operate_outlier_zscore: float = 8.0
    operate_cost_ratio_spike_threshold: float = 0.5
    operate_cost_ratio_spike_days: int = 3
    operate_cost_spike_risk_scale: float = 0.5
    operate_scan_truncated_warn_days: int = 3
    operate_scan_truncated_reduce_to: int = 80
    data_updates_inbox_enabled: bool = True
    data_updates_max_files_per_run: int = 50
    data_updates_provider_enabled: bool = False
    data_updates_provider_mode: str = "SINGLE"
    data_updates_provider_kind: str = "UPSTOX"
    data_updates_provider_priority_order: list[str] = Field(
        default_factory=lambda: ["UPSTOX", "NSE_EOD", "INBOX"]
    )
    data_updates_provider_nse_eod_enabled: bool = True
    data_updates_provider_max_symbols_per_run: int = 120
    data_updates_provider_max_calls_per_run: int = 400
    data_updates_provider_timeframe_enabled: str = "1d"
    data_updates_provider_timeframes: list[str] = Field(default_factory=lambda: ["1d"])
    data_updates_provider_repair_last_n_trading_days: int = 3
    data_updates_provider_backfill_max_days: int = 30
    data_updates_provider_allow_partial_4h_ish: bool = False
    nse_eod_base_url: str = "https://www.nseindia.com"
    nse_eod_timeout_seconds: float = 12.0
    nse_eod_retry_max: int = 2
    nse_eod_retry_backoff_seconds: float = 0.8
    nse_eod_throttle_seconds: float = 0.8
    data_provenance_confidence_upstox: int = 95
    data_provenance_confidence_nse_eod: int = 80
    data_provenance_confidence_inbox: int = 70
    data_quality_confidence_fail_threshold: int = 65
    coverage_missing_latest_warn_pct: float = 10.0
    coverage_missing_latest_fail_pct: float = 25.0
    coverage_inactive_after_missing_days: int = 3
    max_position_value_pct_adv: float = 0.01
    diversification_corr_threshold: float = 0.75
    allowed_sides: list[str] = Field(default_factory=lambda: ["BUY"])
    paper_short_squareoff_time: str = "15:20"
    autopilot_max_symbols_scan: int = 200
    autopilot_max_runtime_seconds: int = 20
    reports_auto_generate_daily: bool = False
    health_window_days_short: int = 20
    health_window_days_long: int = 60
    drift_maxdd_multiplier: float = 1.25
    drift_negative_return_cost_ratio_threshold: float = 0.35
    drift_win_rate_drop_pct: float = 0.15
    drift_return_delta_threshold: float = 0.15
    drift_warning_risk_scale: float = 0.75
    drift_degraded_risk_scale: float = 0.25
    drift_degraded_action: str = "PAUSE"
    evaluations_auto_promote_enabled: bool = False
    evaluations_min_window_days: int = 20
    evaluations_score_margin: float = 0.15
    evaluations_max_dd_multiplier: float = 1.10
    risk_overlay_enabled: bool = True
    risk_overlay_target_vol_annual: float = 0.18
    risk_overlay_lookback_days: int = 20
    risk_overlay_min_scale: float = 0.25
    risk_overlay_max_scale: float = 1.25
    risk_overlay_max_gross_exposure_pct: float = 1.0
    risk_overlay_max_single_name_exposure_pct: float = 0.12
    risk_overlay_max_sector_exposure_pct: float = 0.30
    risk_overlay_corr_clamp_enabled: bool = False
    risk_overlay_corr_threshold: float = 0.65
    risk_overlay_corr_reduce_factor: float = 0.5
    no_trade_enabled: bool = True
    no_trade_regimes: list[str] = Field(default_factory=lambda: ["HIGH_VOL"])
    no_trade_max_realized_vol_annual: float = 0.28
    no_trade_min_breadth_pct: float = 35.0
    no_trade_min_trend_strength: float = 15.0
    no_trade_cooldown_trading_days: int = 2

    four_hour_bars: str = Field(default="09:15-13:15,13:15-15:30")
    upstox_access_token: str | None = None
    upstox_client_id: str | None = None
    upstox_client_secret: str | None = None
    # Backward-friendly aliases for client credentials.
    upstox_api_key: str | None = None
    upstox_api_secret: str | None = None
    upstox_redirect_uri: str = "http://localhost:3000/providers/upstox/callback"
    upstox_base_url: str = "https://api.upstox.com"
    upstox_timeout_seconds: float = 12.0
    upstox_retry_max: int = 2
    upstox_retry_backoff_seconds: float = 1.0
    upstox_throttle_seconds: float = 0.15
    upstox_symbol_map_json: str | None = None
    upstox_intraday_interval: str = "15minute"
    upstox_oauth_state_ttl_seconds: int = 600
    upstox_expires_soon_seconds: int = 21_600
    upstox_persist_env_fallback: bool = False
    upstox_e2e_fake_code: str = "ATLAS_E2E_FAKE_CODE"
    upstox_auto_renew_enabled: bool = False
    upstox_auto_renew_time_ist: str = "06:30"
    upstox_auto_renew_if_expires_within_hours: int = 12
    upstox_auto_renew_lead_hours_before_open: int = 10
    upstox_auto_renew_only_when_provider_enabled: bool = True
    operate_provider_stage_on_token_invalid: str = "SKIP"
    upstox_notifier_base_url: str = "http://127.0.0.1:8000"
    upstox_notifier_secret: str | None = None
    upstox_notifier_pending_no_callback_minutes: int = 15
    upstox_notifier_stale_hours: int = 72
    cred_key: str | None = None
    cred_key_path: str = "data/secrets/atlas_cred.key"
    optuna_storage_url: str | None = None
    optuna_default_trials: int = 150
    optuna_default_timeout_seconds: int | None = None

    @property
    def fast_mode_enabled(self) -> bool:
        return bool(self.fast_mode or self.e2e_fast)

    def ensure_local_paths(self) -> None:
        Path("apps/api/.atlas").mkdir(parents=True, exist_ok=True)
        Path(self.parquet_root).mkdir(parents=True, exist_ok=True)
        Path(self.feature_cache_root).mkdir(parents=True, exist_ok=True)
        Path(self.calendar_data_root).mkdir(parents=True, exist_ok=True)
        Path(self.data_inbox_root).mkdir(parents=True, exist_ok=True)
        Path(self.secrets_root).mkdir(parents=True, exist_ok=True)
        Path(self.cred_key_path).parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    if settings.fast_mode_enabled:
        settings.jobs_inline = True
        settings.job_default_timeout_seconds = min(
            max(30, int(settings.job_default_timeout_seconds)),
            max(30, int(settings.fast_mode_job_timeout_seconds)),
        )
    settings.ensure_local_paths()
    return settings
