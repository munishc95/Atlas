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
    sample_data_root: str = "data/sample"
    redis_url: str = "redis://localhost:6379/0"
    rq_queue_name: str = "atlas"
    jobs_inline: bool = False
    job_default_timeout_seconds: int = 10_800
    job_retry_max: int = 2
    job_retry_backoff_seconds: int = 5

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
    max_position_value_pct_adv: float = 0.01
    diversification_corr_threshold: float = 0.75

    four_hour_bars: str = Field(default="09:15-13:15,13:15-15:30")
    optuna_storage_url: str | None = None
    optuna_default_trials: int = 150
    optuna_default_timeout_seconds: int | None = None

    def ensure_local_paths(self) -> None:
        Path("apps/api/.atlas").mkdir(parents=True, exist_ok=True)
        Path(self.parquet_root).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_local_paths()
    return settings
