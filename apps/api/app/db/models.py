from datetime import date, datetime, timezone
from typing import Any

from sqlalchemy import JSON, Column, Index
from sqlmodel import Field, SQLModel


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Symbol(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, unique=True, max_length=32)
    name: str = Field(max_length=128)
    sector: str | None = Field(default=None, max_length=64)
    metadata_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    provider: str = Field(max_length=64)
    symbol: str = Field(index=True, max_length=32)
    timeframe: str = Field(index=True, max_length=16)
    start_date: date
    end_date: date
    checksum: str | None = Field(default=None, max_length=128)
    created_at: datetime = Field(default_factory=utc_now)


class Strategy(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(max_length=128)
    template: str = Field(index=True, max_length=64)
    params_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    enabled: bool = True
    promoted_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Backtest(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    strategy_id: int | None = Field(default=None, foreign_key="strategy.id")
    symbol: str = Field(index=True, max_length=32)
    timeframe: str = Field(index=True, max_length=16)
    start_date: date
    end_date: date
    config_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class Trade(SQLModel, table=True):
    __table_args__ = (Index("ix_trade_backtest_entry_dt", "backtest_id", "entry_dt"),)

    id: int | None = Field(default=None, primary_key=True)
    backtest_id: int = Field(foreign_key="backtest.id", index=True)
    symbol: str = Field(max_length=32)
    entry_dt: datetime
    exit_dt: datetime
    qty: int
    entry_px: float
    exit_px: float
    pnl: float
    r_multiple: float
    reason: str = Field(max_length=64)


class WalkForwardRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    config_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    summary_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class WalkForwardFold(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="walkforwardrun.id", index=True)
    fold_start: date
    fold_end: date
    params_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ResearchRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)
    dataset_id: int | None = Field(default=None, foreign_key="dataset.id")
    timeframes_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    config_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="QUEUED", index=True, max_length=16)
    summary_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ResearchCandidate(SQLModel, table=True):
    __table_args__ = (
        Index("ix_researchcandidate_run_score", "run_id", "score"),
        Index("ix_researchcandidate_run_rank", "run_id", "rank"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="researchrun.id", index=True)
    symbol: str = Field(index=True, max_length=32)
    timeframe: str = Field(max_length=16)
    strategy_key: str = Field(index=True, max_length=64)
    best_params_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    oos_metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    stress_metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    param_dispersion: float = 0.0
    fold_variance: float = 0.0
    stress_pass_rate: float = 0.0
    score: float = 0.0
    rank: int = 0
    accepted: bool = False
    explanations_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))


class Policy(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, max_length=128)
    created_at: datetime = Field(default_factory=utc_now)
    definition_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    promoted_from_research_run_id: int | None = Field(default=None, foreign_key="researchrun.id")


class PaperState(SQLModel, table=True):
    id: int = Field(default=1, primary_key=True)
    equity: float = 1_000_000.0
    cash: float = 1_000_000.0
    peak_equity: float = 1_000_000.0
    drawdown: float = 0.0
    kill_switch_active: bool = False
    cooldown_days_left: int = 0
    settings_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class PaperPosition(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, max_length=32)
    qty: int
    avg_price: float
    stop_price: float | None = None
    target_price: float | None = None
    opened_at: datetime = Field(default_factory=utc_now)


class PaperOrder(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, max_length=32)
    side: str = Field(max_length=8)
    qty: int
    limit_price: float | None = None
    fill_price: float | None = None
    status: str = Field(default="NEW", max_length=16)
    reason: str | None = Field(default=None, max_length=64)
    created_at: datetime = Field(default_factory=utc_now, index=True)
    updated_at: datetime = Field(default_factory=utc_now)


class AuditLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utc_now)
    type: str = Field(max_length=64)
    payload_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class Job(SQLModel, table=True):
    __table_args__ = (Index("ix_job_status_created_at", "status", "created_at"),)

    id: str = Field(primary_key=True, max_length=64)
    type: str = Field(max_length=64)
    status: str = Field(default="QUEUED", max_length=16)
    created_at: datetime = Field(default_factory=utc_now)
    progress: int = 0
    started_at: datetime | None = None
    ended_at: datetime | None = None
    idempotency_key: str | None = Field(default=None, index=True, max_length=128)
    request_hash: str | None = Field(default=None, index=True, max_length=128)
    logs_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    result_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
