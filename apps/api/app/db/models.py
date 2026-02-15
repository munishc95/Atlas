from datetime import date as dt_date, datetime, timezone
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


class Instrument(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, max_length=32)
    kind: str = Field(index=True, max_length=16)  # EQUITY_CASH | STOCK_FUT | INDEX_FUT
    underlying: str | None = Field(default=None, index=True, max_length=32)
    lot_size: int = 1
    tick_size: float = 0.05
    metadata_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))


class DatasetBundle(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True, max_length=128)
    provider: str = Field(max_length=64)
    description: str | None = Field(default=None, max_length=512)
    symbols_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    supported_timeframes_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class Dataset(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    provider: str = Field(max_length=64)
    symbol: str = Field(index=True, max_length=32)
    timeframe: str = Field(index=True, max_length=16)
    symbols_json: list[str] | None = Field(default=None, sa_column=Column(JSON))
    start_date: dt_date
    end_date: dt_date
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
    start_date: dt_date
    end_date: dt_date
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
    fold_start: dt_date
    fold_end: dt_date
    params_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ResearchRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)
    dataset_id: int | None = Field(default=None, foreign_key="dataset.id")
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
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


class PolicyHealthSnapshot(SQLModel, table=True):
    __table_args__ = (
        Index("ix_policyhealthsnapshot_policy_window_date", "policy_id", "window_days", "asof_date"),
    )

    id: int | None = Field(default=None, primary_key=True)
    policy_id: int = Field(foreign_key="policy.id", index=True)
    asof_date: dt_date = Field(index=True)
    window_days: int = Field(index=True)
    metrics_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default="HEALTHY", index=True, max_length=16)
    reasons_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class PaperRun(SQLModel, table=True):
    __table_args__ = (
        Index("ix_paperrun_mode_created", "mode", "created_at"),
        Index("ix_paperrun_policy_asof", "policy_id", "asof_ts"),
        Index("ix_paperrun_bundle_asof", "bundle_id", "asof_ts"),
    )

    id: int | None = Field(default=None, primary_key=True)
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    policy_id: int | None = Field(default=None, foreign_key="policy.id", index=True)
    asof_ts: datetime = Field(index=True)
    mode: str = Field(default="LIVE", index=True, max_length=16)
    regime: str = Field(default="TREND_UP", max_length=24)
    signals_source: str = Field(default="provided", max_length=24)
    generated_signals_count: int = 0
    selected_signals_count: int = 0
    skipped_signals_count: int = 0
    scanned_symbols: int = 0
    evaluated_candidates: int = 0
    scan_truncated: bool = False
    summary_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    cost_summary_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class PortfolioRiskSnapshot(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utc_now, index=True)
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    policy_id: int | None = Field(default=None, foreign_key="policy.id", index=True)
    realized_vol: float = 0.0
    target_vol: float = 0.0
    scale: float = 1.0
    notes_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class ShadowPaperState(SQLModel, table=True):
    __table_args__ = (
        Index("ix_shadowpaperstate_bundle_policy", "bundle_id", "policy_id", unique=True),
        Index("ix_shadowpaperstate_updated_at", "updated_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    bundle_id: int = Field(default=0, index=True)
    policy_id: int = Field(default=0, index=True)
    updated_at: datetime = Field(default_factory=utc_now)
    state_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    last_run_id: int | None = Field(default=None, foreign_key="paperrun.id", index=True)


class DataQualityReport(SQLModel, table=True):
    __table_args__ = (
        Index("ix_dataquality_bundle_timeframe_created", "bundle_id", "timeframe", "created_at"),
        Index("ix_dataquality_status_created", "status", "created_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    timeframe: str = Field(default="1d", index=True, max_length=16)
    status: str = Field(default="OK", index=True, max_length=8)
    issues_json: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    last_bar_ts: datetime | None = Field(default=None, index=True)
    coverage_pct: float = 100.0
    checked_symbols: int = 0
    total_symbols: int = 0
    created_at: datetime = Field(default_factory=utc_now)


class DataUpdateRun(SQLModel, table=True):
    __table_args__ = (
        Index("ix_dataupdaterun_bundle_timeframe_created", "bundle_id", "timeframe", "created_at"),
        Index("ix_dataupdaterun_status_created", "status", "created_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    timeframe: str = Field(default="1d", index=True, max_length=16)
    status: str = Field(default="QUEUED", index=True, max_length=16)
    inbox_path: str = Field(default="", max_length=512)
    scanned_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    rows_ingested: int = 0
    symbols_affected_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    warnings_json: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    errors_json: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)
    ended_at: datetime | None = Field(default=None)


class DataUpdateFile(SQLModel, table=True):
    __table_args__ = (
        Index("ix_dataupdatefile_bundle_timeframe_hash", "bundle_id", "timeframe", "file_hash"),
        Index("ix_dataupdatefile_status_created", "status", "created_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="dataupdaterun.id", index=True)
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    timeframe: str = Field(default="1d", index=True, max_length=16)
    file_path: str = Field(default="", max_length=1024)
    file_name: str = Field(default="", max_length=256)
    file_hash: str = Field(default="", max_length=128)
    status: str = Field(default="PENDING", index=True, max_length=16)
    reason: str | None = Field(default=None, max_length=128)
    rows_ingested: int = 0
    symbols_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    warnings_json: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    errors_json: list[dict[str, Any]] = Field(default_factory=list, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class OperateEvent(SQLModel, table=True):
    __table_args__ = (
        Index("ix_operateevent_ts", "ts"),
        Index("ix_operateevent_severity_ts", "severity", "ts"),
        Index("ix_operateevent_category_ts", "category", "ts"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utc_now)
    severity: str = Field(default="INFO", max_length=8, index=True)
    category: str = Field(default="SYSTEM", max_length=16, index=True)
    message: str = Field(max_length=256)
    details_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    correlation_id: str | None = Field(default=None, max_length=64, index=True)


class DailyReport(SQLModel, table=True):
    __table_args__ = (
        Index("ix_dailyreport_date_bundle_policy", "date", "bundle_id", "policy_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    date: dt_date = Field(index=True)
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    policy_id: int | None = Field(default=None, foreign_key="policy.id", index=True)
    content_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class MonthlyReport(SQLModel, table=True):
    __table_args__ = (
        Index("ix_monthlyreport_month_bundle_policy", "month", "bundle_id", "policy_id"),
    )

    id: int | None = Field(default=None, primary_key=True)
    month: str = Field(index=True, max_length=7)  # YYYY-MM
    bundle_id: int | None = Field(default=None, foreign_key="datasetbundle.id", index=True)
    policy_id: int | None = Field(default=None, foreign_key="policy.id", index=True)
    content_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class PolicyEvaluation(SQLModel, table=True):
    __table_args__ = (
        Index("ix_policyevaluation_created_at", "created_at"),
        Index("ix_policyevaluation_status_created", "status", "created_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)
    bundle_id: int = Field(foreign_key="datasetbundle.id", index=True)
    regime: str | None = Field(default=None, max_length=24)
    window_start: dt_date = Field(index=True)
    window_end: dt_date = Field(index=True)
    champion_policy_id: int = Field(foreign_key="policy.id", index=True)
    challenger_policy_ids_json: list[int] = Field(default_factory=list, sa_column=Column(JSON))
    status: str = Field(default="QUEUED", index=True, max_length=16)
    summary_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    notes: str | None = Field(default=None, max_length=2048)


class PolicyShadowRun(SQLModel, table=True):
    __table_args__ = (
        Index("ix_policyshadowrun_eval_policy_asof", "evaluation_id", "policy_id", "asof_date"),
    )

    id: int | None = Field(default=None, primary_key=True)
    evaluation_id: int = Field(foreign_key="policyevaluation.id", index=True)
    policy_id: int = Field(foreign_key="policy.id", index=True)
    asof_date: dt_date = Field(index=True)
    run_summary_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=utc_now)


class AutoEvalRun(SQLModel, table=True):
    __table_args__ = (
        Index("ix_autoevalrun_ts", "ts"),
        Index("ix_autoevalrun_bundle_ts", "bundle_id", "ts"),
        Index("ix_autoevalrun_active_policy_ts", "active_policy_id", "ts"),
        Index("ix_autoevalrun_reco_action_ts", "recommended_action", "ts"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utc_now, index=True)
    bundle_id: int = Field(foreign_key="datasetbundle.id", index=True)
    active_policy_id: int = Field(foreign_key="policy.id", index=True)
    recommended_action: str = Field(default="KEEP", index=True, max_length=32)
    recommended_policy_id: int | None = Field(default=None, foreign_key="policy.id", index=True)
    reasons_json: list[str] = Field(default_factory=list, sa_column=Column(JSON))
    score_table_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    lookback_days: int = 20
    digest: str = Field(default="", index=True, max_length=128)
    status: str = Field(default="SUCCEEDED", max_length=16, index=True)
    auto_switch_attempted: bool = False
    auto_switch_applied: bool = False
    details_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class PolicySwitchEvent(SQLModel, table=True):
    __table_args__ = (
        Index("ix_policyswitchevent_ts", "ts"),
        Index("ix_policyswitchevent_autoeval_ts", "auto_eval_id", "ts"),
    )

    id: int | None = Field(default=None, primary_key=True)
    ts: datetime = Field(default_factory=utc_now, index=True)
    from_policy_id: int = Field(foreign_key="policy.id", index=True)
    to_policy_id: int = Field(foreign_key="policy.id", index=True)
    reason: str = Field(default="", max_length=512)
    auto_eval_id: int | None = Field(default=None, foreign_key="autoevalrun.id", index=True)
    cooldown_state_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    mode: str = Field(default="MANUAL", max_length=16, index=True)


class ReplayRun(SQLModel, table=True):
    __table_args__ = (
        Index("ix_replayrun_policy_created", "policy_id", "created_at"),
        Index("ix_replayrun_bundle_created", "bundle_id", "created_at"),
    )

    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utc_now)
    bundle_id: int = Field(foreign_key="datasetbundle.id", index=True)
    policy_id: int = Field(foreign_key="policy.id", index=True)
    regime: str | None = Field(default=None, max_length=24)
    start_date: dt_date
    end_date: dt_date
    seed: int = 7
    status: str = Field(default="QUEUED", index=True, max_length=16)
    summary_json: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


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
    side: str = Field(default="BUY", max_length=8)
    instrument_kind: str = Field(default="EQUITY_CASH", max_length=16)
    lot_size: int = 1
    qty_lots: int = 1
    margin_reserved: float = 0.0
    must_exit_by_eod: bool = False
    qty: int
    avg_price: float
    stop_price: float | None = None
    target_price: float | None = None
    metadata_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    opened_at: datetime = Field(default_factory=utc_now)


class PaperOrder(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    symbol: str = Field(index=True, max_length=32)
    side: str = Field(max_length=8)
    instrument_kind: str = Field(default="EQUITY_CASH", max_length=16)
    lot_size: int = 1
    qty_lots: int = 1
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
