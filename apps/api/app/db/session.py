from typing import Generator

from sqlalchemy import text
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import get_settings


settings = get_settings()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)


def init_db() -> None:
    SQLModel.metadata.create_all(engine)
    _ensure_indexes_and_columns()


def _has_column(table: str, column: str) -> bool:
    with engine.connect() as conn:
        if engine.url.get_backend_name().startswith("sqlite"):
            rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()  # noqa: S608
            return any(str(row[1]) == column for row in rows)
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = :table_name AND column_name = :column_name"
            ),
            {"table_name": table, "column_name": column},
        ).fetchall()
        return len(rows) > 0


def _ensure_indexes_and_columns() -> None:
    is_sqlite = engine.url.get_backend_name().startswith("sqlite")
    if not _has_column("job", "created_at"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE job ADD COLUMN created_at TIMESTAMP"))
            conn.execute(
                text("UPDATE job SET created_at = CURRENT_TIMESTAMP WHERE created_at IS NULL")
            )
    if not _has_column("job", "idempotency_key"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE job ADD COLUMN idempotency_key VARCHAR(128)"))
    if not _has_column("job", "request_hash"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE job ADD COLUMN request_hash VARCHAR(128)"))
    if not _has_column("dataset", "symbols_json"):
        column_type = "TEXT" if is_sqlite else "JSON"
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE dataset ADD COLUMN symbols_json {column_type}"))
    if not _has_column("dataset", "bundle_id"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE dataset ADD COLUMN bundle_id INTEGER"))
    if not _has_column("researchrun", "bundle_id"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE researchrun ADD COLUMN bundle_id INTEGER"))
    if not _has_column("paperposition", "side"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE paperposition ADD COLUMN side VARCHAR(8) DEFAULT 'BUY'"))
    if not _has_column("paperposition", "instrument_kind"):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE paperposition ADD COLUMN instrument_kind VARCHAR(16) "
                    "DEFAULT 'EQUITY_CASH'"
                )
            )
    if not _has_column("paperposition", "lot_size"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE paperposition ADD COLUMN lot_size INTEGER DEFAULT 1"))
    if not _has_column("paperposition", "qty_lots"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE paperposition ADD COLUMN qty_lots INTEGER DEFAULT 1"))
    if not _has_column("paperposition", "margin_reserved"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE paperposition ADD COLUMN margin_reserved FLOAT DEFAULT 0"))
    if not _has_column("paperposition", "must_exit_by_eod"):
        bool_type = "INTEGER" if is_sqlite else "BOOLEAN"
        with engine.begin() as conn:
            conn.execute(
                text(f"ALTER TABLE paperposition ADD COLUMN must_exit_by_eod {bool_type} DEFAULT 0")
            )
    if not _has_column("paperposition", "metadata_json"):
        column_type = "TEXT" if is_sqlite else "JSON"
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE paperposition ADD COLUMN metadata_json {column_type}"))
    if not _has_column("paperrun", "mode"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE paperrun ADD COLUMN mode VARCHAR(16) DEFAULT 'LIVE'"))
    if not _has_column("paperorder", "instrument_kind"):
        with engine.begin() as conn:
            conn.execute(
                text(
                    "ALTER TABLE paperorder ADD COLUMN instrument_kind VARCHAR(16) "
                    "DEFAULT 'EQUITY_CASH'"
                )
            )
    if not _has_column("paperorder", "lot_size"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE paperorder ADD COLUMN lot_size INTEGER DEFAULT 1"))
    if not _has_column("paperorder", "qty_lots"):
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE paperorder ADD COLUMN qty_lots INTEGER DEFAULT 1"))

    with engine.begin() as conn:
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_job_status_created_at ON job (status, created_at)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_trade_backtest_entry_dt ON trade (backtest_id, entry_dt)"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_walkforwardfold_run_id ON walkforwardfold (run_id)")
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_paperorder_created_at ON paperorder (created_at)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_researchcandidate_run_score ON researchcandidate (run_id, score)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_researchcandidate_run_rank ON researchcandidate (run_id, rank)"
            )
        )
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_dataset_bundle_id ON dataset (bundle_id)"))
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_researchrun_bundle_id ON researchrun (bundle_id)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_paperrun_mode_created "
                "ON paperrun (mode, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_paperrun_policy_asof "
                "ON paperrun (policy_id, asof_ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_paperrun_bundle_asof "
                "ON paperrun (bundle_id, asof_ts)"
            )
        )
        conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS ix_shadowpaperstate_bundle_policy "
                "ON shadowpaperstate (bundle_id, policy_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_shadowpaperstate_updated_at "
                "ON shadowpaperstate (updated_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_dailyreport_date_bundle_policy "
                "ON dailyreport (date, bundle_id, policy_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_policyhealthsnapshot_policy_window_date "
                "ON policyhealthsnapshot (policy_id, window_days, asof_date)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_policyevaluation_status_created "
                "ON policyevaluation (status, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_policyshadowrun_eval_policy_asof "
                "ON policyshadowrun (evaluation_id, policy_id, asof_date)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_replayrun_policy_created "
                "ON replayrun (policy_id, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_replayrun_bundle_created "
                "ON replayrun (bundle_id, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_monthlyreport_month_bundle_policy "
                "ON monthlyreport (month, bundle_id, policy_id)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_dataquality_bundle_timeframe_created "
                "ON dataqualityreport (bundle_id, timeframe, created_at)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_dataquality_status_created "
                "ON dataqualityreport (status, created_at)"
            )
        )
        conn.execute(
            text("CREATE INDEX IF NOT EXISTS ix_operateevent_ts ON operateevent (ts)")
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_operateevent_severity_ts "
                "ON operateevent (severity, ts)"
            )
        )
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_operateevent_category_ts "
                "ON operateevent (category, ts)"
            )
        )


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
