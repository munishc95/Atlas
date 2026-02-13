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


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
