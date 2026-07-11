"""Global pytest safety boundaries for Atlas.

The application defaults to a persistent local SQLite database and data directories.
Tests exercise destructive reset paths, so they must never inherit those defaults.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterator

import pytest


_TEST_ROOT = Path(tempfile.mkdtemp(prefix="atlas-pytest-"))
_BASE_ATLAS_ENV: dict[str, str] = {}


def _test_path(name: str) -> str:
    return str(_TEST_ROOT / name)


def _configure_isolated_runtime() -> None:
    test_database_url = os.environ.get("ATLAS_TEST_DATABASE_URL")
    isolated = {
        "ATLAS_ENVIRONMENT": "test",
        "ATLAS_DATABASE_URL": test_database_url
        or f"sqlite:///{(_TEST_ROOT / 'atlas.db').as_posix()}",
        "ATLAS_DUCKDB_PATH": _test_path("ohlcv.duckdb"),
        "ATLAS_PARQUET_ROOT": _test_path("parquet"),
        "ATLAS_FEATURE_CACHE_ROOT": _test_path("features"),
        "ATLAS_DATA_INBOX_ROOT": _test_path("inbox"),
        "ATLAS_SECRETS_ROOT": _test_path("secrets"),
        "ATLAS_TRAIN_DATASETS_ROOT": _test_path("train_datasets"),
        "ATLAS_NSE_BHAVCOPY_CACHE_DIR": _test_path("nse_bhavcopy_cache"),
        "ATLAS_CRED_KEY_PATH": _test_path("secrets/atlas_cred.key"),
        "ATLAS_EVENT_RISK_GENERATED_CALENDAR_PATH": _test_path(
            "inbox/event_risk_generated.csv"
        ),
        "ATLAS_EVENT_RISK_SYNC_META_PATH": _test_path("inbox/event_risk_sync_meta.json"),
        "ATLAS_OPTUNA_STORAGE_URL": f"sqlite:///{(_TEST_ROOT / 'optuna.db').as_posix()}",
        "ATLAS_FAST_MODE": "false",
        "ATLAS_E2E_FAST": "false",
        "ATLAS_TELEGRAM_ENABLED": "false",
        "ATLAS_UPSTOX_AUTO_RENEW_ENABLED": "false",
        "ATLAS_OPERATE_AUTO_RUN_ENABLED": "false",
        "ATLAS_EVENT_RISK_SYNC_BEFORE_SIGNALS": "false",
        "ATLAS_DATA_UPDATES_PROVIDER_ENABLED": "false",
    }
    os.environ.update(isolated)


def _atlas_environment() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key.startswith("ATLAS_")}


def _restore_atlas_environment(snapshot: dict[str, str]) -> None:
    for key in tuple(os.environ):
        if key.startswith("ATLAS_") and key not in snapshot:
            os.environ.pop(key, None)
    os.environ.update(snapshot)


_configure_isolated_runtime()


def pytest_collection_finish() -> None:
    """Capture the canonical test environment after module-level test setup."""

    global _BASE_ATLAS_ENV
    _BASE_ATLAS_ENV = _atlas_environment()


@pytest.fixture(autouse=True)
def isolate_atlas_settings() -> Iterator[None]:
    """Prevent environment and cached settings from leaking between tests."""

    from app.core.config import get_settings

    baseline = _BASE_ATLAS_ENV or _atlas_environment()
    _restore_atlas_environment(baseline)
    get_settings.cache_clear()
    yield
    _restore_atlas_environment(baseline)
    get_settings.cache_clear()


def pytest_unconfigure() -> None:
    """Close database handles before removing the disposable test workspace."""

    try:
        from app.db.session import engine

        engine.dispose()
    except ImportError:
        pass
    shutil.rmtree(_TEST_ROOT, ignore_errors=True)
