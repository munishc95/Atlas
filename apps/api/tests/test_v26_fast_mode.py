from __future__ import annotations

from app.core.config import Settings
from app.services.fast_mode import (
    clamp_job_timeout_seconds,
    clamp_optuna_trials,
    clamp_scan_symbols,
    fast_mode_enabled,
    resolve_seed,
)


def test_fast_mode_caps_scan_and_trials_deterministically() -> None:
    settings = Settings(
        fast_mode=True,
        fast_mode_max_symbols_scan=10,
        fast_mode_max_optuna_trials=4,
        fast_mode_seed=13,
        job_default_timeout_seconds=10_800,
        fast_mode_job_timeout_seconds=900,
    )

    assert fast_mode_enabled(settings) is True
    assert clamp_scan_symbols(settings=settings, requested=200, hard_cap=250) == 10
    assert clamp_scan_symbols(settings=settings, requested=6, hard_cap=250) == 6
    assert clamp_optuna_trials(settings=settings, requested=150) == 4
    assert clamp_optuna_trials(settings=settings, requested=3) == 3
    assert clamp_job_timeout_seconds(settings=settings, requested=10_800) == 900
    assert resolve_seed(settings=settings, value=None, default=7) == 13
    assert resolve_seed(settings=settings, value=21, default=7) == 21


def test_non_fast_mode_keeps_requested_limits() -> None:
    settings = Settings(
        fast_mode=False,
        e2e_fast=False,
        fast_mode_max_symbols_scan=10,
        fast_mode_max_optuna_trials=4,
        job_default_timeout_seconds=10_800,
        fast_mode_job_timeout_seconds=900,
    )

    assert fast_mode_enabled(settings) is False
    assert clamp_scan_symbols(settings=settings, requested=200, hard_cap=120) == 120
    assert clamp_optuna_trials(settings=settings, requested=150) == 150
    assert clamp_job_timeout_seconds(settings=settings, requested=6000) == 6000
    assert resolve_seed(settings=settings, value=None, default=7) == 7
