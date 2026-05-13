from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.db.models import DatasetBundle
from app.db.session import engine, init_db
from app.engine.signal_engine import SignalGenerationResult
from app.services.data_store import DataStore
from app.services.event_risk import evaluate_event_risk
from app.services.event_risk_sync import (
    NSE_ANNOUNCEMENTS_URL,
    refresh_event_risk_calendar_for_signals,
    sync_event_risk_calendar,
)
from app.services.paper import get_or_create_paper_state, preview_policy_signals


def _store(settings: Settings) -> DataStore:
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )


def _bundle(session: Session, symbols: list[str]) -> DatasetBundle:
    unique = uuid4().hex[:8]
    bundle = DatasetBundle(
        name=f"event-risk-sync-{unique}",
        provider="test",
        symbols_json=symbols,
        supported_timeframes_json=["1d"],
    )
    session.add(bundle)
    session.commit()
    session.refresh(bundle)
    assert bundle.id is not None
    return bundle


def _announcements_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "SYMBOL": "NEWSA",
                "BROADCAST DATE/TIME": "13-May-2026 12:30:00",
                "SUBJECT": "Financial Results",
                "DETAILS": "Board approved audited financial results.",
            },
            {
                "SYMBOL": "NEWSB",
                "BROADCAST DATE/TIME": "13-May-2026 13:00:00",
                "SUBJECT": "Order Win",
                "DETAILS": "Receipt of a material contract.",
            },
            {
                "SYMBOL": "OUTSIDE",
                "BROADCAST DATE/TIME": "13-May-2026 13:00:00",
                "SUBJECT": "Financial Results",
                "DETAILS": "Ignored because it is outside the bundle.",
            },
        ]
    )


def test_nse_announcements_sync_feeds_event_risk_calendar(tmp_path: Path) -> None:
    init_db()
    settings = get_settings()
    store = _store(settings)
    output = tmp_path / "event_risk_generated.csv"

    def fake_fetch(
        page_url: str,
        api_url: str,
        start_date: date,
        end_date: date,
        settings: Settings,
    ) -> pd.DataFrame:
        assert api_url == NSE_ANNOUNCEMENTS_URL
        assert start_date == date(2026, 5, 12)
        assert end_date == date(2026, 5, 14)
        return _announcements_frame()

    with Session(engine) as session:
        bundle = _bundle(session, ["NEWSA", "NEWSB"])
        result = sync_event_risk_calendar(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            start_date=date(2026, 5, 12),
            end_date=date(2026, 5, 14),
            output_path=output,
            include_actions=False,
            include_board_meetings=False,
            include_announcements=True,
            fetch_csv=fake_fetch,
        )

    assert result["status"] == "SUCCEEDED"
    assert result["source_counts"]["NSE_ANNOUNCEMENTS_SYNC"] == 2
    written = pd.read_csv(output)
    assert sorted(written["symbol"].tolist()) == ["NEWSA", "NEWSB"]
    assert set(written["source"]) == {"NSE_ANNOUNCEMENTS_SYNC"}

    risk = evaluate_event_risk(
        asof_date=date(2026, 5, 12),
        symbol="NEWSA",
        overrides={
            "event_risk_calendar_path": str(tmp_path / "manual_missing.csv"),
            "event_risk_generated_calendar_path": str(output),
        },
    )
    assert risk["status"] == "FAIL"
    assert "event_risk:results:NEWSA:2026-05-13" in risk["flags"]


def test_event_risk_refresh_uses_recent_meta_cache(tmp_path: Path) -> None:
    init_db()
    settings = get_settings()
    store = _store(settings)
    output = tmp_path / "event_risk_generated.csv"
    meta = tmp_path / "event_risk_sync_meta.json"
    calls = {"count": 0}

    def fake_fetch(
        page_url: str,
        api_url: str,
        start_date: date,
        end_date: date,
        settings: Settings,
    ) -> pd.DataFrame:
        calls["count"] += 1
        return _announcements_frame()

    overrides = {
        "event_risk_sync_before_signals": True,
        "event_risk_sync_include_actions": False,
        "event_risk_sync_include_board_meetings": False,
        "event_risk_sync_include_announcements": True,
        "event_risk_generated_calendar_path": str(output),
        "event_risk_sync_meta_path": str(meta),
        "event_risk_sync_min_interval_minutes": 60,
    }

    with Session(engine) as session:
        bundle = _bundle(session, ["NEWSA", "NEWSB"])
        first = refresh_event_risk_calendar_for_signals(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            asof_dt=datetime(2026, 5, 13, tzinfo=timezone.utc),
            overrides=overrides,
            fetch_csv=fake_fetch,
        )
        second = refresh_event_risk_calendar_for_signals(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            asof_dt=datetime(2026, 5, 13, tzinfo=timezone.utc),
            overrides=overrides,
            fetch_csv=fake_fetch,
        )

    assert first["status"] == "SUCCEEDED"
    assert second["status"] == "SKIPPED"
    assert second["reason"] == "recent_sync"
    assert calls["count"] == 1


def test_event_risk_sync_preserves_existing_rows_when_source_fails(tmp_path: Path) -> None:
    init_db()
    settings = get_settings()
    store = _store(settings)
    output = tmp_path / "event_risk_generated.csv"
    output.write_text(
        (
            "event_date,scope,symbol,event_type,severity,title,source,"
            "blackout_before_days,blackout_after_days\n"
            "2026-05-13,SYMBOL,NEWSA,RESULTS,BLOCK,Old result,"
            "NSE_ANNOUNCEMENTS_SYNC,2,1\n"
        ),
        encoding="utf-8",
    )

    def failing_fetch(
        page_url: str,
        api_url: str,
        start_date: date,
        end_date: date,
        settings: Settings,
    ) -> pd.DataFrame:
        raise RuntimeError("nse unavailable")

    with Session(engine) as session:
        bundle = _bundle(session, ["NEWSA"])
        result = sync_event_risk_calendar(
            session=session,
            settings=settings,
            store=store,
            bundle_id=int(bundle.id),
            start_date=date(2026, 5, 12),
            end_date=date(2026, 5, 14),
            output_path=output,
            include_actions=False,
            include_board_meetings=False,
            include_announcements=True,
            fetch_csv=failing_fetch,
        )

    assert result["status"] == "FAILED"
    preserved = pd.read_csv(output)
    assert preserved.iloc[0]["title"] == "Old result"


def test_preview_refreshes_event_risk_before_generating_signals(monkeypatch) -> None:
    init_db()
    settings = get_settings()
    store = _store(settings)
    refresh_calls: list[dict] = []

    def fake_refresh(**kwargs) -> dict:
        refresh_calls.append(kwargs)
        return {"status": "SUCCEEDED", "refreshed": True, "event_count": 1}

    def fake_generate(**kwargs) -> SignalGenerationResult:
        assert refresh_calls
        return SignalGenerationResult(
            signals=[],
            scan_truncated=False,
            scanned_symbols=0,
            evaluated_candidates=0,
            total_symbols=0,
        )

    monkeypatch.setattr("app.services.paper.refresh_event_risk_calendar_for_signals", fake_refresh)
    monkeypatch.setattr("app.services.paper.generate_signals_for_policy", fake_generate)

    with Session(engine) as session:
        bundle = _bundle(session, ["NEWSA"])
        state = get_or_create_paper_state(session, settings)
        state.settings_json = {
            **dict(state.settings_json or {}),
            "paper_mode": "strategy",
            "event_risk_sync_before_signals": True,
        }
        session.add(state)
        session.commit()

        result = preview_policy_signals(
            session=session,
            settings=settings,
            store=store,
            payload={
                "bundle_id": int(bundle.id),
                "timeframes": ["1d"],
                "symbol_scope": "all",
                "max_symbols_scan": 5,
                "seed": 31,
                "asof": "2026-05-13T10:00:00+00:00",
            },
        )

    assert result["event_risk_refresh"]["status"] == "SUCCEEDED"
    assert refresh_calls[0]["bundle_id"] == int(bundle.id)
