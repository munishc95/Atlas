from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import get_settings
from app.db.session import engine
from app.engine.signal_engine import (
    _market_context_from_frames,
    _market_context_quality_for_side,
    generate_signals_for_policy,
)
from app.services.data_store import DataStore
from app.services.event_risk import evaluate_event_risk


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
    )


def _flat_frame(rows: int = 80) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D", tz="UTC")
    frame = pd.DataFrame(
        {
            "datetime": idx,
            "open": np.full(rows, 100.0),
            "high": np.full(rows, 101.0),
            "low": np.full(rows, 99.0),
            "close": np.full(rows, 100.0),
            "volume": np.full(rows, 1_500_000),
        }
    )
    return frame


def test_signal_engine_uses_asof_without_lookahead() -> None:
    provider = f"sig-{uuid4().hex[:8]}"
    symbol = f"LOOKAHEAD_{uuid4().hex[:6].upper()}"
    store = _store()

    frame_no_fill = _flat_frame()
    frame_no_fill.loc[frame_no_fill.index[-1], "open"] = 100.0
    frame_no_fill.loc[frame_no_fill.index[-1], "high"] = 121.0
    frame_no_fill.loc[frame_no_fill.index[-1], "low"] = 99.0
    frame_no_fill.loc[frame_no_fill.index[-1], "close"] = 120.0

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame_no_fill,
            provider=provider,
        )
        assert dataset.id is not None
        no_fill = generate_signals_for_policy(
            session=session,
            store=store,
            dataset_id=dataset.id,
            asof=frame_no_fill.iloc[-1]["datetime"],
            timeframes=["1d"],
            allowed_templates=["trend_breakout"],
            params_overrides={
                "trend_breakout": {
                    "trend_period": 10,
                    "breakout_lookback": 15,
                    "direction": "both",
                }
            },
            symbol_scope="all",
            max_symbols_scan=5,
            seed=13,
        )
        assert len(no_fill.signals) == 0

    frame_with_fill = pd.concat(
        [
            frame_no_fill,
            pd.DataFrame(
                [
                    {
                        "datetime": frame_no_fill.iloc[-1]["datetime"] + pd.Timedelta(days=1),
                        "open": 121.0,
                        "high": 122.0,
                        "low": 120.0,
                        "close": 121.5,
                        "volume": 1_600_000,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame_with_fill,
            provider=provider,
        )
        assert dataset.id is not None
        with_fill = generate_signals_for_policy(
            session=session,
            store=store,
            dataset_id=dataset.id,
            asof=frame_with_fill.iloc[-1]["datetime"],
            timeframes=["1d"],
            allowed_templates=["trend_breakout"],
            params_overrides={
                "trend_breakout": {
                    "trend_period": 10,
                    "breakout_lookback": 15,
                    "direction": "both",
                }
            },
            symbol_scope="all",
            max_symbols_scan=5,
            seed=13,
        )
        assert any(signal.get("side") == "BUY" for signal in with_fill.signals)
        top = with_fill.signals[0]
        assert str(top.get("signal_at")) < str(top.get("fill_at"))
        assert float(top["stop_price"]) < float(top["entry_price"])
        assert float(top["target_1_price"]) > float(top["entry_price"])
        assert float(top["target_2_price"]) > float(top["target_1_price"])
        assert float(top["risk_per_share"]) == float(top["stop_distance"])


def test_signal_engine_falls_back_for_older_dataset_outside_live_window() -> None:
    provider = f"sig-old-{uuid4().hex[:8]}"
    symbol = f"OLDWIN_{uuid4().hex[:6].upper()}"
    store = _store()
    frame = _flat_frame(90)
    decision_idx = len(frame) - 2
    frame.loc[decision_idx, ["open", "high", "low", "close"]] = [115.0, 122.0, 114.0, 121.0]
    frame.loc[decision_idx, "volume"] = 2_200_000
    frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = [121.5, 122.5, 120.0, 121.8]

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame,
            provider=provider,
        )
        assert dataset.id is not None
        result = generate_signals_for_policy(
            session=session,
            store=store,
            dataset_id=dataset.id,
            asof=pd.Timestamp("2026-05-05", tz="UTC"),
            timeframes=["1d"],
            allowed_templates=["trend_breakout"],
            params_overrides={
                "trend_breakout": {
                    "trend_period": 10,
                    "breakout_lookback": 15,
                    "direction": "both",
                }
            },
            symbol_scope="all",
            max_symbols_scan=5,
            seed=29,
        )

    assert any(signal.get("side") == "BUY" for signal in result.signals)


def test_signal_engine_flags_weak_breakout_candidate_quality() -> None:
    provider = f"sig-quality-{uuid4().hex[:8]}"
    symbol = f"WEAKBRK_{uuid4().hex[:6].upper()}"
    store = _store()
    frame = _flat_frame(90)
    decision_idx = len(frame) - 2
    frame.loc[decision_idx, "open"] = 105.0
    frame.loc[decision_idx, "high"] = 180.0
    frame.loc[decision_idx, "low"] = 100.0
    frame.loc[decision_idx, "close"] = 115.0
    frame.loc[decision_idx, "volume"] = 4_000_000
    frame.loc[len(frame) - 1, "open"] = 116.0
    frame.loc[len(frame) - 1, "high"] = 118.0
    frame.loc[len(frame) - 1, "low"] = 112.0
    frame.loc[len(frame) - 1, "close"] = 114.0

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame,
            provider=provider,
        )
        assert dataset.id is not None
        result = generate_signals_for_policy(
            session=session,
            store=store,
            dataset_id=dataset.id,
            asof=frame.iloc[-1]["datetime"],
            timeframes=["1d"],
            allowed_templates=["trend_breakout"],
            params_overrides={
                "trend_breakout": {
                    "trend_period": 10,
                    "breakout_lookback": 15,
                    "direction": "both",
                }
            },
            symbol_scope="all",
            max_symbols_scan=5,
            seed=19,
        )

    assert len(result.signals) == 1
    signal = result.signals[0]
    assert signal["quality_status"] == "FAIL"
    assert "weak_signal_bar_close" in signal["quality_flags"]


def test_signal_engine_flags_recent_price_discontinuity() -> None:
    provider = f"sig-gap-{uuid4().hex[:8]}"
    symbol = f"GAPBRK_{uuid4().hex[:6].upper()}"
    store = _store()
    frame = _flat_frame(120)
    decision_idx = len(frame) - 2
    gap_idx = decision_idx - 5
    frame.loc[gap_idx, ["open", "high", "low", "close"]] = [45.0, 46.0, 44.0, 45.0]
    frame.loc[gap_idx + 1, ["open", "high", "low", "close"]] = [100.0, 101.0, 99.0, 100.0]
    frame.loc[decision_idx, ["open", "high", "low", "close"]] = [119.0, 122.0, 118.0, 121.0]
    frame.loc[decision_idx, "volume"] = 2_200_000
    frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = [121.5, 122.5, 120.0, 121.8]

    with Session(engine) as session:
        dataset = store.save_ohlcv(
            session=session,
            symbol=symbol,
            timeframe="1d",
            frame=frame,
            provider=provider,
        )
        assert dataset.id is not None
        result = generate_signals_for_policy(
            session=session,
            store=store,
            dataset_id=dataset.id,
            asof=frame.iloc[-1]["datetime"],
            timeframes=["1d"],
            allowed_templates=["trend_breakout"],
            params_overrides={
                "trend_breakout": {
                    "trend_period": 10,
                    "breakout_lookback": 15,
                    "direction": "both",
                }
            },
            symbol_scope="all",
            max_symbols_scan=5,
            seed=23,
        )

    assert len(result.signals) == 1
    signal = result.signals[0]
    assert signal["quality_status"] == "FAIL"
    assert "recent_price_discontinuity" in signal["quality_flags"]


def test_market_context_flags_breadth_breakdown() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for index in range(12):
        frame = _flat_frame(240)
        trend = np.linspace(140.0, 70.0, len(frame))
        frame["open"] = trend
        frame["high"] = trend + 1.0
        frame["low"] = trend - 1.0
        frame["close"] = trend
        frame["volume"] = 1_500_000 + index
        frames[f"WEAK{index}"] = frame

    context = _market_context_from_frames(frames)

    assert context["status"] == "FAIL"
    assert "market_breadth_breakdown" in context["flags"]


def test_bearish_market_context_does_not_fail_sell_candidates() -> None:
    context = {
        "status": "FAIL",
        "flags": ["market_breadth_breakdown", "market_momentum_negative"],
    }

    buy_quality = _market_context_quality_for_side(context, side="BUY")
    sell_quality = _market_context_quality_for_side(context, side="SELL")

    assert buy_quality["status"] == "FAIL"
    assert sell_quality["status"] == "PASS"
    assert sell_quality["flags"] == []


def test_event_risk_calendar_blocks_known_market_event() -> None:
    risk = evaluate_event_risk(asof_date=pd.Timestamp("2024-06-03").date(), symbol="RELIANCE")

    assert risk["status"] == "FAIL"
    assert "event_risk:election_result:market:2024-06-04" in risk["flags"]


def test_event_risk_can_be_disabled_by_override() -> None:
    risk = evaluate_event_risk(
        asof_date=pd.Timestamp("2024-06-03").date(),
        symbol="RELIANCE",
        overrides={"event_risk_enabled": False},
    )

    assert risk["status"] == "PASS"
    assert risk["flags"] == []


def test_event_risk_loads_manual_and_generated_calendars(tmp_path: Path) -> None:
    manual = tmp_path / "manual.csv"
    generated = tmp_path / "generated.csv"
    header = (
        "event_date,scope,symbol,event_type,severity,title,source,"
        "blackout_before_days,blackout_after_days\n"
    )
    manual.write_text(
        header + "2026-05-11,MARKET,,MACRO,WARN,Market event,manual,1,0\n",
        encoding="utf-8",
    )
    generated.write_text(
        header + "2026-05-12,SYMBOL,ABC,RESULTS,BLOCK,ABC results,generated,2,1\n",
        encoding="utf-8",
    )

    risk = evaluate_event_risk(
        asof_date=pd.Timestamp("2026-05-11").date(),
        symbol="ABC",
        overrides={
            "event_risk_calendar_path": str(manual),
            "event_risk_generated_calendar_path": str(generated),
        },
    )

    assert risk["status"] == "FAIL"
    assert "event_risk:macro:market:2026-05-11" in risk["flags"]
    assert "event_risk:results:ABC:2026-05-12" in risk["flags"]
