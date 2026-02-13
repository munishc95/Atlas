from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import get_settings
from app.db.session import engine
from app.engine.signal_engine import generate_signals_for_policy
from app.services.data_store import DataStore


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
