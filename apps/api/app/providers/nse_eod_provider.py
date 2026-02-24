from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import random
import time
from urllib import error, parse, request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import Settings
from app.providers.base import BaseProvider
from app.services.data_store import DataStore

IST_ZONE = ZoneInfo("Asia/Kolkata")


def _format_nse_day(value: datetime) -> str:
    return value.astimezone(IST_ZONE).strftime("%d-%m-%Y")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


class NseEodProvider(BaseProvider):
    """NSE EOD provider for 1d bars with cautious throttling/retry semantics."""

    kind = "NSE_EOD"

    def __init__(
        self,
        *,
        session: Session,
        settings: Settings,
        store: DataStore,
    ) -> None:
        super().__init__()
        self.session = session
        self.settings = settings
        self.store = store

    def supports_timeframes(self) -> set[str]:
        return {"1d"}

    def list_symbols(self, bundle_id: int) -> list[str]:
        return self.store.get_bundle_symbols(self.session, int(bundle_id), timeframe="1d")

    def fetch_ohlc(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        tf = str(timeframe or "1d").strip().lower()
        if tf != "1d":
            raise ValueError(f"NSE EOD only supports 1d timeframe, got '{timeframe}'")

        output: dict[str, pd.DataFrame] = {}
        end_ts = end if isinstance(end, datetime) else datetime.now(UTC)
        start_ts = start if isinstance(start, datetime) else end_ts - timedelta(days=30)

        for symbol in symbols:
            symbol_up = str(symbol).strip().upper()
            if not symbol_up:
                continue
            try:
                if self.settings.fast_mode_enabled:
                    frame = self._fast_mode_frame(
                        symbol=symbol_up,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
                else:
                    frame = self._fetch_symbol_http(
                        symbol=symbol_up,
                        start_ts=start_ts,
                        end_ts=end_ts,
                    )
            except Exception:  # noqa: BLE001
                frame = _empty_frame()
            output[symbol_up] = frame
            time.sleep(max(0.0, float(self.settings.nse_eod_throttle_seconds)))
        return output

    def _fast_mode_frame(
        self,
        *,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> pd.DataFrame:
        start = start_ts.astimezone(UTC).date()
        end = end_ts.astimezone(UTC).date()
        if end < start:
            start, end = end, start
        days = pd.date_range(start=start, end=end, freq="D", tz="UTC")
        if len(days) == 0:
            days = pd.date_range(end=end_ts.astimezone(UTC), periods=6, freq="D", tz="UTC")
        rng = np.random.default_rng(abs(hash(f"{self.settings.fast_mode_seed}:{symbol}")) % 1_000_000)
        base = 100.0 + float(rng.uniform(-10.0, 10.0))
        returns = rng.normal(0.0008, 0.01, size=len(days))
        close = base * np.cumprod(1.0 + returns)
        open_px = np.concatenate([[close[0]], close[:-1]])
        high = np.maximum(open_px, close) * (1.0 + rng.uniform(0.0005, 0.01, size=len(days)))
        low = np.minimum(open_px, close) * (1.0 - rng.uniform(0.0005, 0.01, size=len(days)))
        volume = rng.integers(200_000, 2_000_000, size=len(days))
        return pd.DataFrame(
            {
                "datetime": pd.to_datetime(days, utc=True),
                "open": open_px.astype(float),
                "high": high.astype(float),
                "low": low.astype(float),
                "close": close.astype(float),
                "volume": volume.astype(float),
            }
        )

    def _fetch_symbol_http(
        self,
        *,
        symbol: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> pd.DataFrame:
        base = str(self.settings.nse_eod_base_url or "https://www.nseindia.com").rstrip("/")
        query = parse.urlencode(
            {
                "symbol": symbol,
                "series": '["EQ"]',
                "from": _format_nse_day(start_ts),
                "to": _format_nse_day(end_ts),
            }
        )
        url = f"{base}/api/historical/cm/equity?{query}"
        headers = {
            "Accept": "application/json,text/plain,*/*",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nseindia.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }
        max_retries = max(0, int(self.settings.nse_eod_retry_max))
        backoff = max(0.1, float(self.settings.nse_eod_retry_backoff_seconds))

        for attempt in range(max_retries + 1):
            req = request.Request(url=url, headers=headers, method="GET")
            try:
                self.count_api_call(1)
                with request.urlopen(req, timeout=float(self.settings.nse_eod_timeout_seconds)) as response:
                    body = response.read().decode("utf-8", errors="replace")
                payload = json.loads(body)
                return self._parse_payload(payload)
            except error.HTTPError as exc:
                if exc.code in {401, 403, 404}:
                    return _empty_frame()
                if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    sleep_for = backoff * (2**attempt) + random.uniform(0.0, 0.25)
                    time.sleep(sleep_for)
                    continue
                return _empty_frame()
            except (error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt < max_retries:
                    sleep_for = backoff * (2**attempt) + random.uniform(0.0, 0.25)
                    time.sleep(sleep_for)
                    continue
                return _empty_frame()
        return _empty_frame()

    def _parse_payload(self, payload: object) -> pd.DataFrame:
        if not isinstance(payload, dict):
            return _empty_frame()
        rows = payload.get("data")
        if not isinstance(rows, list):
            return _empty_frame()
        normalized: list[dict[str, float | datetime]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            date_raw = row.get("CH_TIMESTAMP") or row.get("timestamp") or row.get("date")
            open_raw = row.get("CH_OPENING_PRICE") or row.get("open")
            high_raw = row.get("CH_TRADE_HIGH_PRICE") or row.get("high")
            low_raw = row.get("CH_TRADE_LOW_PRICE") or row.get("low")
            close_raw = row.get("CH_CLOSING_PRICE") or row.get("close")
            vol_raw = row.get("CH_TOT_TRADED_QTY") or row.get("volume") or 0
            ts = pd.to_datetime(date_raw, errors="coerce", utc=False)
            if pd.isna(ts):
                continue
            try:
                ts_utc = ts.tz_localize(IST_ZONE).tz_convert(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
                normalized.append(
                    {
                        "datetime": ts_utc.to_pydatetime(),
                        "open": float(open_raw),
                        "high": float(high_raw),
                        "low": float(low_raw),
                        "close": float(close_raw),
                        "volume": float(vol_raw),
                    }
                )
            except (TypeError, ValueError):
                continue
        if not normalized:
            return _empty_frame()
        frame = pd.DataFrame(normalized)
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["datetime"]).sort_values("datetime")
        frame = frame.drop_duplicates(subset=["datetime"], keep="last").reset_index(drop=True)
        return frame[["datetime", "open", "high", "low", "close", "volume"]]

