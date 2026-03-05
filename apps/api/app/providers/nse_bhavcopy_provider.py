from __future__ import annotations

from datetime import UTC, date as dt_date, datetime, time as dt_time, timedelta
import hashlib
import io
from pathlib import Path
import random
import time
from typing import Any
from urllib import error, request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlmodel import Session

from app.core.config import Settings
from app.providers.base import BaseProvider
from app.services.data_store import DataStore
from app.services.fast_mode import fast_mode_enabled
from app.services.trading_calendar import list_trading_days

IST_ZONE = ZoneInfo("Asia/Kolkata")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])


def _safe_time(value: str | None, fallback: str = "15:30") -> dt_time:
    raw = str(value or fallback).strip()
    try:
        hour, minute = raw.split(":", maxsplit=1)
        return dt_time(hour=max(0, min(23, int(hour))), minute=max(0, min(59, int(minute))))
    except Exception:  # noqa: BLE001
        return dt_time(15, 30)


def _normalize_column_token(value: Any) -> str:
    return "".join(ch for ch in str(value or "").upper() if ch.isalnum())


def _first_existing(mapping: dict[str, str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        token = _normalize_column_token(candidate)
        if token in mapping:
            return mapping[token]
    return None


class NseBhavcopyProvider(BaseProvider):
    """NSE bhavcopy-style EOD provider (all symbols/day)."""

    kind = "NSE_BHAVCOPY"

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
        self._cache_root = Path(settings.nse_bhavcopy_cache_dir)
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._session_close = _safe_time(settings.nse_equities_close_time_ist, "15:30")
        self._series_filter = str(settings.nse_bhavcopy_series_filter or "EQ").strip().upper()

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
            raise ValueError(f"NSE_BHAVCOPY only supports 1d timeframe, got '{timeframe}'")

        symbol_tokens = sorted({str(item).strip().upper() for item in symbols if str(item).strip()})
        if not symbol_tokens:
            return {}

        end_ts = end if isinstance(end, datetime) else datetime.now(UTC)
        start_ts = start if isinstance(start, datetime) else end_ts - timedelta(days=120)
        if end_ts.tzinfo is None:
            end_ts = end_ts.replace(tzinfo=UTC)
        else:
            end_ts = end_ts.astimezone(UTC)
        if start_ts.tzinfo is None:
            start_ts = start_ts.replace(tzinfo=UTC)
        else:
            start_ts = start_ts.astimezone(UTC)
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts

        trading_days = list_trading_days(
            start_date=start_ts.astimezone(IST_ZONE).date(),
            end_date=end_ts.astimezone(IST_ZONE).date(),
            segment=str(self.settings.trading_calendar_segment or "EQUITIES"),
            settings=self.settings,
        )
        if not trading_days:
            return {symbol: _empty_frame() for symbol in symbol_tokens}

        max_days = max(1, int(self.settings.nse_bhavcopy_max_trading_days_per_call))
        if len(trading_days) > max_days:
            trading_days = trading_days[-max_days:]

        if fast_mode_enabled(self.settings):
            return self._fast_mode_frames(symbol_tokens=symbol_tokens, trading_days=trading_days)

        rows_by_symbol: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbol_tokens}
        for idx, day in enumerate(trading_days):
            day_frame = self._day_frame(day)
            if not day_frame.empty:
                filtered = day_frame[day_frame["symbol"].isin(symbol_tokens)]
                for symbol, chunk in filtered.groupby("symbol"):
                    rows_by_symbol[str(symbol)].append(chunk.drop(columns=["symbol"]))
            if idx < len(trading_days) - 1:
                time.sleep(max(0.0, float(self.settings.nse_bhavcopy_throttle_seconds)))

        output: dict[str, pd.DataFrame] = {}
        for symbol in symbol_tokens:
            chunks = rows_by_symbol.get(symbol, [])
            if not chunks:
                output[symbol] = _empty_frame()
                continue
            merged = (
                pd.concat(chunks, ignore_index=True)
                .sort_values("datetime")
                .drop_duplicates(subset=["datetime"], keep="last")
                .reset_index(drop=True)
            )
            output[symbol] = merged[["datetime", "open", "high", "low", "close", "volume"]]
        return output

    def _fast_mode_frames(
        self,
        *,
        symbol_tokens: list[str],
        trading_days: list[dt_date],
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbol_tokens:
            seed = int(self.settings.fast_mode_seed)
            rows: list[dict[str, Any]] = []
            for day in trading_days:
                day_seed = int(
                    hashlib.sha1(f"{seed}:{symbol}:{day.isoformat()}".encode("utf-8")).hexdigest()[:12],
                    16,
                )
                rng = np.random.default_rng(day_seed)
                anchor = 100.0 + float(rng.uniform(-12.0, 18.0))
                seasonal = float((day.toordinal() % 251) * 0.08)
                close = anchor + seasonal
                open_px = close * (1.0 + float(rng.uniform(-0.01, 0.01)))
                high = max(open_px, close) * (1.0 + float(rng.uniform(0.001, 0.012)))
                low = min(open_px, close) * (1.0 - float(rng.uniform(0.001, 0.012)))
                volume = float(rng.integers(200_000, 3_000_000))
                rows.append(
                    {
                        "day": day,
                        "open": float(open_px),
                        "high": float(high),
                        "low": float(low),
                        "close": float(close),
                        "volume": float(volume),
                    }
                )
            dt_values = [
                datetime(
                    day.year,
                    day.month,
                    day.day,
                    self._session_close.hour,
                    self._session_close.minute,
                    tzinfo=IST_ZONE,
                ).astimezone(UTC)
                for day in trading_days
            ]
            open_values = np.array([float(item["open"]) for item in rows], dtype=float)
            high_values = np.array([float(item["high"]) for item in rows], dtype=float)
            low_values = np.array([float(item["low"]) for item in rows], dtype=float)
            close_values = np.array([float(item["close"]) for item in rows], dtype=float)
            volume_values = np.array([float(item["volume"]) for item in rows], dtype=float)
            out[symbol] = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(dt_values, utc=True),
                    "open": open_values,
                    "high": high_values,
                    "low": low_values,
                    "close": close_values,
                    "volume": volume_values,
                }
            )
        return out

    def _cache_path(self, day: dt_date) -> Path:
        return self._cache_root / f"{day.strftime('%Y%m%d')}.csv"

    def _render_path(self, day: dt_date) -> str:
        template = str(
            self.settings.nse_bhavcopy_path_template
            or "/products/content/sec_bhavdata_full_{DDMMYYYY}.csv"
        )
        replacements = {
            "{DDMMYYYY}": day.strftime("%d%m%Y"),
            "{YYYYMMDD}": day.strftime("%Y%m%d"),
            "{YYYY}": day.strftime("%Y"),
            "{MM}": day.strftime("%m"),
            "{DD}": day.strftime("%d"),
        }
        value = template
        for key, replacement in replacements.items():
            value = value.replace(key, replacement)
        return value

    def _build_url(self, day: dt_date) -> str:
        base = str(self.settings.nse_bhavcopy_base_url or "https://nsearchives.nseindia.com").rstrip("/")
        path = self._render_path(day).strip()
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    def _day_frame(self, day: dt_date) -> pd.DataFrame:
        cache_path = self._cache_path(day)
        raw: bytes | None = None
        if cache_path.exists() and cache_path.stat().st_size > 0:
            raw = cache_path.read_bytes()
        else:
            raw = self._download_day(day)
            if raw:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(raw)
        if not raw:
            return pd.DataFrame(
                columns=["symbol", "datetime", "open", "high", "low", "close", "volume"]
            )
        parsed = self._parse_day_csv(raw=raw, day=day)
        if parsed.empty:
            # Corrupt cache can happen after interrupted writes. Delete and let future runs refetch.
            try:
                if cache_path.exists():
                    cache_path.unlink()
            except OSError:
                pass
        return parsed

    def _download_day(self, day: dt_date) -> bytes | None:
        max_retries = max(0, int(self.settings.nse_bhavcopy_retry_max))
        backoff = max(0.1, float(self.settings.nse_bhavcopy_retry_backoff_seconds))
        timeout = max(1.0, float(self.settings.nse_bhavcopy_timeout_seconds))
        url = self._build_url(day)
        headers = {
            "Accept": "text/csv,application/octet-stream,*/*",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": "https://www.nseindia.com/",
            "Accept-Language": "en-US,en;q=0.9",
        }
        for attempt in range(max_retries + 1):
            req = request.Request(url=url, headers=headers, method="GET")
            try:
                self.count_api_call(1)
                with request.urlopen(req, timeout=timeout) as response:
                    body = response.read()
                if body:
                    return body
            except error.HTTPError as exc:
                if exc.code in {404}:
                    return None
                if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    sleep_for = backoff * (2**attempt) + random.uniform(0.0, 0.35)
                    time.sleep(sleep_for)
                    continue
                return None
            except (error.URLError, TimeoutError, OSError):
                if attempt < max_retries:
                    sleep_for = backoff * (2**attempt) + random.uniform(0.0, 0.35)
                    time.sleep(sleep_for)
                    continue
                return None
        return None

    def _parse_day_csv(self, *, raw: bytes, day: dt_date) -> pd.DataFrame:
        try:
            frame = pd.read_csv(io.BytesIO(raw), dtype=str)
        except Exception:  # noqa: BLE001
            try:
                text = raw.decode("utf-8", errors="replace")
                frame = pd.read_csv(io.StringIO(text), dtype=str)
            except Exception:  # noqa: BLE001
                return pd.DataFrame(
                    columns=["symbol", "datetime", "open", "high", "low", "close", "volume"]
                )
        if frame.empty:
            return pd.DataFrame(
                columns=["symbol", "datetime", "open", "high", "low", "close", "volume"]
            )

        column_map = {_normalize_column_token(column): str(column) for column in frame.columns}
        symbol_col = _first_existing(column_map, ["SYMBOL", "SYMB"])
        series_col = _first_existing(column_map, ["SERIES"])
        open_col = _first_existing(column_map, ["OPEN", "OPENPRICE"])
        high_col = _first_existing(column_map, ["HIGH", "HIGHPRICE"])
        low_col = _first_existing(column_map, ["LOW", "LOWPRICE"])
        close_col = _first_existing(column_map, ["CLOSE", "CLOSEPRICE", "LAST"])
        volume_col = _first_existing(column_map, ["TOTTRDQTY", "TOTTRD_QTY", "VOLUME"])
        date_col = _first_existing(column_map, ["TIMESTAMP", "DATE1", "DATE", "TRADINGDATE"])
        required = [symbol_col, open_col, high_col, low_col, close_col, volume_col]
        if any(column is None for column in required):
            return pd.DataFrame(
                columns=["symbol", "datetime", "open", "high", "low", "close", "volume"]
            )

        base = frame.copy()
        base["symbol"] = base[str(symbol_col)].astype(str).str.strip().str.upper()
        base = base[base["symbol"].str.len() > 0]
        if series_col is not None and self._series_filter:
            series = base[str(series_col)].astype(str).str.strip().str.upper()
            base = base[series == self._series_filter]
        if base.empty:
            return pd.DataFrame(
                columns=["symbol", "datetime", "open", "high", "low", "close", "volume"]
            )

        for target, source in (
            ("open", str(open_col)),
            ("high", str(high_col)),
            ("low", str(low_col)),
            ("close", str(close_col)),
            ("volume", str(volume_col)),
        ):
            base[target] = pd.to_numeric(base[source], errors="coerce")
        base = base.dropna(subset=["open", "high", "low", "close", "volume"])
        base = base[(base["high"] >= base["low"]) & (base["open"].between(base["low"], base["high"]))]
        base = base[base["close"].between(base["low"], base["high"])]
        if base.empty:
            return pd.DataFrame(
                columns=["symbol", "datetime", "open", "high", "low", "close", "volume"]
            )

        if date_col is not None:
            parsed_date = pd.to_datetime(base[str(date_col)], errors="coerce", dayfirst=True)
            base["bar_date"] = parsed_date.dt.date
            base.loc[base["bar_date"].isna(), "bar_date"] = day
        else:
            base["bar_date"] = day

        close_hour = int(self._session_close.hour)
        close_minute = int(self._session_close.minute)

        def _to_utc_timestamp(value: Any) -> datetime:
            bar_day = value if isinstance(value, dt_date) else day
            return datetime(
                bar_day.year,
                bar_day.month,
                bar_day.day,
                close_hour,
                close_minute,
                tzinfo=IST_ZONE,
            ).astimezone(UTC)

        base["datetime"] = base["bar_date"].map(_to_utc_timestamp)
        out = base[["symbol", "datetime", "open", "high", "low", "close", "volume"]].copy()
        out["datetime"] = pd.to_datetime(out["datetime"], utc=True, errors="coerce")
        out = out.dropna(subset=["datetime"])
        out = out.sort_values(["symbol", "datetime"]).drop_duplicates(
            subset=["symbol", "datetime"],
            keep="last",
        )
        return out.reset_index(drop=True)
