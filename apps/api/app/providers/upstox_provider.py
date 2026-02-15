from __future__ import annotations

from datetime import UTC, datetime, timedelta
import json
import time
from urllib import error, parse, request

import pandas as pd
from sqlmodel import Session, select

from app.core.config import Settings
from app.db.models import InstrumentMap
from app.providers.base import BaseProvider
from app.services.data_store import DataStore


def _utc_floor_date(value: datetime) -> str:
    return value.astimezone(UTC).date().isoformat()


def _parse_upstox_candle_rows(payload: object) -> list[list[object]]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, dict):
        candles = data.get("candles")
        if isinstance(candles, list):
            return [list(row) for row in candles if isinstance(row, (list, tuple))]
    if isinstance(data, list):
        return [list(row) for row in data if isinstance(row, (list, tuple))]
    return []


class UpstoxProvider(BaseProvider):
    kind = "UPSTOX"

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

    def refresh_mapping(self, symbols: list[str] | None = None) -> int:
        raw = self.settings.upstox_symbol_map_json
        if not raw:
            return 0
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return 0
        if not isinstance(parsed, dict):
            return 0
        target = {str(item).upper() for item in (symbols or []) if str(item).strip()}
        updated = 0
        now = datetime.now(UTC)
        for symbol_raw, instrument_key_raw in parsed.items():
            symbol = str(symbol_raw).strip().upper()
            instrument_key = str(instrument_key_raw).strip()
            if not symbol or not instrument_key:
                continue
            if target and symbol not in target:
                continue
            row = self.session.exec(
                select(InstrumentMap)
                .where(InstrumentMap.provider == self.kind)
                .where(InstrumentMap.symbol == symbol)
                .order_by(InstrumentMap.id.desc())
            ).first()
            if row is None:
                row = InstrumentMap(
                    provider=self.kind,
                    symbol=symbol,
                    instrument_key=instrument_key,
                    last_refreshed=now,
                )
            else:
                row.instrument_key = instrument_key
                row.last_refreshed = now
            self.session.add(row)
            updated += 1
        if updated > 0:
            self.session.commit()
        return updated

    def fetch_ohlc(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        tf = str(timeframe or "1d").strip().lower()
        if tf not in self.supports_timeframes():
            raise ValueError(f"Unsupported timeframe '{timeframe}' for {self.kind}")
        self.refresh_mapping(symbols=symbols)
        end_ts = end if isinstance(end, datetime) else datetime.now(UTC)
        start_ts = start if isinstance(start, datetime) else end_ts - timedelta(days=30)
        interval = "day"

        output: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            symbol_up = str(symbol).upper().strip()
            instrument_key = self._instrument_key_for_symbol(symbol_up)
            if not instrument_key:
                output[symbol_up] = pd.DataFrame(
                    columns=["datetime", "open", "high", "low", "close", "volume"]
                )
                continue
            payload = self._fetch_candles_payload(
                instrument_key=instrument_key,
                interval=interval,
                start_ts=start_ts,
                end_ts=end_ts,
            )
            rows = _parse_upstox_candle_rows(payload)
            output[symbol_up] = self._normalize_rows(rows)
            time.sleep(max(0.0, float(self.settings.upstox_throttle_seconds)))
        return output

    def _instrument_key_for_symbol(self, symbol: str) -> str | None:
        row = self.session.exec(
            select(InstrumentMap)
            .where(InstrumentMap.provider == self.kind)
            .where(InstrumentMap.symbol == symbol)
            .order_by(InstrumentMap.last_refreshed.desc(), InstrumentMap.id.desc())
        ).first()
        if row is None:
            return None
        return str(row.instrument_key or "").strip() or None

    def _fetch_candles_payload(
        self,
        *,
        instrument_key: str,
        interval: str,
        start_ts: datetime,
        end_ts: datetime,
    ) -> object:
        base = str(self.settings.upstox_base_url or "https://api.upstox.com").rstrip("/")
        path = (
            f"/v2/historical-candle/{parse.quote(instrument_key, safe='')}/{interval}/"
            f"{_utc_floor_date(end_ts)}/{_utc_floor_date(start_ts)}"
        )
        url = f"{base}{path}"
        headers = {
            "Accept": "application/json",
        }
        token = str(self.settings.upstox_access_token or "").strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        max_retries = max(0, int(self.settings.upstox_retry_max))
        backoff = max(0.1, float(self.settings.upstox_retry_backoff_seconds))

        for attempt in range(max_retries + 1):
            req = request.Request(url=url, headers=headers, method="GET")
            try:
                self.count_api_call(1)
                with request.urlopen(
                    req, timeout=float(self.settings.upstox_timeout_seconds)
                ) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    return json.loads(body)
            except error.HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempt < max_retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                raise
            except (error.URLError, TimeoutError, json.JSONDecodeError):
                if attempt < max_retries:
                    time.sleep(backoff * (2**attempt))
                    continue
                raise
        return {}

    def _normalize_rows(self, rows: list[list[object]]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        normalized: list[dict[str, object]] = []
        for row in rows:
            if len(row) < 6:
                continue
            ts_raw = row[0]
            ts = pd.to_datetime(ts_raw, utc=True, errors="coerce")
            if pd.isna(ts):
                continue
            normalized.append(
                {
                    "datetime": ts.to_pydatetime(),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )
        if not normalized:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(normalized).drop_duplicates(subset=["datetime"], keep="last")
        frame = frame.sort_values("datetime").reset_index(drop=True)
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        return frame
