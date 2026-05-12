from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date, datetime
import json
from pathlib import Path
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from sqlmodel import Session

ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.session import engine, init_db  # noqa: E402
from app.providers.nse_bhavcopy_provider import NseBhavcopyProvider  # noqa: E402
from app.services.data_provenance import confidence_for_provider, upsert_provenance_rows  # noqa: E402
from app.services.data_quality import run_data_quality_report  # noqa: E402
from app.services.data_store import DataStore  # noqa: E402
from app.services.trading_calendar import list_trading_days  # noqa: E402

IST_ZONE = ZoneInfo("Asia/Kolkata")


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _dedupe_symbols(values: list[str]) -> list[str]:
    symbols: list[str] = []
    seen: set[str] = set()
    for value in values:
        symbol = str(value).strip().lstrip("\ufeff").strip().upper()
        if not symbol or symbol.startswith("#"):
            continue
        if symbol not in seen:
            symbols.append(symbol)
            seen.add(symbol)
    return symbols


def _split_symbol_text(value: str | None) -> list[str]:
    if not value:
        return []
    tokens: list[str] = []
    for line in value.splitlines():
        clean = line.split("#", 1)[0].strip()
        if not clean:
            continue
        tokens.extend(part.strip() for part in clean.split(","))
    return _dedupe_symbols(tokens)


def _load_symbols_file(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if stripped.startswith("["):
        payload = json.loads(stripped)
        if not isinstance(payload, list):
            raise ValueError("--symbols-file JSON payload must be a list of symbols")
        return _dedupe_symbols([str(item) for item in payload])
    return _split_symbol_text(text)


def _store() -> DataStore:
    settings = get_settings()
    return DataStore(
        parquet_root=settings.parquet_root,
        duckdb_path=settings.duckdb_path,
        feature_cache_root=settings.feature_cache_root,
        adjustment_mode_default=settings.data_adjustment_mode,
        membership_mode_default=settings.universe_membership_mode,
    )


def _merge_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        base = incoming.copy()
    else:
        base = pd.concat([existing, incoming], ignore_index=True)
    base["datetime"] = pd.to_datetime(base["datetime"], utc=True)
    return (
        base.sort_values("datetime")
        .drop_duplicates(subset=["datetime"], keep="last")
        .reset_index(drop=True)
    )[["datetime", "open", "high", "low", "close", "volume"]]


def run_backfill(
    *,
    bundle_id: int,
    start_date: date,
    end_date: date,
    throttle_seconds: float,
    run_quality: bool,
    dry_run: bool,
    symbols: list[str] | None = None,
) -> dict[str, Any]:
    init_db()
    settings = get_settings()
    store = _store()
    segment = str(settings.trading_calendar_segment or "EQUITIES")
    trading_days = list_trading_days(
        start_date=start_date,
        end_date=end_date,
        segment=segment,
        settings=settings,
    )

    with Session(engine) as session:
        bundle_symbols = store.get_bundle_symbols(session, bundle_id, timeframe="1d")
        provider = NseBhavcopyProvider(session=session, settings=settings, store=store)
        bundle_symbol_set = {symbol.upper() for symbol in bundle_symbols}
        requested_symbols = _dedupe_symbols(symbols or [])
        skipped_requested_symbols = sorted(set(requested_symbols) - bundle_symbol_set)
        symbol_set = (
            {symbol for symbol in requested_symbols if symbol in bundle_symbol_set}
            if requested_symbols
            else bundle_symbol_set
        )
        if not symbol_set:
            raise ValueError(
                "No requested symbols matched the dataset bundle. "
                f"Skipped sample: {skipped_requested_symbols[:10]}"
            )
        rows_by_symbol: dict[str, list[pd.DataFrame]] = defaultdict(list)
        downloaded_days = 0
        matched_rows = 0
        empty_days: list[str] = []

        for index, day in enumerate(trading_days, start=1):
            day_frame = provider._day_frame(day)
            if day_frame.empty:
                empty_days.append(day.isoformat())
            else:
                downloaded_days += 1
                filtered = day_frame[day_frame["symbol"].isin(symbol_set)]
                matched_rows += int(len(filtered))
                for symbol, chunk in filtered.groupby("symbol"):
                    rows_by_symbol[str(symbol)].append(chunk.drop(columns=["symbol"]))
            if index % 25 == 0 or index == len(trading_days):
                print(
                    f"planned={len(trading_days)} processed={index} "
                    f"downloaded_days={downloaded_days} matched_rows={matched_rows}",
                    flush=True,
                )
            if index < len(trading_days):
                time.sleep(max(0.0, throttle_seconds))

        updated_symbols = 0
        added_rows_total = 0
        if not dry_run:
            for symbol in sorted(symbol_set):
                chunks = rows_by_symbol.get(symbol, [])
                if not chunks:
                    continue
                incoming = pd.concat(chunks, ignore_index=True)
                existing = store.load_ohlcv(symbol=symbol, timeframe="1d")
                before = int(len(existing))
                merged = _merge_frames(existing, incoming)
                added = int(len(merged) - before)
                store.save_ohlcv(
                    session=session,
                    symbol=symbol,
                    timeframe="1d",
                    frame=merged,
                    provider="NSE_BHAVCOPY_FREE",
                    bundle_id=bundle_id,
                )
                incoming_dates = (
                    pd.to_datetime(incoming["datetime"], utc=True, errors="coerce")
                    .dt.tz_convert(IST_ZONE)
                    .dt.date
                )
                bar_dates = [day for day in sorted(set(incoming_dates.tolist())) if day is not None]
                if bar_dates:
                    upsert_provenance_rows(
                        session,
                        bundle_id=int(bundle_id),
                        timeframe="1d",
                        symbol=symbol,
                        bar_dates=bar_dates,
                        source_provider="NSE_BHAVCOPY",
                        source_run_kind="nse_bhavcopy_backfill",
                        source_run_id=None,
                        confidence_score=confidence_for_provider(
                            provider="NSE_BHAVCOPY",
                            settings=settings,
                        ),
                        reason="free_nse_bhavcopy_backfill",
                    )
                updated_symbols += 1
                added_rows_total += max(0, added)
                if updated_symbols % 50 == 0:
                    print(
                        f"saved_symbols={updated_symbols} added_rows_total={added_rows_total}",
                        flush=True,
                    )
            session.commit()

        quality: dict[str, Any] | None = None
        if run_quality and not dry_run:
            report = run_data_quality_report(
                session=session,
                settings=settings,
                store=store,
                bundle_id=bundle_id,
                timeframe="1d",
            )
            quality = {
                "id": report.id,
                "status": report.status,
                "coverage_pct": report.coverage_pct,
                "last_bar_ts": report.last_bar_ts.isoformat() if report.last_bar_ts else None,
                "checked_symbols": report.checked_symbols,
                "total_symbols": report.total_symbols,
            }

    return {
        "bundle_id": bundle_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "trading_days_planned": len(trading_days),
        "downloaded_days": downloaded_days,
        "empty_days_count": len(empty_days),
        "empty_days_sample": empty_days[:10],
        "target_symbols": len(symbol_set),
        "target_symbol_sample": sorted(symbol_set)[:20],
        "requested_symbols": len(requested_symbols),
        "skipped_requested_symbols_count": len(skipped_requested_symbols),
        "skipped_requested_symbols_sample": skipped_requested_symbols[:20],
        "symbols_with_rows": len(rows_by_symbol),
        "matched_rows": matched_rows,
        "updated_symbols": updated_symbols,
        "added_rows_total": added_rows_total,
        "dry_run": dry_run,
        "quality": quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Free NSE bhavcopy daily EOD backfill into an Atlas dataset bundle."
    )
    parser.add_argument("--bundle-id", type=int, required=True)
    parser.add_argument("--start-date", type=_parse_day, default=date(2020, 1, 1))
    parser.add_argument("--end-date", type=_parse_day, default=date.today())
    parser.add_argument(
        "--symbols",
        default=None,
        help="Optional comma-separated target symbols. Defaults to every bundle symbol.",
    )
    parser.add_argument(
        "--symbols-file",
        type=Path,
        default=None,
        help="Optional newline/comma-separated or JSON-list file of target symbols.",
    )
    parser.add_argument("--throttle-seconds", type=float, default=0.15)
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_symbols = _split_symbol_text(args.symbols)
    if args.symbols_file:
        target_symbols.extend(_load_symbols_file(args.symbols_file))
    target_symbols = _dedupe_symbols(target_symbols)

    result = run_backfill(
        bundle_id=int(args.bundle_id),
        start_date=args.start_date,
        end_date=args.end_date,
        throttle_seconds=float(args.throttle_seconds),
        run_quality=not bool(args.skip_quality),
        dry_run=bool(args.dry_run),
        symbols=target_symbols or None,
    )
    print(result)


if __name__ == "__main__":
    main()
