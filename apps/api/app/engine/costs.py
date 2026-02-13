from __future__ import annotations

from typing import Any


DEFAULT_COSTS: dict[str, float] = {
    "brokerage_bps": 0.0,
    "stt_delivery_buy_bps": 0.0,
    "stt_delivery_sell_bps": 10.0,
    "stt_intraday_buy_bps": 0.0,
    "stt_intraday_sell_bps": 2.5,
    "exchange_txn_bps": 0.297,
    "sebi_bps": 0.001,
    "stamp_delivery_buy_bps": 1.5,
    "stamp_intraday_buy_bps": 0.3,
    "gst_rate": 0.18,
    "futures_brokerage_bps": 0.0,
    "futures_stt_sell_bps": 1.0,
    "futures_exchange_txn_bps": 0.19,
    "futures_stamp_buy_bps": 0.0,
}


def _settings(overrides: dict[str, Any] | None) -> dict[str, float]:
    merged = dict(DEFAULT_COSTS)
    if not isinstance(overrides, dict):
        return merged
    for key, value in overrides.items():
        if key in merged and isinstance(value, (int, float)):
            merged[key] = float(value)
    return merged


def estimate_equity_delivery_cost(
    notional: float,
    side: str,
    config: dict[str, Any] | None = None,
) -> float:
    if notional <= 0:
        return 0.0
    cfg = _settings(config)
    side_norm = str(side).upper()

    brokerage = notional * cfg["brokerage_bps"] / 10_000
    exchange = notional * cfg["exchange_txn_bps"] / 10_000
    sebi = notional * cfg["sebi_bps"] / 10_000
    stt_bps = cfg["stt_delivery_sell_bps"] if side_norm == "SELL" else cfg["stt_delivery_buy_bps"]
    stamp_bps = cfg["stamp_delivery_buy_bps"] if side_norm == "BUY" else 0.0
    stt = notional * stt_bps / 10_000
    stamp = notional * stamp_bps / 10_000
    gst = cfg["gst_rate"] * (brokerage + exchange)

    return max(0.0, brokerage + exchange + sebi + stt + stamp + gst)


def estimate_intraday_cost(
    notional: float,
    side: str,
    config: dict[str, Any] | None = None,
) -> float:
    if notional <= 0:
        return 0.0
    cfg = _settings(config)
    side_norm = str(side).upper()

    brokerage = notional * cfg["brokerage_bps"] / 10_000
    exchange = notional * cfg["exchange_txn_bps"] / 10_000
    sebi = notional * cfg["sebi_bps"] / 10_000
    stt_bps = cfg["stt_intraday_sell_bps"] if side_norm == "SELL" else cfg["stt_intraday_buy_bps"]
    stamp_bps = cfg["stamp_intraday_buy_bps"] if side_norm == "BUY" else 0.0
    stt = notional * stt_bps / 10_000
    stamp = notional * stamp_bps / 10_000
    gst = cfg["gst_rate"] * (brokerage + exchange)

    return max(0.0, brokerage + exchange + sebi + stt + stamp + gst)


def estimate_futures_cost(
    notional: float,
    side: str,
    config: dict[str, Any] | None = None,
) -> float:
    if notional <= 0:
        return 0.0
    cfg = _settings(config)
    side_norm = str(side).upper()

    brokerage_bps = cfg.get("futures_brokerage_bps", cfg["brokerage_bps"])
    exchange_bps = cfg.get("futures_exchange_txn_bps", cfg["exchange_txn_bps"])
    stt_sell_bps = cfg.get("futures_stt_sell_bps", 1.0)
    stamp_buy_bps = cfg.get("futures_stamp_buy_bps", 0.0)

    brokerage = notional * brokerage_bps / 10_000
    exchange = notional * exchange_bps / 10_000
    sebi = notional * cfg["sebi_bps"] / 10_000
    stt = notional * (stt_sell_bps / 10_000) if side_norm == "SELL" else 0.0
    stamp = notional * (stamp_buy_bps / 10_000) if side_norm == "BUY" else 0.0
    gst = cfg["gst_rate"] * (brokerage + exchange)
    return max(0.0, brokerage + exchange + sebi + stt + stamp + gst)
