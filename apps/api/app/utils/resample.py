from __future__ import annotations


import pandas as pd


def parse_bar_windows(config: str) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for chunk in config.split(","):
        if "-" not in chunk:
            continue
        start, end = chunk.strip().split("-", maxsplit=1)
        windows.append((start, end))
    return windows


def resample_intraday_to_session_bars(
    intraday_df: pd.DataFrame,
    windows: list[tuple[str, str]],
    tz: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Resample intraday data into deterministic session windows like 09:15-13:15 and 13:15-15:30."""
    if intraday_df.empty:
        return intraday_df

    df = intraday_df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(tz)
    rows: list[dict[str, object]] = []

    for day, day_frame in df.groupby(df["datetime"].dt.date):
        for start_hm, end_hm in windows:
            start_dt = pd.Timestamp(f"{day} {start_hm}", tz=tz)
            end_dt = pd.Timestamp(f"{day} {end_hm}", tz=tz)
            bar = day_frame[(day_frame["datetime"] >= start_dt) & (day_frame["datetime"] < end_dt)]
            if bar.empty:
                continue

            rows.append(
                {
                    "datetime": start_dt.tz_convert("UTC"),
                    "open": float(bar.iloc[0]["open"]),
                    "high": float(bar["high"].max()),
                    "low": float(bar["low"].min()),
                    "close": float(bar.iloc[-1]["close"]),
                    "volume": float(bar["volume"].sum()),
                }
            )

    return pd.DataFrame(rows)
