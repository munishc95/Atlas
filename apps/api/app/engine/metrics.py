from __future__ import annotations

from math import sqrt

import pandas as pd


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def calculate_metrics(
    equity: pd.Series,
    trades: pd.DataFrame,
    open_position_count: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    if equity.empty:
        return {
            "cagr": 0.0,
            "max_drawdown": 0.0,
            "calmar": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "exposure_pct": 0.0,
            "turnover": 0.0,
            "avg_holding_period_bars": 0.0,
            "cvar_95": 0.0,
            "tail_loss_norm": 1.0,
        }

    returns = equity.pct_change().dropna()
    cumulative = equity / equity.iloc[0]
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    years = len(returns) / periods_per_year if len(returns) > 0 else 0
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    calmar = _safe_div(cagr, abs(max_drawdown)) if max_drawdown < 0 else 0.0

    mean_ret = float(returns.mean()) if not returns.empty else 0.0
    std_ret = float(returns.std(ddof=0)) if not returns.empty else 0.0
    sharpe = _safe_div(mean_ret, std_ret) * sqrt(periods_per_year) if std_ret > 0 else 0.0

    downside = returns[returns < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    sortino = (
        _safe_div(mean_ret, downside_std) * sqrt(periods_per_year) if downside_std > 0 else 0.0
    )

    wins = trades[trades["pnl"] > 0] if not trades.empty else pd.DataFrame()
    losses = trades[trades["pnl"] <= 0] if not trades.empty else pd.DataFrame()

    win_rate = float(len(wins) / len(trades)) if not trades.empty else 0.0
    avg_win = float(wins["pnl"].mean()) if not wins.empty else 0.0
    avg_loss = float(losses["pnl"].mean()) if not losses.empty else 0.0
    gross_win = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(losses["pnl"].sum()) if not losses.empty else 0.0
    profit_factor = _safe_div(gross_win, abs(gross_loss)) if gross_loss < 0 else 0.0

    exposure_pct = (
        float((open_position_count > 0).mean() * 100) if not open_position_count.empty else 0.0
    )
    turnover = float(trades["notional"].sum() / equity.mean()) if not trades.empty else 0.0
    avg_holding_period = float(trades["holding_bars"].mean()) if not trades.empty else 0.0

    if returns.empty:
        cvar_95 = 0.0
    else:
        var_95 = float(returns.quantile(0.05))
        tail = returns[returns <= var_95]
        cvar_95 = float(tail.mean()) if not tail.empty else var_95

    tail_loss_norm = min(1.0, max(0.0, abs(cvar_95) / 0.05))

    return {
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "exposure_pct": exposure_pct,
        "turnover": turnover,
        "avg_holding_period_bars": avg_holding_period,
        "cvar_95": cvar_95,
        "tail_loss_norm": tail_loss_norm,
    }
