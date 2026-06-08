"""
Performance and risk metrics computed from a daily equity curve.
All return-based metrics are dollar-return-based (not log).
"""

import numpy as np
import pandas as pd
from scipy import stats


TRADING_DAYS = 252


def compute_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series) -> float:
    n_years = len(equity) / TRADING_DAYS
    if n_years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1)


def sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS))


def sortino(returns: pd.Series, rf: float = 0.0) -> float:
    excess = returns - rf / TRADING_DAYS
    downside = excess[excess < 0]
    if len(downside) < 2:
        return np.nan
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return np.nan
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    rolling_max = equity.cummax()
    dd = equity / rolling_max - 1
    return float(dd.min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """Number of trading days in the longest drawdown."""
    rolling_max = equity.cummax()
    underwater = equity < rolling_max

    max_dur = 0
    cur_dur = 0
    for uw in underwater:
        if uw:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0
    return max_dur


def calmar(equity: pd.Series) -> float:
    dd = abs(max_drawdown(equity))
    if dd == 0:
        return np.nan
    return float(cagr(equity) / dd)


def market_beta(returns: pd.Series, bench_returns: pd.Series) -> float:
    r, b = returns.align(bench_returns, join="inner")
    if len(r) < 2:
        return np.nan
    cov_mat = np.cov(r, b)
    bench_var = cov_mat[1, 1]
    if bench_var == 0:
        return np.nan
    return float(cov_mat[0, 1] / bench_var)


def hit_rate(returns: pd.Series) -> float:
    wins = (returns > 0).sum()
    total = (returns != 0).sum()
    return float(wins / total) if total > 0 else np.nan


def profit_factor(returns: pd.Series) -> float:
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return float(gross_profit / gross_loss) if gross_loss > 0 else np.inf


def avg_win_loss_ratio(returns: pd.Series) -> float:
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(wins) == 0 or len(losses) == 0:
        return np.nan
    return float(wins.mean() / abs(losses.mean()))


def annualized_turnover(trade_log: pd.DataFrame, equity: pd.Series) -> float:
    """
    Rough annualized turnover: total notional traded / avg equity / n_years.
    """
    if trade_log.empty or equity.empty:
        return 0.0
    notional_traded = (trade_log["quantity"] * trade_log["fill_price"]).sum()
    avg_equity = equity.mean()
    n_years = len(equity) / TRADING_DAYS
    return float(notional_traded / avg_equity / n_years) if avg_equity > 0 and n_years > 0 else 0.0


def full_metrics(
    equity: pd.Series,
    trade_log: pd.DataFrame | None = None,
    bench_equity: pd.Series | None = None,
) -> dict:
    ret = compute_returns(equity)
    bench_ret = compute_returns(bench_equity) if bench_equity is not None else None

    result = {
        "total_return": total_return(equity),
        "cagr": cagr(equity),
        "sharpe": sharpe(ret),
        "sortino": sortino(ret),
        "max_drawdown": max_drawdown(equity),
        "max_drawdown_duration_days": max_drawdown_duration(equity),
        "calmar": calmar(equity),
        "hit_rate": hit_rate(ret),
        "profit_factor": profit_factor(ret),
        "avg_win_loss_ratio": avg_win_loss_ratio(ret),
        "skewness": float(stats.skew(ret)),
        "excess_kurtosis": float(stats.kurtosis(ret)),
    }

    if bench_ret is not None:
        result["market_beta"] = market_beta(ret, bench_ret)

    if trade_log is not None and not trade_log.empty:
        result["annualized_turnover"] = annualized_turnover(trade_log, equity)

    return result
