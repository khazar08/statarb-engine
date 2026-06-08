"""
Tearsheet generator — produces a multi-panel figure and prints a summary table.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter

from statarb.analytics.metrics import (
    compute_returns, cagr, sharpe, sortino, calmar,
    max_drawdown, market_beta, full_metrics,
)


def _rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    mu = returns.rolling(window).mean()
    sigma = returns.rolling(window).std(ddof=1)
    return (mu / sigma * np.sqrt(252)).rename("rolling_sharpe")


def _drawdown_series(equity: pd.Series) -> pd.Series:
    return (equity / equity.cummax() - 1).rename("drawdown")


def plot_tearsheet(
    equity: pd.Series,
    bench_equity: pd.Series | None = None,
    trade_log: pd.DataFrame | None = None,
    title: str = "Strategy",
    save_path: str | None = None,
) -> plt.Figure:
    returns = compute_returns(equity)
    bench_ret = compute_returns(bench_equity) if bench_equity is not None else None

    fig = plt.figure(figsize=(14, 18))
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Equity curve
    ax_eq = fig.add_subplot(gs[0, :])
    normed = equity / equity.iloc[0]
    ax_eq.plot(normed.index, normed.values, label="Strategy", color="steelblue", lw=1.5)
    if bench_equity is not None:
        normed_b = bench_equity / bench_equity.iloc[0]
        normed_b = normed_b.reindex(normed.index, method="ffill")
        ax_eq.plot(normed_b.index, normed_b.values, label="SPY", color="gray",
                   lw=1, linestyle="--", alpha=0.7)
    ax_eq.set_title(f"{title} — Equity Curve (normalised to 1)", fontsize=11)
    ax_eq.legend(fontsize=9)
    ax_eq.xaxis.set_major_formatter(DateFormatter("%Y"))
    ax_eq.set_ylabel("Normalised value")

    # 2. Drawdown
    ax_dd = fig.add_subplot(gs[1, :])
    dd = _drawdown_series(equity)
    ax_dd.fill_between(dd.index, dd.values * 100, 0, color="tomato", alpha=0.6)
    ax_dd.set_title("Drawdown (%)", fontsize=11)
    ax_dd.set_ylabel("%")
    ax_dd.xaxis.set_major_formatter(DateFormatter("%Y"))

    # 3. Rolling Sharpe (63-day)
    ax_rs = fig.add_subplot(gs[2, :])
    rs = _rolling_sharpe(returns, window=63)
    ax_rs.plot(rs.index, rs.values, color="darkorange", lw=1)
    ax_rs.axhline(0, color="black", lw=0.8, linestyle="--")
    ax_rs.set_title("Rolling 63-day Sharpe (annualised)", fontsize=11)
    ax_rs.xaxis.set_major_formatter(DateFormatter("%Y"))

    # 4. Return distribution
    ax_dist = fig.add_subplot(gs[3, 0])
    ax_dist.hist(returns.values * 100, bins=60, color="steelblue", edgecolor="white",
                 alpha=0.8, density=True)
    ax_dist.set_title("Daily Return Distribution (%)", fontsize=11)
    ax_dist.set_xlabel("%")

    # 5. Monthly returns heatmap
    ax_monthly = fig.add_subplot(gs[3, 1])
    try:
        monthly = returns.resample("ME").apply(lambda r: (1 + r).prod() - 1) * 100
        pivot = monthly.copy()
        pivot.index = pd.MultiIndex.from_arrays(
            [pivot.index.year, pivot.index.month]
        )
        pivot = pivot.unstack(level=1)
        im = ax_monthly.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                               vmin=-10, vmax=10)
        ax_monthly.set_yticks(range(len(pivot.index)))
        ax_monthly.set_yticklabels(pivot.index, fontsize=7)
        ax_monthly.set_xticks(range(12))
        ax_monthly.set_xticklabels(["J","F","M","A","M","J","J","A","S","O","N","D"],
                                    fontsize=7)
        ax_monthly.set_title("Monthly Returns (%)", fontsize=11)
        plt.colorbar(im, ax=ax_monthly, fraction=0.04)
    except Exception:
        ax_monthly.text(0.5, 0.5, "Insufficient data", ha="center", va="center")

    # 6. Per-pair P&L (if trade log available)
    ax_pair = fig.add_subplot(gs[4, :])
    if trade_log is not None and not trade_log.empty and "pair_id" in trade_log.columns:
        pair_pnl = _compute_pair_pnl(trade_log)
        colors = ["steelblue" if v >= 0 else "tomato" for v in pair_pnl.values]
        ax_pair.bar(pair_pnl.index, pair_pnl.values, color=colors)
        ax_pair.set_title("P&L Attribution by Pair ($)", fontsize=11)
        ax_pair.tick_params(axis="x", rotation=45, labelsize=7)
    else:
        ax_pair.axis("off")
        ax_pair.text(0.5, 0.5, "No pair-level trade log", ha="center", va="center")

    plt.suptitle(f"{title} — Full Tearsheet", fontsize=13, y=1.01)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig


def _compute_pair_pnl(trade_log: pd.DataFrame) -> pd.Series:
    rows = []
    for _, row in trade_log.iterrows():
        sign = -1 if row["direction"] == "BUY" else 1
        pnl = sign * row["quantity"] * row["fill_price"]
        pnl -= row.get("commission", 0) + row.get("spread_cost", 0) + row.get("slippage_cost", 0)
        rows.append({"pair_id": row["pair_id"], "pnl": pnl})
    df = pd.DataFrame(rows)
    return df.groupby("pair_id")["pnl"].sum().sort_values()


def print_summary(metrics: dict, dsr_result: dict | None = None) -> None:
    lines = [
        ("Total Return",      f"{metrics.get('total_return', 0)*100:.1f}%"),
        ("CAGR",              f"{metrics.get('cagr', 0)*100:.1f}%"),
        ("Sharpe Ratio",      f"{metrics.get('sharpe', 0):.3f}"),
        ("Sortino Ratio",     f"{metrics.get('sortino', 0):.3f}"),
        ("Calmar Ratio",      f"{metrics.get('calmar', 0):.3f}"),
        ("Max Drawdown",      f"{metrics.get('max_drawdown', 0)*100:.1f}%"),
        ("Max DD Duration",   f"{metrics.get('max_drawdown_duration_days', 0)} days"),
        ("Market Beta",       f"{metrics.get('market_beta', float('nan')):.3f}"),
        ("Hit Rate",          f"{metrics.get('hit_rate', 0)*100:.1f}%"),
        ("Profit Factor",     f"{metrics.get('profit_factor', 0):.2f}"),
        ("Ann. Turnover",     f"{metrics.get('annualized_turnover', 0):.1f}x"),
    ]
    if dsr_result:
        lines += [
            ("---", "---"),
            ("Observed SR",        f"{dsr_result.get('observed_sr', 0):.3f}"),
            ("E[max SR | N trials]", f"{dsr_result.get('expected_max_sr_over_trials', 0):.3f}"),
            ("N Trials",           f"{dsr_result.get('n_trials', 0)}"),
            ("P(true SR > 0)",     f"{dsr_result.get('prob_true_sr_positive', 0):.1%}"),
        ]
    width = max(len(k) for k, _ in lines) + 2
    print("\n" + "=" * (width + 20))
    for k, v in lines:
        print(f"  {k:<{width}} {v}")
    print("=" * (width + 20) + "\n")
