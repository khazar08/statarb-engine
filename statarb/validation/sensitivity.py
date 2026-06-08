"""
Cost sensitivity analysis.

Sweeps commission and slippage multipliers over a grid and reports how
Sharpe and CAGR degrade as frictions increase.  A strategy whose edge
vanishes at 2× costs has no margin of safety.
"""

import copy
import logging
from itertools import product

import numpy as np
import pandas as pd

from statarb.analytics.metrics import full_metrics

logger = logging.getLogger(__name__)


def _run_single(full_run_fn, config: dict) -> dict:
    """
    full_run_fn(config) must return a dict with at minimum:
        "stitched_equity": pd.Series
        "stitched_trade_log": pd.DataFrame
    """
    result = full_run_fn(config)
    equity = result["stitched_equity"]
    if equity.empty:
        return {"sharpe": np.nan, "cagr": np.nan}
    metrics = full_metrics(equity, result.get("stitched_trade_log"))
    return {"sharpe": metrics["sharpe"], "cagr": metrics["cagr"]}


def sweep_costs(
    run_fn,
    base_config: dict,
    commission_multipliers: list[float] | None = None,
    slippage_multipliers: list[float] | None = None,
    borrow_multipliers: list[float] | None = None,
) -> pd.DataFrame:
    """
    Run `run_fn(config)` for every (commission_mult, slippage_mult) pair.
    Returns a DataFrame with columns: commission_mult, slippage_mult,
    borrow_mult, sharpe, cagr.

    `run_fn` is called once per grid point; it should be deterministic.
    For a walk-forward harness this can be expensive — consider a smaller
    universe or shorter date range for the sensitivity pass.
    """
    sens_cfg = base_config.get("sensitivity", {})
    comm_mults = commission_multipliers or sens_cfg.get("commission_multipliers", [0.5, 1.0, 2.0])
    slip_mults = slippage_multipliers or sens_cfg.get("slippage_multipliers", [0.5, 1.0, 2.0, 3.0])
    borr_mults = borrow_multipliers or sens_cfg.get("borrow_multipliers", [0.5, 1.0, 2.0])

    rows = []
    total = len(comm_mults) * len(slip_mults) * len(borr_mults)
    done = 0

    base_comm = base_config["frictions"]["commission_per_share"]
    base_slip = base_config["frictions"]["slippage_bps"]
    base_borr = base_config["frictions"]["borrow_annual_rate"]

    for cm, sm, bm in product(comm_mults, slip_mults, borr_mults):
        cfg = copy.deepcopy(base_config)
        cfg["frictions"]["commission_per_share"] = base_comm * cm
        cfg["frictions"]["slippage_bps"] = base_slip * sm
        cfg["frictions"]["borrow_annual_rate"] = base_borr * bm

        done += 1
        logger.info(
            "Sensitivity %d/%d  comm×%.1f  slip×%.1f  borr×%.1f",
            done, total, cm, sm, bm,
        )

        try:
            metrics = _run_single(run_fn, cfg)
        except Exception as exc:
            logger.warning("Sensitivity run failed: %s", exc)
            metrics = {"sharpe": np.nan, "cagr": np.nan}

        rows.append({
            "commission_mult": cm,
            "slippage_mult": sm,
            "borrow_mult": bm,
            "sharpe": metrics["sharpe"],
            "cagr": metrics["cagr"],
        })

    return pd.DataFrame(rows)


def plot_sensitivity_heatmap(
    sensitivity_df: pd.DataFrame,
    metric: str = "sharpe",
    save_path: str | None = None,
):
    """
    Plot Sharpe (or CAGR) as a 2-D heatmap of commission × slippage
    multipliers, averaged over borrow multiplier levels.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = (
        sensitivity_df
        .groupby(["commission_mult", "slippage_mult"])[metric]
        .mean()
        .unstack("slippage_mult")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title(f"{metric.title()} vs cost multipliers (avg over borrow levels)")
    ax.set_xlabel("Slippage multiplier")
    ax.set_ylabel("Commission multiplier")

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)

    return fig
