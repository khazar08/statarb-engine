"""
CLI entry point.

    python -m statarb run --config configs/baseline.yaml
    python -m statarb run --config configs/baseline.yaml --mode walk-forward
    python -m statarb run --config configs/baseline.yaml --mode sensitivity
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("statarb")


def _flatten_universe(sectors: dict) -> list[str]:
    tickers = []
    for names in sectors.values():
        tickers.extend(names)
    return sorted(set(tickers))


def cmd_run(args):
    import pandas as pd

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # reproducibility
    seed = config["portfolio"].get("random_seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    from statarb.data_handler import DataHandler
    from statarb.analytics.metrics import full_metrics
    from statarb.analytics.tearsheet import plot_tearsheet, print_summary

    universe = _flatten_universe(config["universe"]["sectors"])
    data_cfg = config["data"]

    handler = DataHandler(
        universe=universe,
        start=data_cfg["start"],
        end=data_cfg["end"],
        cache_dir=data_cfg["cache_dir"],
        source=data_cfg.get("source", "yfinance"),
    )
    handler.load()

    # download benchmark
    bench_handler = DataHandler(
        universe=[config["benchmark"]["ticker"]],
        start=data_cfg["start"],
        end=data_cfg["end"],
        cache_dir=data_cfg["cache_dir"],
    )
    bench_handler.load()
    bench_equity = bench_handler._data.xs(
        config["benchmark"]["ticker"], level="ticker"
    )["adj_close"].rename("equity")

    mode = getattr(args, "mode", "walk-forward")

    if mode == "walk-forward":
        _run_walk_forward(config, handler, bench_equity)
    elif mode == "sensitivity":
        _run_sensitivity(config, handler)
    else:
        logger.error("Unknown mode: %s", mode)
        sys.exit(1)


def _run_walk_forward(config, handler, bench_equity):
    import pandas as pd
    from statarb.validation.walk_forward import WalkForwardHarness
    from statarb.analytics.metrics import full_metrics
    from statarb.analytics.deflated_sharpe import deflated_sharpe_ratio
    from statarb.analytics.tearsheet import plot_tearsheet, print_summary

    harness = WalkForwardHarness(handler, config)
    result = harness.run()

    equity = result["stitched_equity"]
    if equity.empty:
        logger.error("No equity curve produced — check your date range and universe.")
        return

    # align benchmark
    bench_slice = bench_equity.reindex(equity.index, method="ffill").dropna()
    bench_aligned = bench_slice / bench_slice.iloc[0] * config["portfolio"]["initial_capital"]

    metrics = full_metrics(
        equity,
        trade_log=result.get("stitched_trade_log"),
        bench_equity=bench_aligned,
    )

    returns_arr = equity.pct_change().dropna().values
    dsr = deflated_sharpe_ratio(
        observed_sr=metrics["sharpe"],
        returns=returns_arr,
        n_trials=result["total_trials"],
    )

    print_summary(metrics, dsr_result=dsr)

    fig = plot_tearsheet(
        equity,
        bench_equity=bench_aligned,
        trade_log=result.get("stitched_trade_log"),
        title="StatArb Walk-Forward (OOS)",
        save_path="tearsheet_oos.png",
    )
    logger.info("Tearsheet saved to tearsheet_oos.png")


def _run_sensitivity(config, handler):
    from statarb.validation.walk_forward import WalkForwardHarness
    from statarb.validation.sensitivity import sweep_costs, plot_sensitivity_heatmap

    def run_fn(cfg):
        h2 = handler.create_window_handler(cfg["data"]["start"], cfg["data"]["end"])
        harness = WalkForwardHarness(h2, cfg)
        return harness.run()

    df = sweep_costs(run_fn, config)
    df.to_csv("sensitivity_results.csv", index=False)
    logger.info("Sensitivity results saved to sensitivity_results.csv")
    fig = plot_sensitivity_heatmap(df, metric="sharpe", save_path="sensitivity_sharpe.png")
    plot_sensitivity_heatmap(df, metric="cagr", save_path="sensitivity_cagr.png")
    logger.info("Sensitivity heatmaps saved.")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m statarb",
        description="Statistical Arbitrage Backtesting Engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a backtest")
    run_p.add_argument("--config", required=True, type=Path, help="YAML config file")
    run_p.add_argument(
        "--mode",
        choices=["walk-forward", "sensitivity"],
        default="walk-forward",
        help="Execution mode",
    )
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
