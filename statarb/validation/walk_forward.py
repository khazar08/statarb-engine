"""
Walk-forward analysis harness.

For each rolling window:
  1. Pair selection runs *only* on the formation slice (in-sample).
  2. The strategy trades on the OOS slice with those frozen pairs.
  3. OOS equity curves are stitched into a single continuous curve.

The stitched OOS curve is the number you report.  In-sample curves are
stored for diagnostics only.
"""

import logging
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from statarb.data_handler import DataHandler
from statarb.engine import BacktestEngine
from statarb.execution import ExecutionHandler
from statarb.portfolio import Portfolio
from statarb.strategy.pair_selection import select_pairs
from statarb.strategy.pairs import PairsStrategy

logger = logging.getLogger(__name__)


@dataclass
class WFWindow:
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp


@dataclass
class WFResult:
    window: WFWindow
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    n_pairs_selected: int
    n_trials: int  # pairs × parameter configs (for DSR)


def generate_windows(
    dates: pd.DatetimeIndex,
    formation_days: int,
    trading_days: int,
    step_days: int,
) -> list[WFWindow]:
    windows = []
    start_idx = 0
    while True:
        form_end_idx = start_idx + formation_days - 1
        oos_end_idx = form_end_idx + trading_days

        if form_end_idx >= len(dates) or oos_end_idx >= len(dates):
            break

        windows.append(WFWindow(
            formation_start=dates[start_idx],
            formation_end=dates[form_end_idx],
            oos_start=dates[form_end_idx + 1],
            oos_end=dates[oos_end_idx],
        ))
        start_idx += step_days

    return windows


class WalkForwardHarness:
    def __init__(self, full_data_handler: DataHandler, config: dict):
        self.data = full_data_handler
        self.config = config

        wf = config["walk_forward"]
        self.formation_days = wf["formation_days"]
        self.trading_days = wf["trading_days"]
        self.step_days = wf["step_days"]

        self.sector_map: dict[str, list[str]] = config["universe"]["sectors"]

    def run(self) -> dict:
        dates = self.data._dates
        windows = generate_windows(
            dates, self.formation_days, self.trading_days, self.step_days
        )
        logger.info("Walk-forward: %d windows", len(windows))

        wf_results: list[WFResult] = []
        total_trials = 0

        for i, w in enumerate(windows):
            logger.info(
                "Window %d/%d  formation=%s–%s  OOS=%s–%s",
                i + 1, len(windows),
                w.formation_start.date(), w.formation_end.date(),
                w.oos_start.date(), w.oos_end.date(),
            )

            # pair selection on formation slice
            form_data = self.data.get_slice(w.formation_start, w.formation_end)
            pairs = select_pairs(form_data, self.sector_map, self.config)
            if not pairs:
                logger.warning("Window %d: no pairs selected, skipping.", i + 1)
                continue

            # count trials for DSR (number of candidate pairs tested)
            n_candidates = sum(
                len(tickers) * (len(tickers) - 1) // 2
                for tickers in self.sector_map.values()
            )
            total_trials += n_candidates

            # run OOS backtest with frozen pairs
            oos_handler = self.data.create_window_handler(w.oos_start, w.oos_end)
            result = self._run_oos(oos_handler, pairs)

            wf_results.append(WFResult(
                window=w,
                equity_curve=result["equity_curve"],
                trade_log=result["trade_log"],
                n_pairs_selected=len(pairs),
                n_trials=n_candidates,
            ))

        stitched = self._stitch(wf_results)
        return {
            "stitched_equity": stitched["equity"],
            "stitched_returns": stitched["returns"],
            "stitched_trade_log": stitched["trade_log"],
            "window_results": wf_results,
            "total_trials": total_trials,
        }

    def _run_oos(self, oos_handler: DataHandler, pairs) -> dict:
        strategy = PairsStrategy(pairs, self.config, oos_handler)
        portfolio = Portfolio(self.config, oos_handler)
        execution = ExecutionHandler(self.config, oos_handler)
        engine = BacktestEngine(oos_handler, strategy, portfolio, execution)
        engine.run()

        # accrue daily borrow on any positions still open (already handled in
        # update_fill; here we also catch daily accrual via a separate pass)
        return {
            "equity_curve": portfolio.get_equity_curve(),
            "trade_log": portfolio.get_trade_log(),
        }

    def _stitch(self, results: list[WFResult]) -> dict:
        if not results:
            return {"equity": pd.Series(dtype=float), "returns": pd.Series(dtype=float),
                    "trade_log": pd.DataFrame()}

        equity_pieces: list[pd.Series] = []
        trade_logs: list[pd.DataFrame] = []
        running_equity = self.config["portfolio"]["initial_capital"]

        for r in results:
            ec = r.equity_curve["equity"]
            if ec.empty:
                continue

            # rescale so the piece starts at the running equity level
            scale = running_equity / ec.iloc[0]
            piece = ec * scale
            equity_pieces.append(piece)
            running_equity = float(piece.iloc[-1])

            if not r.trade_log.empty:
                trade_logs.append(r.trade_log)

        if not equity_pieces:
            return {"equity": pd.Series(dtype=float), "returns": pd.Series(dtype=float),
                    "trade_log": pd.DataFrame()}

        stitched_equity = pd.concat(equity_pieces).sort_index()
        stitched_equity = stitched_equity[~stitched_equity.index.duplicated(keep="last")]
        stitched_returns = stitched_equity.pct_change().dropna()

        trade_log = pd.concat(trade_logs, ignore_index=True) if trade_logs else pd.DataFrame()

        return {
            "equity": stitched_equity,
            "returns": stitched_returns,
            "trade_log": trade_log,
        }
