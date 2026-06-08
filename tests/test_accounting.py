"""
Accounting invariant: cash + mark-to-market(positions) == equity at every step.

We run a small backtest and verify that portfolio.equity_curve[i]['equity']
equals cash + sum(shares * price) at the corresponding date.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from statarb.data_handler import DataHandler
from statarb.events import MarketEvent, SignalEvent
from statarb.portfolio import Portfolio
from statarb.execution import ExecutionHandler
from statarb.engine import BacktestEngine


def _make_handler(prices_a, prices_b, dates):
    rows = []
    for d, pa, pb in zip(dates, prices_a, prices_b):
        ts = pd.Timestamp(d)
        rows.append({"date": ts, "ticker": "A", "adj_close": pa, "volume": 1000})
        rows.append({"date": ts, "ticker": "B", "adj_close": pb, "volume": 1000})
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    h = DataHandler.__new__(DataHandler)
    h.universe = ["A", "B"]
    h.start = pd.Timestamp(dates[0])
    h.end = pd.Timestamp(dates[-1])
    h.cache_dir = None
    h.source = "synthetic"
    h._data = df
    h._dates = df.index.get_level_values("date").unique().sort_values()
    h._current_idx = -1
    return h


def _zero_friction_config():
    return {
        "portfolio": {"initial_capital": 50_000.0, "random_seed": 0},
        "frictions": {
            "commission_per_share": 0.0,
            "commission_min_dollars": 0.0,
            "spread_bps": 0.0,
            "slippage_bps": 0.0,
            "borrow_annual_rate": 0.0,
            "gross_exposure_cap": 10.0,
            "per_pair_notional_fraction": 0.2,
        },
    }


class SingleEntryStrategy:
    """Buy A long, sell B short on day 3; exit on day 7."""

    def __init__(self, handler):
        self.data = handler
        self.events = None
        self._day = 0

    def calculate_signals(self, event: MarketEvent):
        self._day += 1
        if self._day == 3:
            self.events.put(SignalEvent(
                timestamp=event.timestamp,
                pair_id="AB",
                ticker_long="A",
                ticker_short="B",
                z_score=2.5,
                direction="ENTRY",
                hedge_ratio=1.0,
            ))
        elif self._day == 7:
            self.events.put(SignalEvent(
                timestamp=event.timestamp,
                pair_id="AB",
                ticker_long="A",
                ticker_short="B",
                z_score=0.2,
                direction="EXIT",
                hedge_ratio=1.0,
            ))


def test_accounting_invariant():
    n = 15
    dates = [date(2021, 1, 4) + timedelta(days=i) for i in range(n)]
    np.random.seed(7)
    prices_a = 50.0 + np.cumsum(np.random.randn(n) * 0.3)
    prices_b = 48.0 + np.cumsum(np.random.randn(n) * 0.3)

    handler = _make_handler(prices_a.tolist(), prices_b.tolist(), dates)
    config = _zero_friction_config()
    strategy = SingleEntryStrategy(handler)
    portfolio = Portfolio(config, handler)
    execution = ExecutionHandler(config, handler)
    engine = BacktestEngine(handler, strategy, portfolio, execution)
    engine.run()

    ec = portfolio.get_equity_curve()
    assert len(ec) == n, "Equity curve length must equal number of bars"

    for row in ec.itertuples():
        equity_recorded = float(row.equity)
        cash = float(row.cash)
        holdings = float(row.holdings)
        recomputed = cash + holdings
        assert abs(equity_recorded - recomputed) < 1e-4, (
            f"Accounting mismatch on {row.Index}: "
            f"recorded={equity_recorded:.4f}, cash+holdings={recomputed:.4f}"
        )
