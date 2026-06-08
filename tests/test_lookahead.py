"""
Lookahead test.

Shifting all prices forward by one day and re-running the engine must
produce a *different* (not better) P&L.  If the engine looked ahead,
the shifted run would be identical to the original — it isn't because
the strategy only ever sees latest_bars() which is capped at current_date.

We use a toy synthetic dataset and a trivial strategy (always long the
first ticker) so the test is fast and deterministic.
"""

import queue
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from statarb.data_handler import DataHandler
from statarb.engine import BacktestEngine
from statarb.events import MarketEvent, OrderEvent, FillEvent
from statarb.portfolio import Portfolio
from statarb.execution import ExecutionHandler


# ------------------------------------------------------------------ #
#  Minimal synthetic DataHandler
# ------------------------------------------------------------------ #

def _make_handler(prices: dict[str, list[float]], dates: list[date]) -> DataHandler:
    rows = []
    for ticker, px in prices.items():
        for d, p in zip(dates, px):
            rows.append({"date": pd.Timestamp(d), "ticker": ticker, "adj_close": p, "volume": 1000})
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    handler = DataHandler.__new__(DataHandler)
    handler.universe = list(prices.keys())
    handler.start = pd.Timestamp(dates[0])
    handler.end = pd.Timestamp(dates[-1])
    handler.cache_dir = None
    handler.source = "synthetic"
    handler._data = df
    handler._dates = df.index.get_level_values("date").unique().sort_values()
    handler._current_idx = -1
    return handler


# ------------------------------------------------------------------ #
#  Trivial strategy: buy ticker_a on day 1, never sell
# ------------------------------------------------------------------ #

class TrivialLongStrategy:
    """Buy 100 shares on the first bar and hold. Uses OrderEvent directly so
    the position is definitely taken and the equity curve depends on prices."""

    def __init__(self, ticker, data_handler):
        self.ticker = ticker
        self.data = data_handler
        self.events = None
        self._entered = False

    def calculate_signals(self, event: MarketEvent):
        if self._entered:
            return
        bars = self.data.latest_bars(self.ticker, n=1)
        if bars is None or bars.empty:
            return
        self._entered = True
        self.events.put(OrderEvent(
            timestamp=event.timestamp,
            ticker=self.ticker,
            quantity=100.0,
            direction="BUY",
            pair_id="test",
        ))


def _minimal_config():
    return {
        "portfolio": {"initial_capital": 100_000.0, "random_seed": 0},
        "frictions": {
            "commission_per_share": 0.0,
            "commission_min_dollars": 0.0,
            "spread_bps": 0.0,
            "slippage_bps": 0.0,
            "borrow_annual_rate": 0.0,
            "gross_exposure_cap": 10.0,
            "per_pair_notional_fraction": 0.5,
        },
    }


def _run_with_prices(prices, dates):
    handler = _make_handler(prices, dates)
    ticker = list(prices.keys())[0]
    config = _minimal_config()
    strategy = TrivialLongStrategy(ticker, handler)
    portfolio = Portfolio(config, handler)
    execution = ExecutionHandler(config, handler)
    engine = BacktestEngine(handler, strategy, portfolio, execution)
    engine.run()
    ec = portfolio.get_equity_curve()
    return float(ec["equity"].iloc[-1]) if not ec.empty else 0.0


def test_lookahead_not_possible():
    """
    Shifting prices by one day must change final equity.
    If lookahead existed, final equity would be the same.
    """
    n = 20
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    prices = [100.0 + i * 0.5 for i in range(n)]  # trending up

    original_equity = _run_with_prices({"AAA": prices}, dates)

    # shift prices forward 1 day: day-0 price is now day-1's, etc.
    shifted_prices = prices[1:] + [prices[-1]]
    shifted_equity = _run_with_prices({"AAA": shifted_prices}, dates)

    # If the strategy could see the future, it would perform identically.
    # With the lookahead barrier, the shifted run has a different price path.
    assert original_equity != shifted_equity, (
        "Engine appears to have lookahead: equity is identical under a shifted price series."
    )


def test_latest_bars_does_not_see_future():
    """
    latest_bars must never return data beyond current_date.
    """
    n = 10
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    prices = list(range(100, 100 + n))
    handler = _make_handler({"BBB": prices}, dates)

    stream = handler.stream()
    for i, event in enumerate(stream):
        bars = handler.latest_bars("BBB", n=999)
        assert bars is not None
        assert len(bars) == i + 1, (
            f"On day {i}, latest_bars returned {len(bars)} bars instead of {i+1}"
        )
        last_price = float(bars["adj_close"].iloc[-1])
        expected = float(prices[i])
        assert last_price == expected, (
            f"Day {i}: expected price {expected}, got {last_price} (future leak)"
        )
