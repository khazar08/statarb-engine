"""
Cost test: a zero-friction run must beat a positive-friction run by exactly
the modelled cost on a hand-checked toy trade.

We execute one BUY order for 100 shares at $50 (fill at mid).
Frictions:
  commission  = max(100 * 0.005, 1.0) = $0.50
  spread_cost = 50 * (10/10000) / 2 * 100 = $2.50   (half-spread = 5 bps per share)
  slippage    = 50 * (10/10000) * 100     = $5.00
  borrow      = 0 (long leg)

Total friction = $8.00.

The zero-friction run ends with the full $5000 invested; the friction run
ends with $5000 invested but $8.00 less cash — so equity is lower by $8.00.
"""

from datetime import date, timedelta

import pandas as pd
import pytest

from statarb.data_handler import DataHandler
from statarb.events import MarketEvent, OrderEvent, SignalEvent
from statarb.portfolio import Portfolio
from statarb.execution import ExecutionHandler
from statarb.engine import BacktestEngine


def _make_handler(price: float, n: int = 5):
    dates = [date(2020, 6, 1) + timedelta(days=i) for i in range(n)]
    rows = [
        {"date": pd.Timestamp(d), "ticker": "X", "adj_close": price, "volume": 10_000}
        for d in dates
    ]
    df = pd.DataFrame(rows).set_index(["date", "ticker"]).sort_index()

    h = DataHandler.__new__(DataHandler)
    h.universe = ["X"]
    h.start = pd.Timestamp(dates[0])
    h.end = pd.Timestamp(dates[-1])
    h.cache_dir = None
    h.source = "synthetic"
    h._data = df
    h._dates = df.index.get_level_values("date").unique().sort_values()
    h._current_idx = -1
    return h


def _make_config(commission_per_share, spread_bps, slippage_bps, commission_min=0.0):
    return {
        "portfolio": {"initial_capital": 100_000.0, "random_seed": 0},
        "frictions": {
            "commission_per_share": commission_per_share,
            "commission_min_dollars": commission_min,
            "spread_bps": spread_bps,
            "slippage_bps": slippage_bps,
            "borrow_annual_rate": 0.0,
            "gross_exposure_cap": 10.0,
            "per_pair_notional_fraction": 0.5,
        },
    }


class OneBuyStrategy:
    """Buy exactly 100 shares of X on the first bar only."""

    def __init__(self, handler):
        self.data = handler
        self.events = None
        self._done = False

    def calculate_signals(self, event: MarketEvent):
        if self._done:
            return
        self._done = True
        self.events.put(OrderEvent(
            timestamp=event.timestamp,
            ticker="X",
            quantity=100.0,
            direction="BUY",
            pair_id="test",
        ))


def _run(commission, spread_bps, slippage_bps):
    handler = _make_handler(price=50.0, n=3)
    config = _make_config(commission, spread_bps, slippage_bps)
    strategy = OneBuyStrategy(handler)
    portfolio = Portfolio(config, handler)
    execution = ExecutionHandler(config, handler)

    # wire manually (no engine needed since strategy pushes OrderEvent directly)
    import queue
    q = queue.Queue()
    strategy.events = q
    portfolio.events = q
    execution.events = q

    data_stream = handler.stream()

    def dispatch(ev):
        if ev.type == "MARKET":
            strategy.calculate_signals(ev)
            portfolio.update_timeindex(ev)
        elif ev.type == "ORDER":
            execution.execute_order(ev)
        elif ev.type == "FILL":
            portfolio.update_fill(ev)

    data_exhausted = False
    while True:
        if q.empty():
            if data_exhausted:
                break
            try:
                ev = next(data_stream)
                q.put(ev)
            except StopIteration:
                data_exhausted = True
        else:
            dispatch(q.get())

    ec = portfolio.get_equity_curve()
    return float(ec["equity"].iloc[-1])


def test_zero_friction_beats_friction_by_exact_cost():
    # hand-computed cost for 100 shares at $50 (no minimum commission):
    #   commission  = 100 * 0.005              = 0.50
    #   spread cost = 50 * (10/10000) / 2 * 100 = 2.50
    #   slippage    = 50 * (10/10000) * 100     = 5.00
    #   total                                  = 8.00

    equity_zero = _run(commission=0.0, spread_bps=0.0, slippage_bps=0.0)
    equity_full = _run(commission=0.005, spread_bps=10.0, slippage_bps=10.0)

    expected_cost = 8.00
    actual_diff = equity_zero - equity_full

    assert abs(actual_diff - expected_cost) < 0.01, (
        f"Expected friction cost ${expected_cost:.2f}, "
        f"got equity diff ${actual_diff:.2f}"
    )


def test_borrow_cost_charged_on_short():
    """A short position should accrue borrow; a long position should not."""
    handler_short = _make_handler(price=100.0, n=5)
    handler_long = _make_handler(price=100.0, n=5)

    # borrow config: 10% annual = 10/252 per day ≈ 0.0397% / day
    borrow_config = _make_config(0.0, 0.0, 0.0)
    borrow_config["frictions"]["borrow_annual_rate"] = 0.10

    # Test that selling (shorting) adds borrow cost
    from statarb.execution import ExecutionHandler
    from statarb.events import OrderEvent

    handler = _make_handler(price=100.0, n=2)
    exec_handler = ExecutionHandler(borrow_config, handler)

    import queue
    q = queue.Queue()
    exec_handler.events = q
    data_stream = handler.stream()
    next(data_stream)  # advance to day 0

    exec_handler.execute_order(
        OrderEvent(
            timestamp=handler.current_date,
            ticker="X",
            quantity=100.0,
            direction="SELL",
            pair_id="t",
        )
    )
    fill = q.get()
    assert fill.borrow_cost > 0, "Short fill must have positive borrow cost"

    # long fill should have zero borrow
    exec_handler.execute_order(
        OrderEvent(
            timestamp=handler.current_date,
            ticker="X",
            quantity=100.0,
            direction="BUY",
            pair_id="t",
        )
    )
    fill_long = q.get()
    assert fill_long.borrow_cost == 0.0, "Long fill must have zero borrow cost"
