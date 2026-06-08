"""
ExecutionHandler — applies all frictions and converts OrderEvents to FillEvents.

Friction model (all configurable):
  commission  : per-share fee with a per-trade minimum.
  spread_cost : half-spread (crossing the bid-ask); approximated as bps of price.
  slippage    : adverse fill vs. mid, also in bps.
  borrow_cost : one-time borrow fee charged at the moment of shorting (annualized
                rate prorated to expected holding; kept simple here as a flat
                one-day charge since daily borrow is accrued in Portfolio).
"""

import logging

from statarb.events import FillEvent, OrderEvent

logger = logging.getLogger(__name__)


class ExecutionHandler:
    def __init__(self, config: dict, data_handler):
        self.config = config
        self.data = data_handler
        self.events = None  # injected by engine

        f_cfg = config["frictions"]
        self.commission_per_share: float = f_cfg["commission_per_share"]
        self.commission_min: float = f_cfg["commission_min_dollars"]
        self.spread_bps: float = f_cfg["spread_bps"]
        self.slippage_bps: float = f_cfg["slippage_bps"]
        self.borrow_annual_rate: float = f_cfg["borrow_annual_rate"]

    def execute_order(self, event: OrderEvent) -> None:
        bars = self.data.latest_bars(event.ticker, n=1)
        if bars is None or bars.empty:
            logger.warning("No price data for %s on %s; order dropped", event.ticker, event.timestamp)
            return

        mid_price = float(bars["adj_close"].iloc[-1])

        spread_cost = self._compute_spread(mid_price, event.quantity)
        slippage_cost = self._compute_slippage(mid_price, event.quantity, event.direction)
        commission = self._compute_commission(event.quantity, mid_price)

        # fill_price is the clean mid; frictions are charged separately in cash
        fill_price = mid_price

        borrow = 0.0
        if event.direction == "SELL":
            # charge one day of borrow as opening cost; Portfolio accrues rest daily
            borrow = self._compute_borrow(mid_price, event.quantity, days=1)

        fill = FillEvent(
            timestamp=event.timestamp,
            ticker=event.ticker,
            quantity=event.quantity,
            direction=event.direction,
            fill_price=fill_price,
            commission=commission,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            borrow_cost=borrow,
            pair_id=event.pair_id,
        )
        self.events.put(fill)

    # ------------------------------------------------------------------ #

    def _compute_commission(self, quantity: float, price: float) -> float:
        raw = quantity * self.commission_per_share
        return max(raw, self.commission_min)

    def _compute_spread(self, price: float, quantity: float) -> float:
        # half-spread per share, applied to full quantity
        half_spread_per_share = price * (self.spread_bps / 10_000) / 2
        return half_spread_per_share * quantity

    def _compute_slippage(self, price: float, quantity: float, direction: str) -> float:
        slippage_per_share = price * (self.slippage_bps / 10_000)
        return slippage_per_share * quantity

    def _compute_borrow(self, price: float, quantity: float, days: int = 1) -> float:
        daily_rate = self.borrow_annual_rate / 252.0
        return abs(quantity) * price * daily_rate * days
