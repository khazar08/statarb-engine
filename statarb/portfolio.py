"""
Portfolio — the single source of truth for cash, positions, and equity.

Sizing logic: dollar-neutral pairs.
  - Each pair gets at most `per_pair_notional_fraction` of current equity per leg.
  - Total gross exposure is capped at `gross_exposure_cap * equity`.
  - Shares are computed so that the two legs have equal dollar notional
    adjusted by the hedge ratio.
"""

import logging
from datetime import date

import pandas as pd

from statarb.events import FillEvent, MarketEvent, OrderEvent, SignalEvent

logger = logging.getLogger(__name__)


class Portfolio:
    def __init__(self, config: dict, data_handler):
        self.config = config
        self.data = data_handler
        self.events = None  # injected by engine

        p_cfg = config["portfolio"]
        f_cfg = config["frictions"]
        self.initial_capital: float = p_cfg["initial_capital"]
        self.per_pair_frac: float = f_cfg["per_pair_notional_fraction"]
        self.gross_cap: float = f_cfg["gross_exposure_cap"]

        self.cash: float = self.initial_capital
        # positions: ticker -> signed shares (positive = long, negative = short)
        self.positions: dict[str, float] = {}
        # pair -> {"long": ticker, "short": ticker}
        self.open_pairs: dict[str, dict] = {}

        self.equity_curve: list[dict] = []       # [{date, equity, cash}, ...]
        self.trade_log: list[dict] = []

    # ------------------------------------------------------------------ #

    def update_timeindex(self, event: MarketEvent) -> None:
        """Mark-to-market and append to equity curve."""
        mtm = self._mark_to_market(event.timestamp)
        equity = self.cash + mtm
        self.equity_curve.append({
            "date": event.timestamp,
            "equity": equity,
            "cash": self.cash,
            "holdings": mtm,
        })

    def update_signal(self, event: SignalEvent) -> None:
        if event.direction == "ENTRY":
            self._handle_entry(event)
        elif event.direction in ("EXIT", "STOP"):
            self._handle_exit(event)

    def update_fill(self, event: FillEvent) -> None:
        # update position
        sign = 1 if event.direction == "BUY" else -1
        ticker = event.ticker
        self.positions[ticker] = self.positions.get(ticker, 0.0) + sign * event.quantity

        # update cash: buying spends cash, selling receives cash, frictions always paid
        if event.direction == "BUY":
            self.cash -= event.quantity * event.fill_price + event.total_friction
        else:
            self.cash += event.quantity * event.fill_price - event.total_friction

        # daily borrow already charged on open short positions; also charge fill borrow
        self.cash -= event.borrow_cost

        if abs(self.positions[ticker]) < 1e-6:
            del self.positions[ticker]

        self.trade_log.append({
            "date": event.timestamp,
            "ticker": event.ticker,
            "direction": event.direction,
            "quantity": event.quantity,
            "fill_price": event.fill_price,
            "commission": event.commission,
            "spread_cost": event.spread_cost,
            "slippage_cost": event.slippage_cost,
            "borrow_cost": event.borrow_cost,
            "pair_id": event.pair_id,
        })

    # ------------------------------------------------------------------ #
    #  Daily borrow on open short positions
    # ------------------------------------------------------------------ #

    def charge_daily_borrow(self) -> None:
        """Call once per bar after mark-to-market to accrue short-borrow fees."""
        annual_rate = self.config["frictions"]["borrow_annual_rate"]
        daily_rate = annual_rate / 252.0
        for ticker, shares in self.positions.items():
            if shares < 0:
                price_data = self.data.latest_bars(ticker, n=1)
                if price_data is None or price_data.empty:
                    continue
                price = float(price_data["adj_close"].iloc[-1])
                borrow = abs(shares) * price * daily_rate
                self.cash -= borrow

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _handle_entry(self, event: SignalEvent) -> None:
        if event.pair_id in self.open_pairs:
            return  # already open

        equity = self._current_equity()
        target_notional = self.per_pair_frac * equity

        # check gross exposure headroom
        gross = self._gross_exposure()
        if gross + 2 * target_notional > self.gross_cap * equity:
            logger.debug("Gross cap hit, skipping pair %s", event.pair_id)
            return

        price_long = self._latest_price(event.ticker_long)
        price_short = self._latest_price(event.ticker_short)
        if price_long is None or price_short is None:
            return

        shares_long = target_notional / price_long
        shares_short = (target_notional * event.hedge_ratio) / price_short

        if shares_long < 0.01 or shares_short < 0.01:
            return

        self.open_pairs[event.pair_id] = {
            "long": event.ticker_long,
            "short": event.ticker_short,
        }

        ts = event.timestamp
        self.events.put(OrderEvent(ts, event.ticker_long, shares_long, "BUY", event.pair_id))
        self.events.put(OrderEvent(ts, event.ticker_short, shares_short, "SELL", event.pair_id))

    def _handle_exit(self, event: SignalEvent) -> None:
        if event.pair_id not in self.open_pairs:
            return

        legs = self.open_pairs.pop(event.pair_id)
        ts = event.timestamp

        for ticker in [legs["long"], legs["short"]]:
            shares = self.positions.get(ticker, 0.0)
            if abs(shares) < 0.01:
                continue
            direction = "SELL" if shares > 0 else "BUY"
            self.events.put(OrderEvent(ts, ticker, abs(shares), direction, event.pair_id))

    def _mark_to_market(self, ts: date) -> float:
        total = 0.0
        for ticker, shares in self.positions.items():
            price_data = self.data.latest_bars(ticker, n=1)
            if price_data is None or price_data.empty:
                continue
            price = float(price_data["adj_close"].iloc[-1])
            total += shares * price
        return total

    def _current_equity(self) -> float:
        return self.cash + self._mark_to_market(None)

    def _gross_exposure(self) -> float:
        gross = 0.0
        for ticker, shares in self.positions.items():
            price_data = self.data.latest_bars(ticker, n=1)
            if price_data is None or price_data.empty:
                continue
            price = float(price_data["adj_close"].iloc[-1])
            gross += abs(shares) * price
        return gross

    def _latest_price(self, ticker: str) -> float | None:
        bars = self.data.latest_bars(ticker, n=1)
        if bars is None or bars.empty:
            return None
        return float(bars["adj_close"].iloc[-1])

    def get_equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve).set_index("date")

    def get_trade_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.trade_log)
