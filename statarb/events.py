from dataclasses import dataclass, field
from datetime import date


@dataclass
class MarketEvent:
    timestamp: date
    type: str = field(default="MARKET", init=False)


@dataclass
class SignalEvent:
    timestamp: date
    pair_id: str
    ticker_long: str
    ticker_short: str
    z_score: float
    direction: str      # "ENTRY", "EXIT", "STOP"
    hedge_ratio: float  # shares of short per share of long
    type: str = field(default="SIGNAL", init=False)


@dataclass
class OrderEvent:
    timestamp: date
    ticker: str
    quantity: float     # shares (positive)
    direction: str      # "BUY" or "SELL"
    pair_id: str
    type: str = field(default="ORDER", init=False)


@dataclass
class FillEvent:
    timestamp: date
    ticker: str
    quantity: float
    direction: str
    fill_price: float
    commission: float
    spread_cost: float
    slippage_cost: float
    borrow_cost: float  # daily borrow for short legs; 0 for long legs
    pair_id: str
    type: str = field(default="FILL", init=False)

    @property
    def total_friction(self) -> float:
        return self.commission + self.spread_cost + self.slippage_cost

    @property
    def signed_notional(self) -> float:
        # cash out of portfolio (negative = cash leaving)
        sign = 1 if self.direction == "SELL" else -1
        return sign * self.quantity * self.fill_price
