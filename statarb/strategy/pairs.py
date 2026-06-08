"""
PairsStrategy — event-driven z-score signal generator.

For each active pair the strategy maintains:
  - A rolling spread history (for z-score computation)
  - A KalmanHedgeRatio instance (if hedge_ratio == "kalman")
  - The current position state: FLAT, LONG_SPREAD, SHORT_SPREAD

Signal conventions (from the portfolio's perspective):
  direction="ENTRY"  ticker_long goes long,  ticker_short goes short
  direction="EXIT"   flatten both legs
  direction="STOP"   flatten both legs and deactivate pair until next window
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from statarb.events import MarketEvent, SignalEvent
from statarb.strategy.kalman import KalmanHedgeRatio
from statarb.strategy.pair_selection import PairSpec

logger = logging.getLogger(__name__)


@dataclass
class PairState:
    spec: PairSpec
    kalman: KalmanHedgeRatio | None
    spread_history: deque
    position: str = "FLAT"          # "FLAT", "LONG_SPREAD", "SHORT_SPREAD"
    bars_in_trade: int = 0
    active: bool = True
    current_beta: float = 0.0       # live hedge ratio (OLS or Kalman)


class PairsStrategy:
    """
    On each MarketEvent:
      1. Fetch latest prices for all active pairs.
      2. Update Kalman filter (or use static OLS beta).
      3. Recompute spread and rolling z-score.
      4. Apply entry / exit / stop rules.
      5. Push SignalEvents into the shared event queue.
    """

    def __init__(
        self,
        pairs: list[PairSpec],
        config: dict,
        data_handler,
    ):
        self.config = config
        self.data = data_handler
        self.events = None   # injected by engine

        s_cfg = config["strategy"]
        use_kalman = s_cfg.get("hedge_ratio", "kalman") == "kalman"
        delta = s_cfg.get("kalman_delta", 1e-4)
        obs_noise = s_cfg.get("kalman_obs_noise", 1e-2)
        self.entry_thr = s_cfg["entry_threshold"]
        self.exit_thr = s_cfg["exit_threshold"]
        self.stop_thr = s_cfg["stop_threshold"]
        self.zscore_multiplier = s_cfg.get("zscore_window_multiplier", 1.0)

        self.pair_states: dict[str, PairState] = {}
        for spec in pairs:
            kf = KalmanHedgeRatio(delta=delta, obs_noise=obs_noise) if use_kalman else None
            # warm-up window = ceil(half_life * zscore_multiplier), at least 20
            window_size = max(20, int(np.ceil(spec.half_life * self.zscore_multiplier)))
            self.pair_states[spec.pair_id] = PairState(
                spec=spec,
                kalman=kf,
                spread_history=deque(maxlen=window_size * 3),  # keep extra for rolling
                current_beta=spec.beta_ols,
            )

    def calculate_signals(self, event: MarketEvent) -> None:
        ts = event.timestamp
        for pid, ps in self.pair_states.items():
            if not ps.active:
                continue
            self._process_pair(ts, ps)

    def _process_pair(self, ts: date, ps: PairState) -> None:
        bars_y = self.data.latest_bars(ps.spec.ticker_y, n=1)
        bars_x = self.data.latest_bars(ps.spec.ticker_x, n=1)
        if bars_y is None or bars_x is None or bars_y.empty or bars_x.empty:
            return

        price_y = float(bars_y["adj_close"].iloc[-1])
        price_x = float(bars_x["adj_close"].iloc[-1])
        if price_y <= 0 or price_x <= 0:
            return

        log_y = np.log(price_y)
        log_x = np.log(price_x)

        # update hedge ratio
        if ps.kalman is not None:
            _, beta = ps.kalman.update(log_x, log_y)
            ps.current_beta = beta
        else:
            ps.current_beta = ps.spec.beta_ols

        spread = log_y - ps.current_beta * log_x
        ps.spread_history.append(spread)

        window = max(20, int(np.ceil(ps.spec.half_life * self.zscore_multiplier)))
        history = list(ps.spread_history)
        if len(history) < window:
            return  # not enough data yet

        recent = np.array(history[-window:])
        mu = recent.mean()
        sigma = recent.std(ddof=1)
        if sigma < 1e-10:
            return

        z = (spread - mu) / sigma

        self._apply_rules(ts, ps, z)
        if ps.position != "FLAT":
            ps.bars_in_trade += 1

    def _apply_rules(self, ts: date, ps: PairState, z: float) -> None:
        spec = ps.spec
        time_stop = int(ps.spec.half_life * 2)

        if ps.position == "FLAT":
            if z > self.entry_thr:
                # spread is high: short y (ticker_y), long x (ticker_x)
                self._emit(ts, ps, "ENTRY", z, long_t=spec.ticker_x, short_t=spec.ticker_y)
                ps.position = "SHORT_SPREAD"
                ps.bars_in_trade = 0

            elif z < -self.entry_thr:
                # spread is low: long y (ticker_y), short x (ticker_x)
                self._emit(ts, ps, "ENTRY", z, long_t=spec.ticker_y, short_t=spec.ticker_x)
                ps.position = "LONG_SPREAD"
                ps.bars_in_trade = 0

        elif ps.position in ("SHORT_SPREAD", "LONG_SPREAD"):
            if abs(z) > self.stop_thr:
                self._emit(ts, ps, "STOP", z,
                           long_t=spec.ticker_x if ps.position == "SHORT_SPREAD" else spec.ticker_y,
                           short_t=spec.ticker_y if ps.position == "SHORT_SPREAD" else spec.ticker_x)
                ps.position = "FLAT"
                ps.active = False
                logger.debug("Pair %s stopped out at z=%.2f", spec.pair_id, z)

            elif abs(z) < self.exit_thr:
                self._emit(ts, ps, "EXIT", z,
                           long_t=spec.ticker_x if ps.position == "SHORT_SPREAD" else spec.ticker_y,
                           short_t=spec.ticker_y if ps.position == "SHORT_SPREAD" else spec.ticker_x)
                ps.position = "FLAT"

            elif ps.bars_in_trade >= time_stop:
                self._emit(ts, ps, "EXIT", z,
                           long_t=spec.ticker_x if ps.position == "SHORT_SPREAD" else spec.ticker_y,
                           short_t=spec.ticker_y if ps.position == "SHORT_SPREAD" else spec.ticker_x)
                ps.position = "FLAT"
                logger.debug("Pair %s time-stopped after %d bars", spec.pair_id, ps.bars_in_trade)

    def _emit(self, ts, ps: PairState, direction: str, z: float,
              long_t: str, short_t: str) -> None:
        if self.events is None:
            return
        self.events.put(
            SignalEvent(
                timestamp=ts,
                pair_id=ps.spec.pair_id,
                ticker_long=long_t,
                ticker_short=short_t,
                z_score=z,
                direction=direction,
                hedge_ratio=abs(ps.current_beta),
            )
        )
