# statarb-engine

An event-driven backtesting framework for cointegration-based pairs trading on US equities.

The point of this project is **not** "I found a profitable strategy."  The point is an honest backtesting *system* — one that structurally prevents lookahead, charges every realistic friction, and reports performance only after correcting for multiple testing.  A Sharpe ratio that survives walk-forward analysis and Deflated Sharpe correction is worth more than a Sharpe of 3.4 on an in-sample curve.

---

## Architecture

```
MarketEvent → SignalEvent → OrderEvent → FillEvent
```

Four components, one shared event queue, single-threaded loop.  Lookahead is structurally impossible: the Strategy only receives `MarketEvent`s in chronological order and can only query `data_handler.latest_bars()`, which is capped at the current simulation timestamp.

```
statarb/
  data_handler.py      streaming, lookahead-safe data access
  events.py            Market / Signal / Order / Fill dataclasses
  engine.py            event loop
  portfolio.py         sizing, risk caps, equity curve
  execution.py         frictions → fills
  strategy/
    kalman.py          from-scratch Kalman hedge ratio
    pair_selection.py  correlation → Engle-Granger → half-life rank
    pairs.py           z-score signal logic
  analytics/
    metrics.py         Sharpe / Sortino / Calmar / drawdown / turnover / beta
    deflated_sharpe.py DSR + P(true SR > 0)
    tearsheet.py       equity curve, drawdown, rolling Sharpe, heatmap
  validation/
    walk_forward.py    rolling reformation, stitched OOS curve
    sensitivity.py     cost-grid sweep and heatmap
```

---

## Strategy: cointegration-based pairs trading

**Pair selection** (run in-sample only, re-run every walk-forward window):
1. Candidate filter: within-sector price correlation > 0.8.
2. Engle-Granger cointegration test; keep pairs with ADF p-value < 0.05 on residuals.
3. OU half-life filter: 2 ≤ half-life ≤ 60 trading days.
4. Rank by half-life (faster reversion preferred); take top *k* deduplicated pairs.

**Hedge ratio** — two implementations:
- **Static OLS**: `β` estimated over the formation window.  Simple but stale out-of-sample.
- **Kalman filter** (default): `[α_t, β_t]` as a random-walk hidden state updated each bar.  Implemented from scratch; no `pykalman`.  Outperforms OLS when cointegration relationship drifts.

**Signal rules** (z-score of spread `s_t = log(P_y) − β_t log(P_x)`):
| Condition | Action |
|-----------|--------|
| `\|z\| > 2.0` | enter (short rich leg, long cheap leg, dollar-neutral) |
| `\|z\| < 0.5` | exit |
| `\|z\| > 3.5` | stop-loss: exit and deactivate pair until next reformation |
| bars in trade > 2 × half-life | time-stop: force exit |

---

## Friction model

All parameters live in `configs/baseline.yaml` under `frictions:`.

| Friction | Default | Notes |
|----------|---------|-------|
| Commission | $0.005/share ($1 min) | configurable per-share |
| Half-spread | 5 bps | half the bid-ask, paid on entry and exit |
| Slippage | 5 bps | adverse fill vs. mid |
| Short borrow | 1% / year | daily accrual on short notional |

Omitting any one of these — especially borrow — is the most common way backtests overstate market-neutral returns.

---

## Validation methodology

### Walk-forward analysis
Roll a 24-month formation / 6-month OOS window across the full history, stepping 6 months at a time.  Pairs and parameters are re-selected on each formation window using only in-sample data, then frozen and traded on the next OOS slice.  The stitched OOS curve is the number reported.

### Deflated Sharpe Ratio (DSR)
Testing many pairs × parameter sets means the best in-sample Sharpe is an order statistic, not an estimate of skill.  The DSR (Bailey & López de Prado 2014) adjusts for:
- The number of independent trials *N*.
- The non-normality (skew and kurtosis) of returns, which widens the SR estimator distribution.

We report `P(true SR > 0 | observed SR, N)`.  A strategy with 99 trials and a reported Sharpe of 1.2 may have a DSR probability of 60% — meaning the honest estimate of real-world performance is barely better than a coin flip.

### Cost sensitivity
`python -m statarb run --config configs/baseline.yaml --mode sensitivity` sweeps commission and slippage multipliers over a grid and saves a heatmap.  A robust strategy should show positive Sharpe at 2× baseline costs.

---

## Quick start

```bash
pip install -e ".[dev]"

# walk-forward backtest (downloads data on first run, caches to data/cache/)
python -m statarb run --config configs/baseline.yaml

# cost sensitivity sweep
python -m statarb run --config configs/baseline.yaml --mode sensitivity

# run tests
pytest tests/ -v
```

---

## Limitations and why this might be wrong

**Survivorship bias** — `yfinance` returns only currently listed tickers.  The fixed universe used here excludes delisted names (Lehman, Bear Stearns, Enron-era refiners, etc.).  This inflates returns; the magnitude depends on how many sector-peers were deleted over the sample period.  A point-in-time constituent list would fix this.

**Fill assumption** — orders are assumed to fill at the adjusted-close price on the same bar the signal is generated.  In reality, daily-bar strategies fill at next-day open or VWAP, and the close-to-open gap can eat or amplify P&L.

**Borrow availability** — the model assumes any short position can always be borrowed at the configured rate.  In practice, small-caps and distressed names become hard-to-borrow or unborrow-able, precisely when the short leg is most needed.

**Cointegration is not stationary** — a pair that cointegrated 2010–2012 may have broken permanently by 2015.  The walk-forward reformation helps but does not eliminate regime dependence.  Stop-losses mitigate the worst cases.

**Residual data snooping** — the DSR reduces but does not eliminate selection bias.  The universe, sector groupings, and the choice of threshold parameters all represent implicit degrees of freedom that are not counted in *N*.

**Transaction costs at scale** — at $1M the slippage model is approximately correct.  At $100M, moving the market becomes the dominant cost; the model does not scale.

---

## Out of scope

- Intraday / tick data, market microstructure, queue position.
- Live trading / broker integration.
- Options, futures, FX.
- Portfolio-level risk management beyond gross exposure and per-pair caps.

---

## Stretch goals (documented, not implemented)

- **Regime overlay**: use an HMM (see [regime-monte-carlo](../regime-monte-carlo)) to only trade pairs when the market is in a mean-reverting regime.  Conditioning on regime should improve walk-forward Sharpe and reduce stop-outs.
- **Johansen cointegration**: trade eigenvector portfolios across *k* > 2 assets instead of bilateral pairs.
- **C++/Numba hot path**: the Kalman update and z-score loop are the inner loop.  A pybind11 or Numba version with a benchmark table would demonstrate the performance gain.
- **Paper-trading hookup**: freeze the walk-forward-selected strategy and run it forward on live data via Alpaca as a true out-of-sample test.
