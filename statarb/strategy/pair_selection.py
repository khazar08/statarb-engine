"""
Pair selection pipeline (run in-sample only).

Pipeline per sector:
  1. Correlation pre-screen (> min_correlation)
  2. Engle-Granger cointegration test (ADF p-value < coint_pvalue)
  3. OU half-life filter (min_half_life <= hl <= max_half_life)
  4. Rank by half-life, return top max_pairs across all sectors.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller

logger = logging.getLogger(__name__)


@dataclass
class PairSpec:
    ticker_y: str       # "rich" leg — shorted when z > 0
    ticker_x: str       # "cheap" leg — longed when z > 0
    beta_ols: float     # OLS hedge ratio from formation window
    half_life: float    # days
    adf_pvalue: float
    sector: str
    pair_id: str = ""

    def __post_init__(self):
        if not self.pair_id:
            self.pair_id = f"{self.ticker_y}_{self.ticker_x}"


def _log_prices(data: pd.DataFrame, ticker: str) -> pd.Series | None:
    try:
        s = data.xs(ticker, level="ticker")["adj_close"]
    except KeyError:
        return None
    s = s.dropna()
    if (s <= 0).any():
        return None
    return np.log(s)


def _ols_regression(log_y: pd.Series, log_x: pd.Series) -> tuple[float, float, pd.Series]:
    """OLS of log_y on log_x. Returns (intercept, beta, residuals)."""
    X = add_constant(log_x.values)
    res = OLS(log_y.values, X).fit()
    intercept, beta = res.params
    residuals = pd.Series(res.resid, index=log_y.index)
    return float(intercept), float(beta), residuals


def _adf_pvalue(residuals: pd.Series) -> float:
    try:
        result = adfuller(residuals.dropna(), autolag="AIC", maxlag=10)
        return float(result[1])
    except Exception:
        return 1.0


def compute_half_life(spread: pd.Series) -> float:
    """
    Fit AR(1): delta_s = lambda * s_{t-1} + mu + eps
    Half-life = -ln(2) / lambda
    """
    s = spread.dropna()
    delta_s = s.diff().dropna()
    s_lag = s.shift(1).dropna()
    delta_s, s_lag = delta_s.align(s_lag, join="inner")

    X = add_constant(s_lag.values)
    res = OLS(delta_s.values, X).fit()
    lam = res.params[1]  # coefficient on lagged spread

    if lam >= 0:
        # not mean-reverting
        return np.inf
    return float(-np.log(2) / lam)


def select_pairs(
    data: pd.DataFrame,
    sector_map: dict[str, list[str]],
    config: dict,
) -> list[PairSpec]:
    """
    Run the full selection pipeline on `data` (a formation-window slice).
    Returns at most config['strategy']['max_pairs'] PairSpecs, sorted by half-life.
    """
    s_cfg = config["strategy"]
    min_corr = s_cfg.get("min_correlation", 0.80)
    coint_pval = s_cfg.get("coint_pvalue", 0.05)
    min_hl = s_cfg.get("min_half_life", 2)
    max_hl = s_cfg.get("max_half_life", 60)
    max_pairs = s_cfg.get("max_pairs", 10)

    candidates: list[PairSpec] = []

    for sector, tickers in sector_map.items():
        log_px: dict[str, pd.Series] = {}
        for t in tickers:
            lp = _log_prices(data, t)
            if lp is not None and len(lp) > 60:
                log_px[t] = lp

        ticker_list = list(log_px.keys())

        for i in range(len(ticker_list)):
            for j in range(i + 1, len(ticker_list)):
                t1, t2 = ticker_list[i], ticker_list[j]
                s1, s2 = log_px[t1].align(log_px[t2], join="inner")

                if len(s1) < 100:
                    continue

                # correlation pre-screen
                if s1.corr(s2) < min_corr:
                    continue

                # Engle-Granger: try both directions, keep best
                best = None
                for y_t, x_t, sy, sx in [(t1, t2, s1, s2), (t2, t1, s2, s1)]:
                    try:
                        _, beta, resid = _ols_regression(sy, sx)
                    except Exception:
                        continue
                    pval = _adf_pvalue(resid)
                    if pval >= coint_pval:
                        continue
                    hl = compute_half_life(resid)
                    if not (min_hl <= hl <= max_hl):
                        continue
                    if best is None or pval < best[0]:
                        best = (pval, y_t, x_t, beta, hl, resid)

                if best is None:
                    continue

                pval, y_t, x_t, beta, hl, _ = best
                candidates.append(
                    PairSpec(
                        ticker_y=y_t,
                        ticker_x=x_t,
                        beta_ols=beta,
                        half_life=hl,
                        adf_pvalue=pval,
                        sector=sector,
                    )
                )

    # rank by half-life (prefer faster mean-reversion), deduplicate tickers
    candidates.sort(key=lambda p: p.half_life)
    used: set[str] = set()
    selected: list[PairSpec] = []
    for p in candidates:
        if p.ticker_y in used or p.ticker_x in used:
            continue
        used.add(p.ticker_y)
        used.add(p.ticker_x)
        selected.append(p)
        if len(selected) >= max_pairs:
            break

    logger.info(
        "Pair selection: %d candidates -> %d selected (max %d)",
        len(candidates), len(selected), max_pairs,
    )
    return selected
