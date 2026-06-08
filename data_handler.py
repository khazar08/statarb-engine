import hashlib
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import yfinance as yf

from statarb.events import MarketEvent

logger = logging.getLogger(__name__)


class DataHandler:
    """
    Lookahead-safe data access for the event-driven engine.

    The key contract: `latest_bars` and any other per-ticker query only
    return data at or before `self.current_date`.  The engine advances
    `current_date` by iterating `stream()`, so the strategy physically
    cannot observe future prices.
    """

    def __init__(
        self,
        universe: list[str],
        start: str | date,
        end: str | date,
        cache_dir: str | Path,
        source: str = "yfinance",
    ):
        self.universe = sorted(universe)
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.cache_dir = Path(cache_dir)
        self.source = source

        self._data: pd.DataFrame | None = None   # MultiIndex (date, ticker)
        self._dates: pd.DatetimeIndex | None = None
        self._current_idx: int = -1

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def load(self) -> pd.DataFrame:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / "prices.parquet"

        if cache_path.exists():
            logger.info("Loading prices from cache: %s", cache_path)
            df = pd.read_parquet(cache_path)
        else:
            logger.info("Downloading price data for %d tickers", len(self.universe))
            df = self._download()
            df.to_parquet(cache_path)
            logger.info("Cached to %s", cache_path)

        self._log_hash(cache_path)
        self._data = df
        self._dates = (
            df.index.get_level_values("date").unique().sort_values()
        )
        return df

    def stream(self) -> Iterator[MarketEvent]:
        """Yields one MarketEvent per trading day in chronological order."""
        if self._data is None:
            raise RuntimeError("Call load() before stream()")
        for i, dt in enumerate(self._dates):
            self._current_idx = i
            yield MarketEvent(timestamp=dt.date())

    @property
    def current_date(self):
        if self._current_idx < 0 or self._dates is None:
            return None
        return self._dates[self._current_idx]

    def latest_bars(self, ticker: str, n: int = 1) -> pd.DataFrame | None:
        """Return the last n bars for ticker up to and including current_date."""
        if self._current_idx < 0:
            return None
        cutoff = self._dates[self._current_idx]
        try:
            ticker_data = self._data.xs(ticker, level="ticker")
        except KeyError:
            return None
        filtered = ticker_data[ticker_data.index <= cutoff]
        return filtered.iloc[-n:] if len(filtered) >= 1 else None

    def get_slice(self, start, end) -> pd.DataFrame:
        """Return the full cross-section between two dates (used by pair selection)."""
        if self._data is None:
            raise RuntimeError("Call load() before get_slice()")
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        idx_dates = self._data.index.get_level_values("date")
        mask = (idx_dates >= start_ts) & (idx_dates <= end_ts)
        return self._data.loc[mask]

    def create_window_handler(self, start, end) -> "DataHandler":
        """Create a DataHandler that covers exactly [start, end] from the cache."""
        handler = DataHandler.__new__(DataHandler)
        handler.universe = self.universe
        handler.start = pd.Timestamp(start)
        handler.end = pd.Timestamp(end)
        handler.cache_dir = self.cache_dir
        handler.source = self.source

        slice_df = self.get_slice(start, end)
        handler._data = slice_df
        handler._dates = (
            slice_df.index.get_level_values("date").unique().sort_values()
        )
        handler._current_idx = -1
        return handler

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _download(self) -> pd.DataFrame:
        raw = yf.download(
            self.universe,
            start=self.start.strftime("%Y-%m-%d"),
            end=self.end.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        # yfinance returns MultiIndex columns: (field, ticker)
        # We want MultiIndex rows: (date, ticker)
        adj_close = raw["Close"].copy()
        volume = raw["Volume"].copy()

        # Stack to (date, ticker)
        adj_close.index = pd.DatetimeIndex(adj_close.index)
        adj_close.index.name = "date"
        adj_close.columns.name = "ticker"

        stacked = adj_close.stack(future_stack=True).rename("adj_close").reset_index()
        stacked_vol = volume.stack(future_stack=True).rename("volume").reset_index()

        df = stacked.merge(stacked_vol, on=["date", "ticker"])
        df = df.dropna(subset=["adj_close"])
        df["date"] = pd.DatetimeIndex(df["date"])
        df = df.set_index(["date", "ticker"]).sort_index()
        return df

    def _log_hash(self, path: Path):
        log_path = self.cache_dir / "run_hashes.log"
        h = hashlib.md5(path.read_bytes()).hexdigest()
        with open(log_path, "a") as f:
            f.write(f"{datetime.utcnow().isoformat()} {path.name} md5={h}\n")
