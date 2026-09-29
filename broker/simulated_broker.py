"""
SimulatedBroker - historical data + simulated fills for the fast evolution loop.

Data comes from yfinance (cached to CSV so repeat runs are instant and work
offline). If yfinance is unreachable, the broker can fall back to a clearly
labelled SYNTHETIC market (regime-switching random walks) so the evolution
demo still runs - handy for rehearsing a recording on a plane.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import config
from broker.base import (
    BrokerError,
    CostModel,
    DateLike,
    MarketData,
    MarketDataSource,
    is_crypto,
    normalize_ohlcv,
)

log = logging.getLogger(__name__)

# Rough annualised volatility per symbol for the synthetic generator.
_SYNTHETIC_PROFILE: dict[str, tuple[float, float]] = {
    # symbol: (starting price, annual volatility)
    "AAPL": (40.0, 0.30),
    "NVDA": (5.0, 0.50),
    "TSLA": (20.0, 0.60),
    "BTC-USD": (4000.0, 0.65),
}


def _to_date(value: DateLike | None) -> date:
    if value is None:
        return date.today()
    return pd.Timestamp(value).date()


class SimulatedBroker(MarketDataSource):
    """yfinance-backed data source with a commission/slippage fill model."""

    name = "yfinance"

    def __init__(
        self,
        cache_dir: Path = config.DATA_CACHE_DIR,
        costs: CostModel | None = None,
        *,
        use_cache: bool = True,
        force_synthetic: bool = False,
        allow_synthetic_fallback: bool = True,
        seed: int | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.costs = costs or CostModel.from_config()
        self.use_cache = use_cache
        self.force_synthetic = force_synthetic
        self.allow_synthetic_fallback = allow_synthetic_fallback
        self.seed = seed

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def load_market_data(
        self,
        symbols: list[str],
        start: DateLike = config.HISTORY_START,
        end: DateLike | None = config.HISTORY_END,
        min_bars: int = config.INDICATOR_WARMUP_BARS + 60,
    ) -> MarketData:
        """Fetch every symbol; drop the ones that fail; fall back to synthetic if all fail."""
        if self.force_synthetic:
            return self._synthetic_market(symbols, start, end, reason="--synthetic flag")

        bars: dict[str, pd.DataFrame] = {}
        notes: list[str] = []
        for symbol in symbols:
            try:
                df = self.get_history(symbol, start, end)
            except BrokerError as exc:
                notes.append(f"{symbol}: {exc}")
                log.warning("Skipping %s: %s", symbol, exc)
                continue
            if len(df) < min_bars:
                notes.append(f"{symbol}: only {len(df)} bars (need {min_bars}) - skipped")
                continue
            bars[symbol] = df

        if bars:
            return MarketData(bars=bars, source="yfinance", notes=notes)
        if self.allow_synthetic_fallback:
            return self._synthetic_market(symbols, start, end, reason="yfinance unavailable", extra_notes=notes)
        raise BrokerError("No market data could be loaded:\n  " + "\n  ".join(notes))

    def get_history(self, symbol: str, start: DateLike, end: DateLike | None = None) -> pd.DataFrame:
        """OHLCV bars (config.BAR_INTERVAL) for one symbol, served from the CSV cache when possible."""
        start_d, end_d = _to_date(start), _to_date(end)
        cache_file = self.cache_dir / f"{symbol.replace('/', '-')}_{start_d}_{end_d}_{config.BAR_INTERVAL}.csv"

        if self.use_cache and cache_file.is_file():
            try:
                cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                return normalize_ohlcv(cached, symbol)
            except (OSError, ValueError, BrokerError) as exc:
                log.warning("Ignoring corrupt cache %s: %s", cache_file.name, exc)

        df = self._download_yfinance(symbol, start_d, end_d)
        if self.use_cache:
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(cache_file)
            except OSError as exc:
                log.warning("Could not write cache %s: %s", cache_file, exc)
        return df

    # ------------------------------------------------------------------ #
    # yfinance
    # ------------------------------------------------------------------ #
    @staticmethod
    def _download_yfinance(symbol: str, start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise BrokerError("yfinance is not installed (pip install yfinance)") from exc

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)  # keep the rich UI clean
        try:
            raw = yf.Ticker(symbol).history(
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # yfinance `end` is exclusive
                interval=config.BAR_INTERVAL,
                auto_adjust=True,
            )
        except Exception as exc:  # yfinance raises a zoo of exception types
            raise BrokerError(f"yfinance download failed: {exc}") from exc
        if raw is None or raw.empty:
            raise BrokerError("yfinance returned no data (offline, rate-limited or bad ticker)")
        return normalize_ohlcv(raw, symbol)

    # ------------------------------------------------------------------ #
    # Synthetic market (offline fallback / rehearsals)
    # ------------------------------------------------------------------ #
    def _synthetic_market(
        self,
        symbols: list[str],
        start: DateLike,
        end: DateLike | None,
        *,
        reason: str,
        extra_notes: list[str] | None = None,
    ) -> MarketData:
        bars = {s: self.generate_synthetic(s, start, end, self.seed) for s in symbols}
        notes = list(extra_notes or [])
        notes.append(f"SYNTHETIC random-walk prices ({reason}) - not real market data")
        return MarketData(bars=bars, source="synthetic", notes=notes)

    @staticmethod
    def generate_synthetic(symbol: str, start: DateLike, end: DateLike | None, seed: int | None = None) -> pd.DataFrame:
        """
        Regime-switching geometric random walk with fat tails.

        Deterministic per (symbol, seed) so a recorded episode can be replayed.
        """
        digest = hashlib.sha256(f"{symbol}|{seed}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))

        crypto = is_crypto(symbol)
        if config.IS_INTRADAY:
            step = config.BAR_MINUTES
            dates = pd.date_range(pd.Timestamp(_to_date(start)), pd.Timestamp(_to_date(end)) + pd.Timedelta(days=1),
                                  freq=f"{step}min", inclusive="left")
            if not crypto:  # US regular session, 13:30-20:00 UTC, weekdays
                minutes = dates.hour * 60 + dates.minute
                dates = dates[(dates.dayofweek < 5) & (minutes >= 13 * 60 + 30) & (minutes < 20 * 60)]
            bars_per_day = (1440 if crypto else 390) / step
            periods_per_year = (365 if crypto else 252) * bars_per_day
        else:
            dates = pd.date_range(_to_date(start), _to_date(end), freq="D" if crypto else "B")
            periods_per_year = 365 if crypto else 252
        n = len(dates)
        if n < 2:
            raise BrokerError(f"Synthetic window for {symbol} is empty")
        start_price, vol = _SYNTHETIC_PROFILE.get(symbol.upper(), (100.0, 0.35))

        # Markov regimes: bull / chop / bear with sticky transitions.
        regime_drift = np.array([0.45, 0.02, -0.35])
        transition = np.array([[0.985, 0.010, 0.005],
                               [0.010, 0.980, 0.010],
                               [0.010, 0.015, 0.975]])
        regimes = np.empty(n, dtype=int)
        regimes[0] = 0
        draws = rng.random(n)
        cumulative = transition.cumsum(axis=1)
        for k in range(1, n):  # vectorised draws, sequential chain (fast for 20k+ intraday bars)
            regimes[k] = int(np.searchsorted(cumulative[regimes[k - 1]], draws[k]))

        daily_vol = vol / np.sqrt(periods_per_year)
        shocks = rng.standard_t(df=4, size=n) / np.sqrt(2.0)  # t(4) has variance 2
        log_ret = regime_drift[regimes] / periods_per_year + daily_vol * shocks
        close = start_price * np.exp(np.cumsum(log_ret))

        gap = rng.normal(0.0, daily_vol * 0.3, size=n)
        open_ = np.concatenate(([start_price], close[:-1])) * np.exp(gap)
        wick_hi = np.abs(rng.normal(0.0, daily_vol * 0.6, size=n))
        wick_lo = np.abs(rng.normal(0.0, daily_vol * 0.6, size=n))
        high = np.maximum(open_, close) * np.exp(wick_hi)
        low = np.minimum(open_, close) * np.exp(-wick_lo)
        volume = rng.lognormal(mean=15.0, sigma=0.4, size=n)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=dates,
        )
        return normalize_ohlcv(df, symbol)
