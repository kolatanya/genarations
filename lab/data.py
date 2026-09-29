"""
Lab market data: years of 5-minute bars for a whole stock universe, aligned on
one regular-session grid, plus the "senses" the bots can use:

  * SPY / QQQ (market mood) and the VIX fear index (previous day's close)
  * Earnings dates (yfinance) and Fed (FOMC) announcement days
  * Headline sentiment from Alpaca's free news feed

Everything is cached under data_cache/ so only the first run downloads.
If Alpaca is unreachable a clearly-labelled synthetic market is generated.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import config
from broker.base import BrokerError

log = logging.getLogger(__name__)

ET = "America/New_York"
SESSION_OPEN_MIN = 9 * 60 + 30
SESSION_CLOSE_MIN = 16 * 60

# Fed rate-decision (FOMC statement) days. Published a year ahead - update yearly.
FOMC_DAYS: frozenset[date] = frozenset(date.fromisoformat(d) for d in (
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
))

# Tiny finance sentiment lexicon (free, instant, no API costs).
_POSITIVE = frozenset("""beat beats beating surge surges surged soar soars soared jump jumps jumped rally rallies rallied
gain gains gained record upgrade upgrades upgraded outperform outperforms bullish strong stronger growth grows raise
raises raised boost boosts boosted exceed exceeds exceeded profit profits win wins approval approved partnership
buyback dividend expands expansion breakthrough tops topped rebound rebounds optimistic""".split())
_NEGATIVE = frozenset("""miss misses missed plunge plunges plunged drop drops dropped fall falls fell slump slumps slumped
downgrade downgrades downgraded underperform bearish weak weaker cut cuts cutting lawsuit sue sued probe investigation
fraud recall recalls layoff layoffs loss losses warning warns warned decline declines declined slide slides tumble
tumbles tumbled crash crashes fine fined ban banned delay delays halt halted risk risks concern concerns""".split())
_WORD = re.compile(r"[a-z]+")


def headline_sentiment(text: str) -> float:
    """-1 (very negative) .. +1 (very positive) from word counts."""
    words = _WORD.findall(text.lower())
    pos = sum(w in _POSITIVE for w in words)
    neg = sum(w in _NEGATIVE for w in words)
    return (pos - neg) / (pos + neg + 1.0)


@dataclass
class LabData:
    """All symbols on one common 5-minute grid (exchange time, regular session)."""

    symbols: list[str]
    index: pd.DatetimeIndex                 # naive America/New_York timestamps
    open: np.ndarray                        # [n_symbols, n_bars]
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    context: dict[str, pd.DataFrame]        # SPY / QQQ bars on the same grid
    vix: pd.Series                          # daily VIX close (indexed by date)
    earnings: dict[str, list[date]] = field(default_factory=dict)
    news: pd.DataFrame | None = None        # columns: time (naive ET), symbol, score
    source: str = "alpaca-sip"
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        idx = self.index
        self.dates = np.array(idx.date)
        day_codes, _ = pd.factorize(self.dates)
        self.day_id = day_codes.astype(np.int64)
        self.minute_of_day = ((idx.hour * 60 + idx.minute) - SESSION_OPEN_MIN).to_numpy().astype(np.int64)
        nxt = np.append(self.day_id[1:], -1)
        self.last_bar_of_day = (self.day_id != nxt)
        self.unique_days = pd.unique(self.dates)

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_bars(self) -> int:
        return len(self.index)

    def day_bounds(self) -> np.ndarray:
        """[n_days, 2] array of (first_bar, last_bar_exclusive) per trading day."""
        starts = np.flatnonzero(np.r_[True, self.day_id[1:] != self.day_id[:-1]])
        ends = np.r_[starts[1:], self.n_bars]
        return np.stack([starts, ends], axis=1)

    def slice_days(self, first_day: int, last_day: int) -> tuple[int, int]:
        """Bar range [t0, t1) covering trading days first_day..last_day-1."""
        bounds = self.day_bounds()
        last_day = min(last_day, len(bounds))
        return int(bounds[first_day, 0]), int(bounds[last_day - 1, 1])


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_lab_data(
    symbols: list[str] | None = None,
    days: int = config.LAB_HISTORY_DAYS,
    *,
    synthetic: bool = False,
    seed: int | None = None,
    progress=None,
) -> LabData:
    """Universe + context bars, VIX, events and news. `progress(msg)` gets status updates."""
    symbols = list(symbols or config.LAB_UNIVERSE)
    say = progress or (lambda msg: None)
    if synthetic:
        return synthetic_lab_data(symbols, days=min(days, 180), seed=seed)

    end = datetime.now(timezone.utc) - timedelta(minutes=16)   # free plan: SIP data older than 15 min
    start = end - timedelta(days=days)
    try:
        raw = _cached_alpaca_bars(symbols + config.LAB_CONTEXT, start, end, say)
    except BrokerError as exc:
        log.warning("Alpaca history unavailable: %s", exc)
        data = synthetic_lab_data(symbols, days=min(days, 180), seed=seed)
        data.notes.append(f"Alpaca history unavailable ({exc}) - using SYNTHETIC prices")
        return data

    missing = [s for s in symbols + config.LAB_CONTEXT if s not in raw or raw[s].empty]
    symbols = [s for s in symbols if s not in missing]
    if not symbols or any(c in missing for c in config.LAB_CONTEXT):
        raise BrokerError(f"Missing history for {missing}")

    grid = raw["SPY"].index  # the most liquid ticker defines the session grid
    arrays = {k: np.empty((len(symbols), len(grid))) for k in ("open", "high", "low", "close", "volume")}
    for i, sym in enumerate(symbols):
        aligned = _align(raw[sym], grid)
        for k in arrays:
            arrays[k][i] = aligned[k].to_numpy()
    context = {c: _align(raw[c], grid) for c in config.LAB_CONTEXT}

    say("Loading VIX, earnings dates and news...")
    first_day, last_day = grid[0].date(), grid[-1].date()
    data = LabData(
        symbols=symbols, index=grid, context=context,
        vix=_load_vix(first_day, last_day, context["SPY"]),
        earnings=_load_earnings(symbols),
        news=_load_news(symbols, grid[0], grid[-1], say) if config.LAB_NEWS else None,
        source="alpaca-sip",
        **arrays,
    )
    if missing:
        data.notes.append(f"No history for {', '.join(missing)} - skipped")
    return data


def _align(df: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex to the common grid; bars with no trades carry the last close with zero volume."""
    out = df.reindex(grid)
    close = out["close"].ffill().bfill()
    out["close"] = close
    for col in ("open", "high", "low"):
        out[col] = out[col].fillna(close)
    out["volume"] = out["volume"].fillna(0.0)
    return out


def _cached_alpaca_bars(symbols: list[str], start: datetime, end: datetime, say) -> dict[str, pd.DataFrame]:
    config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(",".join(sorted(symbols)).encode()).hexdigest()[:8]
    cache = config.DATA_CACHE_DIR / f"lab_5m_{key}_{start:%Y%m%d}_{end:%Y%m%d}.pkl"
    if cache.is_file():
        try:
            return pd.read_pickle(cache)
        except Exception as exc:  # corrupt cache -> re-download
            log.warning("Ignoring corrupt cache %s: %s", cache.name, exc)

    if not config.ALPACA_API_KEY or not config.ALPACA_SECRET_KEY:
        raise BrokerError("Alpaca keys missing (needed for years of 5-minute history)")
    try:
        from alpaca.data.enums import Adjustment, DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise BrokerError("alpaca-py is not installed") from exc

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    out: dict[str, pd.DataFrame] = {}
    chunk = 4
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        say(f"Downloading 2 years of 5-minute bars: {', '.join(batch)} ({i + len(batch)}/{len(symbols)})...")
        try:
            df = client.get_stock_bars(StockBarsRequest(
                symbol_or_symbols=batch, timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start, end=end, feed=DataFeed.SIP, adjustment=Adjustment.ALL,
            )).df
        except Exception as exc:
            raise BrokerError(f"Alpaca bars download failed: {exc}") from exc
        for sym in batch:
            if isinstance(df.index, pd.MultiIndex) and sym in df.index.get_level_values("symbol"):
                out[sym] = _regular_session(df.xs(sym, level="symbol"))
    pd.to_pickle(out, cache)
    for old in config.DATA_CACHE_DIR.glob(f"lab_5m_{key}_*.pkl"):  # keep only the newest download
        if old != cache:
            old.unlink(missing_ok=True)
    return out


def _regular_session(df: pd.DataFrame) -> pd.DataFrame:
    idx = pd.DatetimeIndex(df.index).tz_convert(ET)
    minutes = idx.hour * 60 + idx.minute
    keep = (minutes >= SESSION_OPEN_MIN) & (minutes < SESSION_CLOSE_MIN) & (idx.dayofweek < 5)
    out = df.loc[keep, ["open", "high", "low", "close", "volume"]].astype(float)
    out.index = idx[keep].tz_localize(None)
    return out[~out.index.duplicated(keep="last")].sort_index()


def _load_vix(first: date, last: date, spy: pd.DataFrame) -> pd.Series:
    cache = config.DATA_CACHE_DIR / f"lab_vix_{last:%Y%m%d}.csv"
    try:
        if cache.is_file():
            s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        else:
            import yfinance as yf

            logging.getLogger("yfinance").setLevel(logging.CRITICAL)
            raw = yf.Ticker("^VIX").history(start=(first - timedelta(days=10)).isoformat(),
                                            end=(last + timedelta(days=1)).isoformat(), interval="1d")
            if raw.empty:
                raise ValueError("empty")
            s = raw["Close"]
            s.index = pd.DatetimeIndex(s.index).tz_localize(None).normalize()
            s.to_frame("vix").to_csv(cache)
        s.index = pd.DatetimeIndex(s.index).normalize()
        return s.astype(float)
    except Exception as exc:  # fall back to SPY realised volatility as a fear proxy
        log.warning("VIX unavailable (%s) - using SPY realised volatility instead", exc)
        daily = spy["close"].groupby(spy.index.date).last()
        rv = np.log(daily).diff().rolling(20).std() * np.sqrt(252) * 100
        rv.index = pd.DatetimeIndex(rv.index)
        return rv.bfill()


def _load_earnings(symbols: list[str]) -> dict[str, list[date]]:
    cache = config.DATA_CACHE_DIR / "lab_earnings.json"
    try:
        if cache.is_file() and (datetime.now().timestamp() - cache.stat().st_mtime) < 7 * 86400:
            data = json.loads(cache.read_text())
            if all(s in data for s in symbols):
                return {s: [date.fromisoformat(d) for d in data[s]] for s in symbols}
    except (OSError, ValueError):
        pass
    out: dict[str, list[date]] = {}
    try:
        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        for s in symbols:
            try:
                df = yf.Ticker(s).get_earnings_dates(limit=16)
                out[s] = sorted({ts.date() for ts in pd.DatetimeIndex(df.index)}) if df is not None else []
            except Exception:
                out[s] = []
        cache.write_text(json.dumps({s: [d.isoformat() for d in v] for s, v in out.items()}))
    except Exception as exc:
        log.warning("Earnings calendar unavailable: %s", exc)
    return out


def _load_news(symbols: list[str], first: pd.Timestamp, last: pd.Timestamp, say) -> pd.DataFrame | None:
    """Headline sentiment per (time, symbol). Cached; incremental top-ups are cheap."""
    cache = config.DATA_CACHE_DIR / "lab_news.pkl"
    have: pd.DataFrame | None = None
    if cache.is_file():
        try:
            have = pd.read_pickle(cache)
        except Exception:
            have = None
    start = first.tz_localize(ET).tz_convert("UTC")
    if have is not None and len(have) and have["time"].max() >= last - pd.Timedelta(days=2) \
            and have["time"].min() <= first + pd.Timedelta(days=7):
        return have
    if have is not None and len(have) and have["time"].min() <= first + pd.Timedelta(days=7):
        start = have["time"].max().tz_localize(ET).tz_convert("UTC")
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        rows: list[tuple[pd.Timestamp, str, float]] = []
        wanted = set(symbols)
        end = datetime.now(timezone.utc)
        for i in range(0, len(symbols), 5):
            batch = symbols[i:i + 5]
            say(f"Downloading news headlines: {', '.join(batch)} ({i + len(batch)}/{len(symbols)})...")
            ns = client.get_news(NewsRequest(symbols=",".join(batch), start=start.to_pydatetime(), end=end, limit=None,
                                             exclude_contentless=False))
            for item in ns.data.get("news", []):
                score = headline_sentiment(f"{item.headline} {item.summary or ''}")
                t = pd.Timestamp(item.created_at).tz_convert(ET).tz_localize(None)
                for sym in set(item.symbols or []) & wanted & set(batch):
                    rows.append((t, sym, score))
        new = pd.DataFrame(rows, columns=["time", "symbol", "score"])
        merged = pd.concat([have, new]) if have is not None else new
        merged = merged.drop_duplicates().sort_values("time").reset_index(drop=True)
        merged.to_pickle(cache)
        return merged
    except Exception as exc:
        log.warning("News unavailable: %s", exc)
        return have


# --------------------------------------------------------------------------- #
# Synthetic market (tests / offline rehearsal / monkey baseline shape)
# --------------------------------------------------------------------------- #
def session_grid(first: date, n_days: int) -> pd.DatetimeIndex:
    days = pd.bdate_range(first, periods=n_days)
    minutes = np.arange(SESSION_OPEN_MIN, SESSION_CLOSE_MIN, 5)
    stamps = [d + pd.Timedelta(minutes=int(m)) for d in days for m in minutes]
    return pd.DatetimeIndex(stamps)


def synthetic_lab_data(symbols: list[str], days: int = 120, seed: int | None = None) -> LabData:
    """Correlated random walks with a shared market factor. Clearly labelled; no real patterns."""
    rng = np.random.default_rng(seed if seed is not None else 7)
    n_days = max(int(days * 5 / 7), 30)
    grid = session_grid(date(2025, 1, 6), n_days)
    n = len(grid)
    bar_vol = 0.25 / np.sqrt(252 * 78)
    market = rng.standard_t(4, n) / np.sqrt(2) * bar_vol
    prices: dict[str, pd.DataFrame] = {}

    def walk(beta: float, idio: float, start: float) -> pd.DataFrame:
        r = beta * market + rng.standard_t(4, n) / np.sqrt(2) * bar_vol * idio
        close = start * np.exp(np.cumsum(r))
        open_ = np.r_[start, close[:-1]] * np.exp(rng.normal(0, bar_vol * 0.2, n))
        wig = np.abs(rng.normal(0, bar_vol * 0.5, (2, n)))
        return pd.DataFrame({
            "open": open_, "close": close,
            "high": np.maximum(open_, close) * np.exp(wig[0]),
            "low": np.minimum(open_, close) * np.exp(-wig[1]),
            "volume": rng.lognormal(10, 0.5, n),
        }, index=grid)

    for s in symbols:
        prices[s] = walk(rng.uniform(0.7, 1.4), rng.uniform(0.8, 2.0), rng.uniform(50, 500))
    context = {"SPY": walk(1.0, 0.1, 500.0), "QQQ": walk(1.2, 0.3, 450.0)}
    arrays = {k: np.stack([prices[s][k].to_numpy() for s in symbols]) for k in ("open", "high", "low", "close", "volume")}
    vix = pd.Series(rng.uniform(12, 30, n_days), index=pd.DatetimeIndex(pd.unique(grid.normalize())))
    return LabData(symbols=list(symbols), index=grid, context=context, vix=vix, source="synthetic",
                   notes=["SYNTHETIC random-walk prices - not real market data"], **arrays)


def shuffle_days(data: LabData, rng: np.random.Generator) -> LabData:
    """
    The MONKEY market: every symbol's trading days are shuffled independently,
    then prices are rebuilt from the returns. Intraday shapes and volatility
    survive; every real cross-day and cross-stock pattern is destroyed.
    """
    bounds = data.day_bounds()
    n_days = len(bounds)
    day_len = bounds[:, 1] - bounds[:, 0]

    def rebuild(o, h, l, c, v):  # noqa: E741
        order = rng.permutation(n_days)
        # only swap days of equal length (half-days stay put) so the grid is unchanged
        for L in np.unique(day_len):
            same = np.flatnonzero(day_len == L)
            order[same] = rng.permutation(same)
        prev = np.r_[c[0], c[:-1]]
        src = np.concatenate([np.arange(bounds[s, 0], bounds[s, 1]) for s in order])
        new_close = c[0] * np.cumprod((c / prev)[src])
        new_prev = np.r_[c[0], new_close[:-1]]
        return (new_prev * (o / prev)[src], new_prev * (h / prev)[src],
                new_prev * (l / prev)[src], new_close, v[src])

    arrays = {k: np.empty_like(getattr(data, k)) for k in ("open", "high", "low", "close", "volume")}
    for i in range(data.n_symbols):
        res = rebuild(data.open[i], data.high[i], data.low[i], data.close[i], data.volume[i])
        for k, a in zip(("open", "high", "low", "close", "volume"), res):
            arrays[k][i] = a
    context = {}
    for name, df in data.context.items():
        res = rebuild(*(df[k].to_numpy() for k in ("open", "high", "low", "close", "volume")))
        context[name] = pd.DataFrame(dict(zip(("open", "high", "low", "close", "volume"), res)), index=data.index)
    vix = pd.Series(rng.permutation(data.vix.to_numpy()), index=data.vix.index)
    return LabData(symbols=list(data.symbols), index=data.index, context=context, vix=vix,
                   earnings={}, news=None, source=f"monkey({data.source})",
                   notes=["MONKEY market: days shuffled, all real patterns destroyed"], **arrays)
