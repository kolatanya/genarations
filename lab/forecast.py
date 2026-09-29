"""
FORECASTER ARENA - bots that are scored purely on PREDICTION ACCURACY.

Two contests, both on daily bars for ~100 large US stocks since 2012:

  DIRECTION  "Will this stock close UP tomorrow?"
  BIG MOVE   "Will tomorrow's move be bigger than this stock's recent typical move?"

A forecaster is one rule tree (see lab/gp.py): TRUE means "UP" (or "BIG"),
FALSE means "DOWN" (or "CALM"). It must make a call for every stock on every
day - no cherry-picking easy days.

Accuracy is always compared with the best NAIVE guess (e.g. "always say UP",
which is right ~53% of the time in a bull market). EDGE = accuracy minus that
baseline. Only edge counts.

Evolution: 4 islands x 6 forecasters (3 mutants, 1 crossover child, 1 wildcard),
random 4-of-6 training eras per generation, a validation gate on unseen years,
a Hall of Fame, meteors for stuck/converged islands, and a verified purge.
Final exam: the untouched last ~20% of history, plus a PERMUTATION luck test
(shuffle the answers 1,000 times and see how often random answers score as well).
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import warnings
import weakref
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from engine.evaluator import EvolutionIntegrityError
from lab import gp
from lab.features import FeatureSet, _rsi

log = logging.getLogger(__name__)

FORECAST_UNIVERSE: list[str] = """AAPL MSFT AMZN GOOGL META NVDA TSLA JPM V MA UNH HD PG JNJ XOM CVX KO PEP MRK ABBV LLY PFE
WMT COST BAC WFC C GS MS AXP BLK SCHW INTC AMD AVGO QCOM TXN CSCO ORCL CRM ADBE IBM NFLX DIS CMCSA T VZ NKE MCD SBUX
LOW TGT BKNG AMGN GILD BMY TMO ABT DHR MDT ISRG CVS CI HUM CAT DE HON GE MMM BA LMT RTX UPS FDX UNP CSX NEE DUK SO
AMT PLD SPG COP SLB EOG OXY MO PM CL KMB GIS MDLZ ADP INTU AMAT MU LRCX KLAC ADI PYPL""".split()
START = "2012-01-01"
WARMUP_DAYS = 260
STAGNATION_LIMIT = 15
TARGETS = {"direction": ("UP", "DOWN", "Will it close UP tomorrow?"),
           "volatility": ("BIG", "CALM", "Will tomorrow's move be BIGGER than usual?")}

DAILY_FEATURES: list[tuple[str, str]] = [
    ("ret_1", "yesterday's return"), ("ret_5", "1-week return"), ("ret_20", "1-month return"),
    ("ret_60", "3-month return"), ("ret_120", "6-month return"),
    ("rsi_2", "RSI 2"), ("rsi_14", "RSI 14"),
    ("sma_dev_10", "% from 10-day avg"), ("sma_dev_50", "% from 50-day avg"), ("sma_dev_200", "% from 200-day avg"),
    ("vol_20", "1-month volatility"), ("vol_ratio", "recent vs normal volatility"),
    ("move_size", "yesterday's move vs normal"), ("range_ratio", "yesterday's range vs normal"),
    ("gap", "opening gap"), ("vol_z", "volume spike"),
    ("dist_high", "% below 1-year high"), ("dist_low", "% above 1-year low"),
    ("spy_ret_1", "market yesterday"), ("spy_ret_20", "market 1-month"), ("spy_trend", "market vs 200-day avg"),
    ("spy_vol_20", "market volatility"), ("vix", "VIX fear index"), ("vix_chg_5", "VIX 1-week change"),
    ("rs_rank_20", "1-month strength rank"), ("rs_rank_120", "6-month strength rank"),
    ("dow", "day of week (0=Mon)"),
    # earnings calendar
    ("earn_next", "earnings reaction in tomorrow's move"), ("days_to_earn", "trading days until earnings"),
    ("days_since_earn", "trading days since earnings"),
    # implied volatility (the options market's own forecasts, market-wide)
    ("vix9d_ratio", "VIX 9-day / VIX (near-term fear)"), ("vix_term", "VIX / VIX 3-month (fear curve)"),
    ("vxn", "Nasdaq implied volatility"), ("vvix", "volatility of the VIX"), ("iv_rank", "VIX rank over the past year"),
    # this stock's own earnings history
    ("earn_react", "how big this stock's earnings moves usually are (x its normal move)"),
    ("earn_expected", "earnings tomorrow x how big its earnings moves usually are (0 if no earnings)"),
    # its sector (the 11 GICS sectors, equal-weighted)
    ("sector_ret_1", "its sector's move today"), ("sector_move", "its sector's move today vs normal"),
    ("sector_vol_ratio", "its sector's recent vs normal volatility"),
    ("idio_move", "the stock's own move today, apart from its sector, vs normal"),
]
DAILY_NAMES = [n for n, _ in DAILY_FEATURES]
DAILY_BINARY = frozenset({"earn_next"})
PLAIN = frozenset({"dow", "vix", "vol_z", "vol_ratio", "move_size", "range_ratio", "spy_vol_20", "vol_20",
                   "days_to_earn", "days_since_earn", "vix9d_ratio", "vix_term", "vxn", "vvix", "iv_rank",
                   "earn_react", "earn_expected", "sector_move", "sector_vol_ratio", "idio_move"})


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DailyData:
    symbols: list[str]
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    spy: pd.Series
    vix: pd.Series
    source: str = "yfinance"
    iv: dict = field(default_factory=dict)          # implied-volatility indices (see lab/daily_extra.py)
    earnings: dict = field(default_factory=dict)    # symbol -> earnings timestamps
    sectors: dict = field(default_factory=dict)     # symbol -> GICS sector

    @property
    def n_symbols(self) -> int:
        return len(self.symbols)

    @property
    def n_days(self) -> int:
        return len(self.index)


def load_daily(symbols: list[str] | None = None, start: str = START, progress=None,
               synthetic: bool = False, seed: int | None = None) -> DailyData:
    from lab.daily_extra import IV_TICKERS, iv_context, load_earnings, sp500_sectors, sp500_universe

    say = progress or (lambda m: None)
    if synthetic:
        return _synthetic_daily(list(symbols or FORECAST_UNIVERSE)[:30], seed)
    symbols = list(symbols or sp500_universe(FORECAST_UNIVERSE))
    config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = config.DATA_CACHE_DIR / f"forecast_daily_{len(symbols)}_{date.today():%Y%m%d}.pkl"
    if cache.is_file():
        raw = pd.read_pickle(cache)
    else:
        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        tickers = symbols + ["SPY"] + [t for t in IV_TICKERS.values()]
        parts = []
        for i in range(0, len(tickers), 100):
            say(f"Downloading daily history since {start}: {min(i + 100, len(tickers))}/{len(tickers)} tickers "
                "(one-time per day)...")
            part = yf.download(tickers[i:i + 100], start=start, interval="1d", auto_adjust=True,
                               group_by="ticker", threads=True, progress=False)
            if part is not None and not part.empty:
                parts.append(part)
        if not parts:
            raise EvolutionIntegrityError("yfinance returned no daily data")
        raw = pd.concat(parts, axis=1)
        raw.to_pickle(cache)
        for old in config.DATA_CACHE_DIR.glob(f"forecast_daily_{len(symbols)}_*.pkl"):
            if old != cache:
                old.unlink(missing_ok=True)
    have = set(raw.columns.get_level_values(0))
    spy = raw["SPY"]["Close"].dropna()
    grid = pd.DatetimeIndex(spy.index).tz_localize(None) if spy.index.tz is not None else pd.DatetimeIndex(spy.index)
    spy.index = grid
    syms = [s for s in symbols if s in have and raw[s]["Close"].notna().sum() > 300]
    arrays = {k: np.full((len(syms), len(grid)), np.nan) for k in ("open", "high", "low", "close", "volume")}
    for i, s in enumerate(syms):
        df = raw[s].copy()
        df.index = grid if len(df) == len(grid) else pd.DatetimeIndex(df.index).tz_localize(None)
        df = df.reindex(grid)
        first = df["Close"].first_valid_index()
        df.loc[first:, "Close"] = df.loc[first:, "Close"].ffill()      # fill halts, never before listing
        for k, col in (("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")):
            arrays[k][i] = df[col].to_numpy(dtype=float)
    iv = iv_context(raw, grid)
    earnings = load_earnings(syms, say)
    return DailyData(syms, grid, spy=spy, vix=iv["vix"], iv=iv, earnings=earnings, sectors=sp500_sectors(),
                     source="yfinance", **arrays)


def _synthetic_daily(symbols: list[str], seed: int | None) -> DailyData:
    rng = np.random.default_rng(seed if seed is not None else 11)
    grid = pd.bdate_range("2014-01-01", periods=2200)
    T, S = len(grid), len(symbols)
    vol = np.exp(np.cumsum(rng.normal(0, 0.05, (S, T)), axis=1) * 0.2) * 0.015   # volatility clusters
    mkt = rng.normal(0.0003, 0.01, T)
    r = 0.9 * mkt + rng.standard_t(4, (S, T)) / np.sqrt(2) * vol
    close = 50 * np.exp(np.cumsum(r, axis=1))
    open_ = np.concatenate([close[:, :1], close[:, :-1]], axis=1) * np.exp(rng.normal(0, 0.002, (S, T)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (S, T))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (S, T))))
    spy = pd.Series(400 * np.exp(np.cumsum(mkt)), index=grid)
    ny = "America/New_York"
    earnings = {sym: [pd.Timestamp(grid[i]).tz_localize(ny) + pd.Timedelta(hours=16, minutes=5)
                      for i in range(int(rng.integers(5, 60)), T, 63)] for sym in symbols}
    sectors = {sym: f"Sector {i % 3}" for i, sym in enumerate(symbols)}
    return DailyData(list(symbols), grid, open_, high, low, close, rng.lognormal(14, 0.4, (S, T)), spy,
                     pd.Series(rng.uniform(12, 30, T), index=grid), source="synthetic", earnings=earnings,
                     sectors=sectors)


# --------------------------------------------------------------------------- #
# Features and answers
# --------------------------------------------------------------------------- #
@dataclass
class Answers:
    up: np.ndarray        # [S, T] bool: closed up the NEXT day
    big: np.ndarray       # [S, T] bool: next day's move > median of the previous 60 moves
    valid: np.ndarray     # [S, T] bool: an answer exists
    next_ret: np.ndarray  # [S, T] next-day simple return


def _sector_senses(out: np.ndarray, d: DailyData, lr_all: np.ndarray, sd60_all: np.ndarray) -> None:
    """Equal-weighted sector return per day, and each stock's move apart from its sector (all known at the close)."""
    groups: dict[str, list[int]] = {}
    for i, sym in enumerate(d.symbols):
        sec = (d.sectors or {}).get(sym)
        if sec:
            groups.setdefault(sec, []).append(i)
    for members in groups.values():
        block = lr_all[members]
        n_ok = np.isfinite(block).sum(axis=0)
        sec_r = np.where(n_ok >= 3, np.nanmean(np.where(np.isfinite(block), block, np.nan), axis=0), np.nan)
        ser = pd.Series(sec_r)
        sd60 = ser.rolling(60, min_periods=40).std()
        move = (ser.abs() / sd60).to_numpy()
        volr = (ser.rolling(5).std() / sd60).to_numpy()
        for i in members:
            out[DAILY_NAMES.index("sector_ret_1"), i] = sec_r
            out[DAILY_NAMES.index("sector_move"), i] = move
            out[DAILY_NAMES.index("sector_vol_ratio"), i] = volr
            out[DAILY_NAMES.index("idio_move"), i] = np.abs(lr_all[i] - sec_r) / sd60_all[i]


def build_daily_features(d: DailyData) -> tuple[FeatureSet, Answers]:
    S, T = d.n_symbols, d.n_days
    out = np.full((len(DAILY_NAMES), S, T), np.nan, dtype=np.float32)
    idx = d.index

    def put(name, s, v):
        out[DAILY_NAMES.index(name), s] = np.asarray(v, dtype=np.float32)

    spy = d.spy.reindex(idx).ffill()
    spy_r = np.log(spy / spy.shift(1))
    ctx = {"spy_ret_1": spy_r, "spy_ret_20": np.log(spy / spy.shift(20)),
           "spy_trend": spy / spy.rolling(200, min_periods=100).mean() - 1,
           "spy_vol_20": spy_r.rolling(20).std() * np.sqrt(252),
           "vix": d.vix, "vix_chg_5": d.vix.pct_change(5)}
    iv = d.iv or {}
    nan = pd.Series(np.nan, index=idx)
    vix_s = iv.get("vix", d.vix).reindex(idx)
    ctx.update({
        "vix9d_ratio": iv.get("vix9d", nan).reindex(idx) / vix_s,
        "vix_term": vix_s / iv.get("vix3m", nan).reindex(idx),
        "vxn": iv.get("vxn", nan).reindex(idx),
        "vvix": iv.get("vvix", nan).reindex(idx),
        "iv_rank": vix_s.rolling(252, min_periods=120).rank(pct=True),
    })
    r20 = np.full((S, T), np.nan)
    r120 = np.full((S, T), np.nan)
    lr_all = np.full((S, T), np.nan)       # daily log returns, for the sector senses
    sd60_all = np.full((S, T), np.nan)
    next_ret = np.full((S, T), np.nan)
    big = np.zeros((S, T), dtype=bool)
    big_valid = np.zeros((S, T), dtype=bool)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s in range(S):
            c = pd.Series(d.close[s], index=idx)
            o, h, lo, v = (pd.Series(a[s], index=idx) for a in (d.open, d.high, d.low, d.volume))
            lr = np.log(c / c.shift(1))
            for k in (1, 5, 20, 60, 120):
                put(f"ret_{k}", s, np.log(c / c.shift(k)))
            put("rsi_2", s, _rsi(c, 2))
            put("rsi_14", s, _rsi(c, 14))
            for n in (10, 50, 200):
                put(f"sma_dev_{n}", s, c / c.rolling(n, min_periods=n).mean() - 1)
            sd60 = lr.rolling(60, min_periods=40).std()
            put("vol_20", s, lr.rolling(20).std() * np.sqrt(252))
            put("vol_ratio", s, lr.rolling(5).std() / sd60)
            put("move_size", s, lr.abs() / sd60)
            rng_pct = (h - lo) / c
            put("range_ratio", s, rng_pct / rng_pct.rolling(20).mean())
            put("gap", s, o / c.shift(1) - 1)
            put("vol_z", s, (v - v.rolling(20).mean()) / v.rolling(20).std())
            put("dist_high", s, c / h.rolling(252, min_periods=120).max() - 1)
            put("dist_low", s, c / lo.rolling(252, min_periods=120).min() - 1)
            for name, ser in ctx.items():
                put(name, s, ser.to_numpy())
            put("dow", s, idx.dayofweek.to_numpy())
            from lab.daily_extra import earnings_features

            e_next, e_to, e_since = earnings_features(d.earnings.get(d.symbols[s], []), idx)
            put("earn_next", s, e_next)
            put("days_to_earn", s, e_to)
            put("days_since_earn", s, e_since)
            lr_all[s], sd60_all[s] = lr.to_numpy(), sd60.to_numpy()
            r20[s] = np.log(c / c.shift(20)).to_numpy()
            r120[s] = np.log(c / c.shift(120)).to_numpy()
            nr = (c.shift(-1) / c - 1).to_numpy()
            next_ret[s] = nr
            typical = (c / c.shift(1) - 1).abs().rolling(60, min_periods=40).median().to_numpy()  # known today
            big[s] = np.abs(nr) > typical
            big_valid[s] = np.isfinite(nr) & np.isfinite(typical)
            from lab.daily_extra import earnings_reaction

            react = earnings_reaction(e_next, nr, typical)
            put("earn_react", s, react)
            put("earn_expected", s, np.where(e_next > 0, react, 0.0))
        _sector_senses(out, d, lr_all, sd60_all)
        for name, arr in (("rs_rank_20", r20), ("rs_rank_120", r120)):
            ok = np.isfinite(arr)
            ranks = np.where(ok, arr, np.inf).argsort(axis=0).argsort(axis=0).astype(float)
            counts = ok.sum(axis=0)
            out[DAILY_NAMES.index(name)] = np.where(ok, ranks / np.maximum(counts - 1, 1), np.nan).astype(np.float32)
    valid = np.isfinite(next_ret)
    fs = FeatureSet(list(DAILY_NAMES), out, np.zeros((S, T)), np.zeros((S, T), dtype=bool), np.zeros(1))
    return fs, Answers(next_ret > 0, big, valid & big_valid, next_ret)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForecastScore:
    fitness: float
    accuracy: float
    baseline: float        # accuracy of the best naive guess in the same windows
    edge: float            # accuracy - baseline
    eras_positive: int
    n_eras: int
    says_true: float       # share of calls that were UP / BIG
    complexity: int
    n: int

    def rank_key(self, is_alpha: bool = False) -> tuple:
        return (self.fitness, self.accuracy, is_alpha)


class ForecastScorer:
    def __init__(self, fs: FeatureSet, answers: Answers, target: str) -> None:
        if target not in TARGETS:
            raise ValueError(f"target must be one of {list(TARGETS)}")
        self.fs, self.ans, self.target = fs, answers, target
        self.label = answers.up if target == "direction" else answers.big
        self.valid = answers.valid if target == "volatility" else np.isfinite(answers.next_ret)
        self._ev: dict[tuple[int, int], gp.TreeEvaluator] = {}

    def predict(self, tree: gp.Node, t0: int, t1: int) -> np.ndarray:
        ev = self._ev.get((t0, t1))
        if ev is None:
            if len(self._ev) > 20:
                self._ev.clear()
            ev = self._ev[(t0, t1)] = gp.TreeEvaluator(self.fs, t0, t1)
        return ev(tree)

    def window_stats(self, pred: np.ndarray, t0: int, t1: int) -> tuple[float, float, int, float]:
        lab, ok = self.label[:, t0:t1], self.valid[:, t0:t1]
        n = int(ok.sum())
        if n == 0:
            return 0.5, 0.5, 0, 0.0
        acc = float((pred[ok] == lab[ok]).mean())
        p = float(lab[ok].mean())
        return acc, max(p, 1 - p), n, float(pred[ok].mean())

    def score(self, tree: gp.Node, windows: list[tuple[int, int]]) -> ForecastScore:
        accs, bases, ns, trues = [], [], [], []
        for a, b in windows:
            acc, base, n, tr = self.window_stats(self.predict(tree, a, b), a, b)
            accs.append(acc), bases.append(base), ns.append(n), trues.append(tr)
        w = np.array(ns, dtype=float) / max(sum(ns), 1)
        acc = float(np.dot(w, accs))
        base = float(np.dot(w, bases))
        edges = [x - y for x, y in zip(accs, bases)]
        pos = sum(e > 0 for e in edges)
        size = gp.size(tree)
        mean_edge = float(np.mean(edges)) * 100
        factor = (0.5 + 0.5 * pos / len(edges)) / (1 + config.LAB_COMPLEXITY_PENALTY * size)
        fitness = mean_edge * factor if mean_edge >= 0 else mean_edge / factor
        return ForecastScore(fitness, acc, base, acc - base, pos, len(edges), float(np.dot(w, trues)), size, sum(ns))

    def luck_test(self, pred: np.ndarray, t0: int, t1: int, runs: int = 1000, seed: int = 0) -> float:
        """P-value: how often shuffled answers (same stock, days in random order) score at least as well."""
        rng = np.random.default_rng(seed)
        lab, ok = self.label[:, t0:t1], self.valid[:, t0:t1]
        real, _, _, _ = self.window_stats(pred, t0, t1)
        hits = 0
        for _ in range(runs):
            perm = rng.permutation(t1 - t0)
            lp, okp = lab[:, perm], ok[:, perm]
            m = ok & okp
            if (pred[m] == lp[m]).mean() >= real:
                hits += 1
        return (hits + 1) / (runs + 1)


# --------------------------------------------------------------------------- #
# Evolution
# --------------------------------------------------------------------------- #
@dataclass
class Forecaster:
    bot_id: int
    tree: gp.Node
    island: int
    role: str
    born: int
    score: ForecastScore | None = None


@dataclass
class Island:
    idx: int
    alpha: Forecaster
    rate: float = config.MUTATION_RATE
    reign: int = 0


@dataclass
class ForecastResult:
    name: str
    accuracy: float
    baseline: float
    edge: float
    p_value: float | None
    says_true: float
    extra: str


@dataclass
class ForecastSummary:
    target: str
    generations: list[int]
    train_acc: list[float]        # best Alpha training accuracy per generation
    val_acc: list[float]          # Hall-of-Fame leader validation accuracy per generation
    baseline_train: float
    baseline_val: float
    hall: list[dict]
    champion: dict | None
    results: list[ForecastResult]
    windows: dict[str, str]
    trials: int
    equity: dict[str, np.ndarray] = field(default_factory=dict)
    move_ratio: tuple[float, float] | None = None   # avg |move| on BIG calls, on CALM calls
    chart: str | None = None
    checkpoint: str = ""


def textbook_tree(target: str) -> gp.Node:
    with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
        if target == "direction":
            return gp.cond("ret_5", False, 0.25)          # short-term losers bounce (reversal)
        # big move yesterday -> big move tomorrow; earnings reaction tomorrow -> big move (84% of the time)
        return gp.Or(gp.cond("move_size", True, 0.6), gp.cond("earn_next", True, 0.5))


class ForecastEngine:
    def __init__(self, fs: FeatureSet, answers: Answers, index: pd.DatetimeIndex, target: str, display, *,
                 rng: random.Random | None = None, n_islands: int = 4, bots: int = 6, n_eras: int = 6,
                 eras_per_gen: int = 4, save: bool = True) -> None:
        self.scorer = ForecastScorer(fs, answers, target)
        self.fs, self.index, self.target, self.display = fs, index, target, display
        self.rng = rng or random.Random()
        self.n_islands, self.bots = n_islands, bots
        self.save = save
        T = len(index)
        usable = T - WARMUP_DAYS - 1
        a = WARMUP_DAYS
        b = a + int(usable * 0.6)
        c = b + int(usable * 0.2)
        self.train, self.val, self.test = (a, b), (b, c), (c, T - 1)
        edges = np.linspace(a, b, n_eras + 1).astype(int)
        self.eras = [(int(edges[i]), int(edges[i + 1])) for i in range(n_eras)]
        self.eras_per_gen = min(eras_per_gen, n_eras)
        fs.fit_quantiles(*self.train)
        self.islands: list[Island] = []
        self.hall: list[dict] = []
        self._val: dict[str, ForecastScore] = {}
        self._next = 1
        self.trials = 0
        self.train_acc: list[float] = []
        self.val_acc: list[float] = []
        self.gens: list[int] = []

    def dates(self, w: tuple[int, int]) -> str:
        return f"{self.index[w[0]].date()} → {self.index[w[1] - 1].date()}"

    def _new(self, tree, island, role, gen) -> Forecaster:
        self._next += 1
        return Forecaster(self._next - 1, tree, island, role, gen)

    def val_score(self, tree) -> ForecastScore:
        k = gp.key(tree)
        if k not in self._val:
            self._val[k] = self.scorer.score(tree, [self.val])
        return self._val[k]

    # ------------------------------------------------------------------ #
    def run(self, generations: int) -> ForecastSummary:
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            depth = config.LAB_MAX_TREE_DEPTH
            for i in range(self.n_islands):
                tree = textbook_tree(self.target) if i == 0 else gp.random_tree(self.rng, depth)
                self.islands.append(Island(i, self._new(tree, i, "ALPHA", 1)))
            try:
                for gen in range(1, generations + 1):
                    self._generation(gen, generations)
            except KeyboardInterrupt:
                pass
            return self._finalize()

    def _generation(self, gen: int, last: int) -> None:
        depth = config.LAB_MAX_TREE_DEPTH
        eras = [self.eras[i] for i in sorted(self.rng.sample(range(len(self.eras)), self.eras_per_gen))]
        rows, tombstones = [], []
        for isl in self.islands:
            alpha = isl.alpha
            kids = self._spawn(isl, gen, depth)
            pool = [alpha, *kids]
            for f in pool:
                f.score = self.scorer.score(f.tree, eras)
                self.trials += 1
            f = None  # a lingering loop variable would keep one purged bot alive
            ranked = sorted(pool, key=lambda f: f.score.rank_key(f is alpha), reverse=True)
            top = ranked[0]
            status = "DEFENDED"
            if top is not alpha:
                v_top, v_al = self.val_score(top.tree), self.val_score(alpha.tree)
                status = "USURPED" if v_top.edge >= v_al.edge else "BLOCKED_VALIDATION"
            winner = top if status == "USURPED" else alpha
            if winner is alpha:
                isl.rate = min(config.MUTATION_RATE_MAX, isl.rate * config.MUTATION_STUCK_BOOST)
                isl.reign += 1
            else:
                isl.rate = max(config.MUTATION_RATE_MIN, isl.rate * config.MUTATION_PROGRESS_DECAY)
                isl.reign = 1
            losers = [(f.bot_id, f.role, f.score) for f in ranked if f is not winner]
            tombstones += [weakref.ref(f) for f in pool if f is not winner]
            role = winner.role if winner is not alpha else "ALPHA"
            winner.role = "ALPHA"
            isl.alpha = winner
            rows.append({"island": isl.idx, "bot": winner.bot_id, "role": role, "status": status,
                         "score": winner.score, "val": self.val_score(winner.tree), "losers": losers,
                         "challenger": top.bot_id if top is not alpha else None, "rate": isl.rate})
            pool.clear(), ranked.clear(), kids.clear()
            del top, alpha, winner
        gc.collect()
        lingering = sum(r() is not None for r in tombstones)
        if lingering:
            raise EvolutionIntegrityError(f"{lingering} purged forecasters still in memory")
        events = self._hall_update(gen) + self._meteors(gen, depth)
        self.gens.append(gen)
        self.train_acc.append(max(i.alpha.score.accuracy for i in self.islands if i.alpha.score))
        self.val_acc.append(self.hall[0]["val_accuracy"] if self.hall else float("nan"))
        self.display.forecast_generation(self, gen, last, rows, events, len(tombstones))

    def _spawn(self, isl: Island, gen: int, depth: int) -> list[Forecaster]:
        a = isl.alpha
        seen = {gp.key(a.tree)}
        kids = []

        def add(tree, role):
            for _ in range(4):
                if gp.key(tree) not in seen:
                    break
                tree = gp.mutate(tree, self.rng, max(isl.rate, 0.15), depth)
            seen.add(gp.key(tree))
            kids.append(self._new(tree, isl.idx, role, gen))

        for _ in range(self.bots - 3):
            add(gp.mutate(a.tree, self.rng, isl.rate, depth), "MUTANT")
        partners = [gp.from_json(h["tree"]) for h in self.hall] or [i.alpha.tree for i in self.islands if i is not isl]
        if partners:
            add(gp.crossover(a.tree, self.rng.choice(partners), self.rng, depth), "CROSSOVER")
        else:
            add(gp.mutate(a.tree, self.rng, isl.rate * 2, depth), "MUTANT")
        roll = self.rng.random()
        if gen % config.LAB_MIGRATION_EVERY == 0 and len(self.islands) > 1:
            add(self.islands[(isl.idx + 1) % len(self.islands)].alpha.tree, "MIGRANT")
        elif self.hall and roll < 0.3:
            add(gp.from_json(self.rng.choice(self.hall)["tree"]), "VETERAN")
        elif roll < 0.5:
            add(gp.random_tree(self.rng, depth), "RANDOM")
        else:
            add(gp.mutate(a.tree, self.rng, min(0.6, isl.rate * 3), depth), "EXPLORER")
        return kids

    def _hall_update(self, gen: int) -> list[str]:
        events = []
        known = {h["key"] for h in self.hall}
        for isl in self.islands:
            k = gp.key(isl.alpha.tree)
            if k in known:
                continue
            v = self.val_score(isl.alpha.tree)
            entry = {"key": k, "bot_id": isl.alpha.bot_id, "island": isl.idx, "generation": gen,
                     "tree": gp.to_json(isl.alpha.tree), "val_accuracy": v.accuracy, "val_edge": v.edge,
                     "train_accuracy": isl.alpha.score.accuracy, "train_edge": isl.alpha.score.edge}
            self.hall.append(entry)
            known.add(k)
            self.hall.sort(key=lambda h: (h["val_edge"], h["train_edge"]), reverse=True)
            if len(self.hall) > config.LAB_HALL_OF_FAME:
                if self.hall.pop() is entry:
                    continue
            events.append(f"🏛  Bot #{entry['bot_id']} enters the Hall of Fame at #{self.hall.index(entry) + 1} "
                          f"(validation accuracy {v.accuracy:.2%}, edge {v.edge * 100:+.2f} pts)")
        return events

    def _meteors(self, gen: int, depth: int) -> list[str]:
        events, seen = [], {}
        for isl in self.islands:
            k = gp.key(isl.alpha.tree)
            why = (f"evolved the same forecaster as Island {seen[k] + 1}" if k in seen else
                   f"Bot #{isl.alpha.bot_id} reigned {isl.reign} generations" if isl.reign >= STAGNATION_LIMIT else None)
            seen.setdefault(k, isl.idx)
            if why:
                isl.alpha = self._new(gp.random_tree(self.rng, depth), isl.idx, "ALPHA", gen + 1)
                isl.alpha.score = self.scorer.score(isl.alpha.tree, self.eras)
                isl.reign, isl.rate = 0, config.MUTATION_RATE
                events.append(f"☄  METEOR STRIKE on Island {isl.idx + 1}: {why}. New dynasty: Bot #{isl.alpha.bot_id}.")
        return events

    # ------------------------------------------------------------------ #
    def _finalize(self) -> ForecastSummary:
        t0, t1 = self.test
        sc = self.scorer
        lab, ok = sc.label[:, t0:t1], sc.valid[:, t0:t1]
        results: list[ForecastResult] = []
        equity: dict[str, np.ndarray] = {}
        move_ratio = None
        yes, no, _question = TARGETS[self.target]
        train_majority = bool(sc.label[:, self.train[0]:self.train[1]][sc.valid[:, self.train[0]:self.train[1]]].mean() >= 0.5)

        def add(name: str, pred: np.ndarray, luck: bool = True):
            nonlocal move_ratio
            acc, base, n, tr = sc.window_stats(pred, t0, t1)
            extra = ""
            if self.target == "direction":
                eq = self._traded_equity(pred, t0, t1)
                equity[name] = eq
                extra = f"if traded: {eq[-1] / eq[0] - 1:+.1%}"
            else:
                moves = np.abs(sc.ans.next_ret[:, t0:t1])
                big_m = float(np.nanmean(moves[ok & pred])) if (ok & pred).any() else float("nan")
                calm_m = float(np.nanmean(moves[ok & ~pred])) if (ok & ~pred).any() else float("nan")
                extra = f"avg move {big_m:.2%} on {yes} calls vs {calm_m:.2%} on {no}" if np.isfinite(big_m + calm_m) else ""
                if name.startswith("Champion"):
                    move_ratio = (big_m, calm_m)
            p = sc.luck_test(pred, t0, t1, runs=500) if luck else None
            results.append(ForecastResult(name, acc, base, acc - base, p, tr, extra))

        champ = self.hall[0] if self.hall else None
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            if champ:
                add(f"Champion (Bot #{champ['bot_id']})", sc.predict(gp.from_json(champ["tree"]), t0, t1))
                members = self.hall[:config.LAB_ENSEMBLE_SIZE]
                if len(members) > 1:
                    votes = sum(sc.predict(gp.from_json(m["tree"]), t0, t1).astype(int) for m in members)
                    add(f"Ensemble vote ({len(members)} bots)", votes * 2 > len(members))
            add("Textbook rule", sc.predict(textbook_tree(self.target), t0, t1))
        add(f"Naive: always say {yes if train_majority else no}", np.full(lab.shape, train_majority), luck=False)
        rng = np.random.default_rng(0)
        add("Coin flip", rng.random(lab.shape) < 0.5, luck=False)
        if self.target == "direction":
            equity["Buy & hold"] = self._traded_equity(np.ones(lab.shape, dtype=bool), t0, t1, cost=0.0)

        summary = ForecastSummary(
            self.target, self.gens, self.train_acc, self.val_acc,
            baseline_train=self._base(self.train), baseline_val=self._base(self.val),
            hall=self.hall, champion=champ, results=results,
            windows={"train": self.dates(self.train), "validation": self.dates(self.val), "test": self.dates(self.test)},
            trials=self.trials, equity=equity, move_ratio=move_ratio)
        if self.save:
            summary.checkpoint = self._save(summary)
            from utils.charts import render_forecast_report
            out = render_forecast_report(summary, config.LAB_DIR / f"forecast_{self.target}.png")
            summary.chart = str(out.relative_to(config.BASE_DIR)).replace("\\", "/") if out else None
        return summary

    def _base(self, w) -> float:
        lab, ok = self.scorer.label[:, w[0]:w[1]], self.scorer.valid[:, w[0]:w[1]]
        p = float(lab[ok].mean())
        return max(p, 1 - p)

    def _traded_equity(self, pred: np.ndarray, t0: int, t1: int, cost: float = 0.0005) -> np.ndarray:
        """Each day hold (equally weighted) every stock predicted UP; cash otherwise. 5 bps per change."""
        nr = np.nan_to_num(self.scorer.ans.next_ret[:, t0:t1])
        held = pred & np.isfinite(self.scorer.ans.next_ret[:, t0:t1])
        n = held.sum(axis=0)
        gross = np.where(n > 0, (nr * held).sum(axis=0) / np.maximum(n, 1), 0.0)
        w = held / np.maximum(n, 1)
        turnover = np.abs(np.diff(np.concatenate([np.zeros((w.shape[0], 1)), w], axis=1), axis=1)).sum(axis=0)
        return config.INITIAL_BALANCE * np.cumprod(np.r_[1.0, 1 + gross - cost * turnover])

    def _save(self, summary: ForecastSummary) -> str:
        path = config.LAB_DIR / f"forecast_{self.target}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"target": self.target, "saved_at": datetime.now().isoformat(timespec="seconds"),
                   "windows": summary.windows, "hall": self.hall, "trials": self.trials,
                   "features": DAILY_NAMES, "quantiles": self.fs.quantiles.tolist(),
                   "results": [r.__dict__ for r in summary.results]}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
        os.replace(tmp, path)
        return str(path.relative_to(config.BASE_DIR)).replace("\\", "/")


def describe_tree(obj, fs: FeatureSet) -> str:
    with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
        return gp.describe(gp.from_json(obj), fs)

