"""
The bots' senses: ~33 features per stock per 5-minute bar.

Every feature at bar t only uses information available at the CLOSE of bar t
(no look-ahead). Features are computed ONCE and shared by every bot, so a
rule like "rsi_14 < 30 AND vwap_dev < -0.2%" costs two array comparisons.

Rule thresholds are stored as QUANTILES (0..1) of each feature, measured on
the training period only - e.g. q=0.1 on rsi_14 means "the 10% lowest RSI
readings seen in training". That keeps mutation scale-free across features.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lab.data import FOMC_DAYS, LabData

FEATURES: list[tuple[str, str]] = [
    ("ret_1", "5-min return"), ("ret_3", "15-min return"), ("ret_6", "30-min return"), ("ret_12", "1-hour return"),
    ("rsi_7", "RSI 7"), ("rsi_14", "RSI 14"), ("rsi_28", "RSI 28"),
    ("sma_dev_10", "% from 10-bar avg"), ("sma_dev_20", "% from 20-bar avg"),
    ("sma_dev_50", "% from 50-bar avg"), ("sma_dev_200", "% from 200-bar avg"),
    ("macd_fast", "MACD 6/13 histogram"), ("macd_std", "MACD 12/26 histogram"),
    ("vwap_dev", "% from today's VWAP"), ("atr_pct", "volatility (ATR %)"), ("vol_z", "volume spike (z)"),
    ("tod", "minutes since open"),
    ("h1_trend", "1-hour-scale trend"), ("d1_trend", "daily trend (vs 20-day avg)"),
    ("gap", "opening gap"), ("day_ret", "return since today's open"),
    ("spy_ret_12", "SPY 1-hour return"), ("spy_day_ret", "SPY return today"), ("spy_d1_trend", "SPY daily trend"),
    ("qqq_ret_12", "QQQ 1-hour return"),
    ("vix", "VIX fear index"), ("vix_chg_5d", "VIX 5-day change"),
    ("rs_rank_1d", "strength rank today (0-1)"), ("rs_rank_5d", "strength rank 5 days (0-1)"),
    ("earnings_near", "earnings within 1 day"), ("fomc_day", "Fed decision day"),
    ("news_sent", "news sentiment 24h"), ("news_count", "news volume 24h"),
]
FEATURE_NAMES: list[str] = [n for n, _ in FEATURES]
FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
BINARY_FEATURES = frozenset({"earnings_near", "fomc_day"})
N_QUANTILES = 101


@dataclass
class FeatureSet:
    names: list[str]
    values: np.ndarray        # [n_features, n_symbols, n_bars] float32 (NaN during warm-up)
    atr: np.ndarray           # [n_symbols, n_bars] ATR in price units (for stops / sizing)
    event_block: np.ndarray   # [n_symbols, n_bars] bool: earnings or Fed day
    spy_day_ret: np.ndarray   # [n_days] SPY close-to-close daily return (regime labels)
    quantiles: np.ndarray | None = None   # [n_features, N_QUANTILES] from the training period

    def fit_quantiles(self, t0: int, t1: int, step: int = 3) -> np.ndarray:
        """Quantile tables from TRAINING bars only (no peeking at validation/test)."""
        q = np.linspace(0, 1, N_QUANTILES)
        tables = np.zeros((len(self.names), N_QUANTILES), dtype=np.float64)
        for f, name in enumerate(self.names):
            if name in BINARY_FEATURES:
                tables[f] = 0.5
                continue
            sample = self.values[f, :, t0:t1:step].ravel()
            sample = sample[np.isfinite(sample)]
            tables[f] = np.quantile(sample, q) if sample.size else 0.0
        self.quantiles = tables
        return tables

    def threshold(self, feature: int, q: float) -> float:
        table = self.quantiles[feature]
        pos = min(max(q, 0.0), 1.0) * (N_QUANTILES - 1)
        lo = int(pos)
        hi = min(lo + 1, N_QUANTILES - 1)
        return float(table[lo] + (table[hi] - table[lo]) * (pos - lo))


# --------------------------------------------------------------------------- #
def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).where(dn != 0, 100.0).where(up.notna())


def _macd_hist(close: pd.Series, fast: int, slow: int, sig: int) -> pd.Series:
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return (line - line.ewm(span=sig, adjust=False).mean()) / close


def _daily_trend(close: pd.Series, day_id: np.ndarray, n: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """(close / SMA of the previous n daily closes - 1, today's open gap, return since today's open)."""
    days = pd.Series(close.to_numpy(), index=day_id)
    day_close = days.groupby(level=0).last()
    prev_sma = day_close.rolling(n, min_periods=5).mean().shift(1)
    prev_close = day_close.shift(1)
    sma_b = pd.Series(prev_sma.reindex(day_id).to_numpy(), index=close.index)
    prev_b = pd.Series(prev_close.reindex(day_id).to_numpy(), index=close.index)
    return close / sma_b - 1, prev_b, sma_b


def build_features(data: LabData) -> FeatureSet:
    S, T = data.n_symbols, data.n_bars
    F = len(FEATURE_NAMES)
    out = np.full((F, S, T), np.nan, dtype=np.float32)
    atr_out = np.zeros((S, T))
    idx = data.index
    day_id = data.day_id
    first_bar = np.r_[True, day_id[1:] != day_id[:-1]]

    def put(name: str, s: int, values) -> None:
        out[FEATURE_INDEX[name], s] = np.asarray(values, dtype=np.float32)

    # ---- context (same for every symbol) --------------------------------
    spy = data.context["SPY"]["close"]
    qqq = data.context["QQQ"]["close"]
    spy_open_day = pd.Series(np.where(first_bar, data.context["SPY"]["open"].to_numpy(), np.nan), index=idx).ffill()
    spy_d1, spy_prev, _ = _daily_trend(spy, day_id)
    ctx = {
        "spy_ret_12": np.log(spy / spy.shift(12)).to_numpy(),
        "spy_day_ret": (spy / spy_open_day - 1).to_numpy(),
        "spy_d1_trend": spy_d1.to_numpy(),
        "qqq_ret_12": np.log(qqq / qqq.shift(12)).to_numpy(),
    }
    vix_daily = data.vix.sort_index()
    vix_daily = vix_daily[~vix_daily.index.duplicated(keep="last")]
    vix_prev = vix_daily.shift(1)  # yesterday's close only - no peeking at today's
    udays = pd.DatetimeIndex(pd.to_datetime(data.unique_days)).normalize()

    def per_bar(series: pd.Series) -> np.ndarray:
        per_day = series.reindex(series.index.union(udays)).ffill().reindex(udays).to_numpy()
        return per_day[day_id]

    vix_b = per_bar(vix_prev)
    vix_5 = per_bar(vix_prev.pct_change(5))
    fomc = np.array([d in FOMC_DAYS for d in data.dates], dtype=np.float32)
    minute = data.minute_of_day.astype(np.float32)

    ret_78 = np.full((S, T), np.nan)
    ret_390 = np.full((S, T), np.nan)
    event_block = np.zeros((S, T), dtype=bool)
    unique_days = np.array(data.unique_days)

    for s, sym in enumerate(data.symbols):
        o = pd.Series(data.open[s], index=idx)
        h = pd.Series(data.high[s], index=idx)
        lo = pd.Series(data.low[s], index=idx)
        c = pd.Series(data.close[s], index=idx)
        v = pd.Series(data.volume[s], index=idx)
        logc = np.log(c)
        for k, name in ((1, "ret_1"), (3, "ret_3"), (6, "ret_6"), (12, "ret_12")):
            put(name, s, logc - logc.shift(k))
        for n, name in ((7, "rsi_7"), (14, "rsi_14"), (28, "rsi_28")):
            put(name, s, _rsi(c, n))
        for n in (10, 20, 50, 200):
            put(f"sma_dev_{n}", s, c / c.rolling(n, min_periods=n).mean() - 1)
        put("macd_fast", s, _macd_hist(c, 6, 13, 5))
        put("macd_std", s, _macd_hist(c, 12, 26, 9))

        typical = (h + lo + c) / 3
        pv = (typical * v).groupby(day_id).cumsum()
        vv = v.groupby(day_id).cumsum()
        vwap = (pv / vv.replace(0, np.nan)).fillna(c)
        put("vwap_dev", s, c / vwap - 1)

        tr = pd.concat([h - lo, (h - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
        atr_out[s] = atr.bfill().to_numpy()
        put("atr_pct", s, atr / c)
        mu, sd = v.rolling(78, min_periods=20).mean(), v.rolling(78, min_periods=20).std()
        put("vol_z", s, (v - mu) / sd.replace(0, np.nan))
        put("tod", s, minute)
        put("h1_trend", s, c / c.rolling(240, min_periods=60).mean() - 1)

        d1, prev_close, _ = _daily_trend(c, day_id)
        day_open = pd.Series(np.where(first_bar, o.to_numpy(), np.nan), index=idx).ffill()
        put("d1_trend", s, d1)
        put("gap", s, day_open / prev_close - 1)
        put("day_ret", s, c / day_open - 1)
        for name, arr in ctx.items():
            put(name, s, arr)
        put("vix", s, vix_b)
        put("vix_chg_5d", s, vix_5)
        ret_78[s] = (logc - logc.shift(78)).to_numpy()
        ret_390[s] = (logc - logc.shift(390)).to_numpy()

        # Events: earnings (the day before, of, and after) and Fed days
        earn = np.zeros(T, dtype=np.float32)
        e_dates = data.earnings.get(sym, [])
        if e_dates:
            e = np.array(sorted(e_dates), dtype="datetime64[D]")
            d = unique_days.astype("datetime64[D]")
            pos = np.searchsorted(e, d)
            nearest = np.full(len(d), np.inf)
            for p_off in (0, -1):
                p = np.clip(pos + p_off, 0, len(e) - 1)
                gap_days = np.abs(np.busday_count(np.minimum(d, e[p]), np.maximum(d, e[p])))
                nearest = np.minimum(nearest, gap_days)
            day_flag = (nearest <= 1).astype(np.float32)
            earn = day_flag[day_id]
        put("earnings_near", s, earn)
        put("fomc_day", s, fomc)
        event_block[s] = (earn > 0) | (fomc > 0)

        # News: sentiment of headlines published in the 24h up to this bar's close
        sent = np.zeros(T)
        cnt = np.zeros(T)
        if data.news is not None and len(data.news):
            ns = data.news[data.news["symbol"] == sym]
            if len(ns):
                times = ns["time"].to_numpy(dtype="datetime64[ns]")
                order = np.argsort(times)
                times, scores = times[order], ns["score"].to_numpy()[order]
                csum = np.r_[0.0, np.cumsum(scores)]
                bar_end = (idx + pd.Timedelta(minutes=5)).to_numpy(dtype="datetime64[ns]")
                hi_i = np.searchsorted(times, bar_end, side="right")
                lo_i = np.searchsorted(times, bar_end - np.timedelta64(24, "h"), side="right")
                sent = csum[hi_i] - csum[lo_i]
                cnt = (hi_i - lo_i).astype(float)
        put("news_sent", s, sent)
        put("news_count", s, cnt)

    for name, arr in (("rs_rank_1d", ret_78), ("rs_rank_5d", ret_390)):
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # warm-up columns are all-NaN
            filled = np.where(np.isfinite(arr), arr, np.nanmedian(arr, axis=0, keepdims=True))
        ranks = filled.argsort(axis=0).argsort(axis=0) / max(S - 1, 1)
        ranks[:, ~np.isfinite(arr).any(axis=0)] = 0.5
        out[FEATURE_INDEX[name]] = ranks.astype(np.float32)

    spy_close_day = pd.Series(spy.to_numpy(), index=day_id).groupby(level=0).last()
    spy_day = spy_close_day.pct_change().fillna(0.0).to_numpy()
    return FeatureSet(FEATURE_NAMES, out, atr_out, event_block, spy_day)
