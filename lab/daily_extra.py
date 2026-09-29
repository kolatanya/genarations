"""
Extra senses for the daily Forecaster / Prophecy League:

  * the S&P 500 universe (from Wikipedia, cached monthly; falls back to 100 stocks)
  * earnings calendars (yfinance, cached per stock) mapped to the exact daily
    move they affect: an after-close report moves the NEXT day, a before-open
    report moves that SAME day
  * market implied-volatility indices: VIX 9-day / 30-day / 3-month (the fear
    "term structure"), Nasdaq VXN and VVIX. Per-stock options history is not
    available for free, so these market-wide gauges are the honest substitute.

Note: using TODAY's S&P 500 members for past years adds survivorship bias
(companies that shrank or failed are missing). Accuracy is always judged
against naive baselines on the same stocks, which limits the damage.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request

import numpy as np
import pandas as pd

import config

log = logging.getLogger(__name__)

IV_TICKERS = {"vix9d": "^VIX9D", "vix": "^VIX", "vix3m": "^VIX3M", "vxn": "^VXN", "vvix": "^VVIX"}
EARNINGS_DIR = config.DATA_CACHE_DIR / "earnings"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def sp500_universe(fallback: list[str]) -> list[str]:
    cache = config.DATA_CACHE_DIR / "sp500.json"
    try:
        if cache.is_file() and time.time() - cache.stat().st_mtime < 30 * 86400:
            syms = json.loads(cache.read_text())
            if len(syms) > 400:
                return syms
        req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (evolution-arena research)"})
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
        table = html.split('id="constituents"', 1)[1].split("</table>", 1)[0]
        syms = []
        for row in table.split("<tr")[2:]:
            cell = row.split("<td", 2)[1] if "<td" in row else ""
            m = re.search(r">([A-Z][A-Z.\-]{0,6})</a>", cell)
            if m:
                syms.append(m.group(1).replace(".", "-"))
        syms = list(dict.fromkeys(syms))
        if len(syms) < 400:
            raise ValueError(f"only parsed {len(syms)} tickers")
        config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(syms))
        return syms
    except Exception as exc:
        log.warning("S&P 500 list unavailable (%s) - using the 100-stock universe", exc)
        return list(fallback)


def sp500_sectors() -> dict[str, str]:
    """Ticker -> GICS sector (11 sectors), from the same Wikipedia table; cached for 30 days. {} if unavailable."""
    cache = config.DATA_CACHE_DIR / "sp500_sectors.json"
    try:
        if cache.is_file() and time.time() - cache.stat().st_mtime < 30 * 86400:
            return json.loads(cache.read_text())
        req = urllib.request.Request(WIKI_URL, headers={"User-Agent": "Mozilla/5.0 (evolution-arena research)"})
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
        table = html.split('id="constituents"', 1)[1].split("</table>", 1)[0]
        out = {}
        for row in table.split("<tr")[2:]:
            cells = [re.sub(r"<[^>]+>", "", c.split(">", 1)[1]).strip() for c in row.split("<td")[1:4]]
            if len(cells) == 3 and cells[0]:
                out[cells[0].replace(".", "-")] = cells[2]
        if len(out) < 400:
            raise ValueError(f"only parsed {len(out)} sectors")
        config.DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out))
        return out
    except Exception as exc:
        log.warning("S&P 500 sectors unavailable (%s) - sector senses disabled", exc)
        return {}


def earnings_reaction(earn_next: np.ndarray, next_ret: np.ndarray, normal: np.ndarray, last_n: int = 8) -> np.ndarray:
    """
    How big this stock's earnings reactions USUALLY are, in multiples of its normal daily move:
    the average of |move| / normal over its last `last_n` earnings reactions.

    A reaction flagged on day j is the move from close j to close j+1, so it only becomes
    known at day j+1 - the value at day t never uses a reaction that hadn't finished by t.
    """
    T = len(earn_next)
    known = np.full(T, np.nan)
    for j in np.flatnonzero(earn_next > 0):
        if j + 1 < T and np.isfinite(next_ret[j]) and np.isfinite(normal[j]) and normal[j] > 0:
            known[j + 1] = abs(next_ret[j]) / normal[j]
    s = pd.Series(known)
    events = s.dropna()
    if events.empty:
        return np.full(T, np.nan)
    avg = events.rolling(last_n, min_periods=1).mean()
    return avg.reindex(range(T)).ffill().to_numpy()


def load_earnings(symbols: list[str], say=lambda m: None, max_age_days: int = 7) -> dict[str, list[pd.Timestamp]]:
    """Earnings announcement timestamps (America/New_York), past and scheduled, per stock."""
    EARNINGS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, list[pd.Timestamp]] = {}
    stale = []
    for s in symbols:
        f = EARNINGS_DIR / f"{s}.json"
        if f.is_file() and time.time() - f.stat().st_mtime < max_age_days * 86400:
            try:
                out[s] = [pd.Timestamp(x) for x in json.loads(f.read_text())]
                continue
            except ValueError:
                pass
        stale.append(s)
    if stale:
        import yfinance as yf

        logging.getLogger("yfinance").setLevel(logging.CRITICAL)
        for i, s in enumerate(stale, 1):
            if i % 10 == 1:
                say(f"Downloading earnings calendars ({i}/{len(stale)}, one-time, cached for a week)...")
            stamps: list[pd.Timestamp] = []
            for attempt in range(3):
                try:
                    df = yf.Ticker(s).get_earnings_dates(limit=72)
                    if df is not None and len(df):
                        idx = pd.DatetimeIndex(df.index)
                        idx = idx.tz_convert("America/New_York") if idx.tz is not None else idx.tz_localize("America/New_York")
                        stamps = sorted(set(idx))
                    break
                except Exception as exc:  # rate limits: back off and retry
                    if "Too Many" in str(exc) or "429" in str(exc):
                        time.sleep(5 * (attempt + 1))
                    else:
                        break
            (EARNINGS_DIR / f"{s}.json").write_text(json.dumps([t.isoformat() for t in stamps]))
            out[s] = stamps
            time.sleep(0.15)
    return out


def earnings_features(stamps: list[pd.Timestamp], index: pd.DatetimeIndex, cap: int = 10) -> tuple[np.ndarray, ...]:
    """
    (earn_next, days_to_earn, days_since_earn) per trading day t, where a
    prediction made at t's close is about the move from close t to close t+1.

    earn_next[t] = 1 when that move includes an earnings reaction:
      report after the close on day t, or before/during the session on day t+1.
    Report dates are published weeks ahead, so looking up to `cap` trading
    days forward is information a real trader has.
    """
    T = len(index)
    earn_next = np.zeros(T, dtype=np.float32)
    if not stamps:
        return earn_next, np.full(T, np.nan, np.float32), np.full(T, np.nan, np.float32)
    days = index.normalize()
    pos = {d: i for i, d in enumerate(days)}
    for e in stamps:
        d = pd.Timestamp(e.date())
        minutes = e.hour * 60 + e.minute
        i = pos.get(d)
        if i is None:  # announcement on a non-trading day -> affects the next session
            j = int(np.searchsorted(days.values, np.datetime64(d)))
            if 0 < j <= T:
                earn_next[j - 1] = 1
            continue
        if minutes >= 16 * 60:
            earn_next[i] = 1                      # after close -> tomorrow's move
        elif minutes == 0:
            earn_next[i] = 1                      # time unknown -> flag both possible moves
            if i > 0:
                earn_next[i - 1] = 1
        elif i > 0:
            earn_next[i - 1] = 1                  # before/during the session -> today's move
    flagged = np.flatnonzero(earn_next > 0)
    to = np.full(T, np.nan, np.float32)
    since = np.full(T, np.nan, np.float32)
    if len(flagged):
        nxt = np.searchsorted(flagged, np.arange(T), side="left")
        ok = nxt < len(flagged)
        gap = np.where(ok, flagged[np.minimum(nxt, len(flagged) - 1)] - np.arange(T), np.inf)
        to = np.where(gap <= cap, gap, cap).astype(np.float32)
        prv = np.searchsorted(flagged, np.arange(T), side="right") - 1
        back = np.where(prv >= 0, np.arange(T) - flagged[np.maximum(prv, 0)], np.inf)
        since = np.where(back <= cap, back, cap).astype(np.float32)
    return earn_next, to, since


def iv_context(raw: pd.DataFrame, grid: pd.DatetimeIndex) -> dict[str, pd.Series]:
    """Implied-volatility indices aligned to the trading-day grid (previous values carried forward)."""
    have = set(raw.columns.get_level_values(0))
    out = {}
    for key, ticker in IV_TICKERS.items():
        if ticker in have:
            s = raw[ticker]["Close"].copy()
            s.index = pd.DatetimeIndex(s.index).tz_localize(None) if getattr(s.index, "tz", None) else s.index
            out[key] = s.reindex(grid).ffill()
        else:
            out[key] = pd.Series(np.nan, index=grid)
    return out

