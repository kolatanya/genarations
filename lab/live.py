"""
Lab live trading on Alpaca PAPER - the ensemble votes every 5 minutes.

Each cycle (every 60 s while the US market is open):
  1. Pull the last ~30 days of 5-minute bars for the universe + SPY/QQQ
     (yfinance, consolidated prices), rebuild every feature with the SAME
     quantile thresholds the bots were evolved with.
  2. Label today's market regime and let the ensemble vote on the last
     COMPLETED bar (same timing as the backtest).
  3. Manage open positions: ATR stop / trailing stop / target, max hold,
     exit votes, and flatten everything at 15:55 ET.
  4. Open new positions on fresh votes (volatility-sized, whole shares,
     longs and shorts).

Safety systems:
  * DAILY LOSS LIMIT - down more than the ensemble's limit today -> flatten, stop for the day
  * DRIFT ALARM      - rolling 5-day live return below the backtest's worst-5% -> benched
                       (no new trades) until you run with --unbench
  * KILL SWITCH      - create the file checkpoints/lab/STOP -> flatten everything and exit
  * SHADOW CHALLENGER- if checkpoints/lab/lab_challenger.json exists (from --mode reevolve),
                       it trades a virtual $ book side by side; after 10 trading days it's
                       recommended for promotion only if it out-earned the live ensemble.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from broker.base import BrokerError
from lab.data import ET, LabData, _align, _load_earnings, headline_sentiment
from lab.ensemble import Ensemble
from lab.features import FEATURE_INDEX, build_features
from lab.gp import TreeEvaluator

log = logging.getLogger(__name__)

STATE_FILE = config.LAB_DIR / "live_state.json"
KILL_FILE = config.LAB_DIR / "STOP"
CHALLENGER_FILE = config.LAB_DIR / "lab_challenger.json"


@dataclass
class Vote:
    symbol: str
    price: float
    long: bool
    short: bool
    exit: bool
    atr: float
    event_day: bool


class VirtualBook:
    """Paper-within-paper: a simulated account used for dry runs and the shadow challenger."""

    def __init__(self, state: dict) -> None:
        self.s = state
        self.s.setdefault("cash", config.INITIAL_BALANCE * 10)   # $100k, like a fresh Alpaca paper account
        self.s.setdefault("positions", {})

    def equity(self, prices: dict[str, float]) -> float:
        eq = self.s["cash"]
        for sym, p in self.s["positions"].items():
            eq += p["qty"] * p["dir"] * (prices.get(sym, p["entry"]) - p["entry"]) + p["qty"] * p["entry"]
        return eq

    def open(self, sym: str, direction: int, qty: int, price: float, meta: dict) -> None:
        cost = price * (1 + direction * config.LAB_BASE_COST_BPS / 1e4)
        self.s["cash"] -= qty * cost
        self.s["positions"][sym] = {"dir": direction, "qty": qty, "entry": cost, **meta}

    def close(self, sym: str, price: float) -> float:
        p = self.s["positions"].pop(sym)
        fill = price * (1 - p["dir"] * config.LAB_BASE_COST_BPS / 1e4)
        pnl = p["dir"] * p["qty"] * (fill - p["entry"])
        self.s["cash"] += p["qty"] * p["entry"] + pnl
        return pnl


def fetch_live_data(symbols: list[str], days: int = 30) -> LabData:
    """Recent 5-minute bars for the universe + context from yfinance, on one session grid."""
    import yfinance as yf

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    tickers = symbols + config.LAB_CONTEXT
    raw = yf.download(tickers, period=f"{days}d", interval="5m", group_by="ticker", auto_adjust=True,
                      prepost=False, progress=False, threads=True)
    if raw is None or raw.empty:
        raise BrokerError("yfinance returned no intraday data")
    frames: dict[str, pd.DataFrame] = {}
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        df = raw[t].dropna(how="all").copy()
        df.columns = [c.lower() for c in df.columns]
        idx = pd.DatetimeIndex(df.index)
        idx = idx.tz_convert(ET) if idx.tz is not None else idx.tz_localize("UTC").tz_convert(ET)
        df.index = idx.tz_localize(None)
        frames[t] = df[["open", "high", "low", "close", "volume"]].astype(float)
    if "SPY" not in frames:
        raise BrokerError("SPY intraday data unavailable")
    grid = frames["SPY"].index
    syms = [s for s in symbols if s in frames]
    arrays = {k: np.stack([_align(frames[s], grid)[k].to_numpy() for s in syms]) for k in
              ("open", "high", "low", "close", "volume")}
    vix = yf.Ticker("^VIX").history(period="3mo", interval="1d")["Close"]
    vix.index = pd.DatetimeIndex(vix.index).tz_localize(None).normalize()
    return LabData(symbols=syms, index=grid, context={c: _align(frames[c], grid) for c in config.LAB_CONTEXT if c in frames},
                   vix=vix, earnings=_load_earnings(syms), news=_recent_news(syms), source="yfinance-live", **arrays)


_NEWS_CACHE: dict[str, object] = {"at": 0.0, "df": None}


def _recent_news(symbols: list[str]) -> pd.DataFrame | None:
    if not config.LAB_NEWS or not config.ALPACA_API_KEY:
        return None
    if time.time() - float(_NEWS_CACHE["at"]) < 900:          # refresh every 15 minutes
        return _NEWS_CACHE["df"]  # type: ignore[return-value]
    try:
        from alpaca.data.historical.news import NewsClient
        from alpaca.data.requests import NewsRequest

        client = NewsClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
        start = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=30)
        rows = []
        for i in range(0, len(symbols), 10):
            batch = symbols[i:i + 10]
            ns = client.get_news(NewsRequest(symbols=",".join(batch), start=start.to_pydatetime(), limit=None))
            for item in ns.data.get("news", []):
                t = pd.Timestamp(item.created_at).tz_convert(ET).tz_localize(None)
                for sym in set(item.symbols or []) & set(batch):
                    rows.append((t, sym, headline_sentiment(f"{item.headline} {item.summary or ''}")))
        df = pd.DataFrame(rows, columns=["time", "symbol", "score"])
        _NEWS_CACHE.update(at=time.time(), df=df)
        return df
    except Exception as exc:
        log.warning("Live news unavailable: %s", exc)
        return _NEWS_CACHE["df"]  # type: ignore[return-value]


class LabLive:
    def __init__(self, deploy: dict, broker, display, *, dry_run: bool, unbench: bool = False) -> None:
        if deploy.get("ensemble") is None:
            raise BrokerError("This Lab checkpoint has no ensemble to deploy - run the Evolution Lab first")
        self.ensemble = Ensemble.from_json(deploy["ensemble"])
        self.quantiles = np.array(deploy["quantiles"])
        self.drift_line = deploy.get("drift_five_day_p5")
        self.broker = broker
        self.display = display
        self.dry_run = dry_run or broker is None
        self.state = self._load_state()
        for k, default in (("positions", {}), ("acted", {}), ("daily", []), ("benched", False)):
            self.state.setdefault(k, default)
        self._last_prices: dict[str, float] = {}
        if unbench:
            self.state["benched"] = False
        self.book = VirtualBook(self.state.setdefault("virtual", {}))
        self.challenger = None
        if CHALLENGER_FILE.is_file():
            ch = json.loads(CHALLENGER_FILE.read_text(encoding="utf-8")).get("deploy", {})
            if ch.get("ensemble"):
                self.challenger = Ensemble.from_json(ch["ensemble"])
                self.challenger_q = np.array(ch["quantiles"])
                self.shadow = VirtualBook(self.state.setdefault("shadow", {}))

    # ------------------------------------------------------------------ #
    def _load_state(self) -> dict:
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"positions": {}, "benched": False, "daily": [], "acted": {}}

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=float), encoding="utf-8")
        os.replace(tmp, STATE_FILE)

    # ------------------------------------------------------------------ #
    def run(self, poll_seconds: int = 60, max_cycles: int | None = None) -> None:
        cycle = 0
        try:
            while max_cycles is None or cycle < max_cycles:
                cycle += 1
                if KILL_FILE.exists():
                    self.display.warn("KILL SWITCH file found - flattening everything and stopping.")
                    self._flatten_all("kill switch")
                    self._save_state()
                    return
                self._cycle(cycle)
                self._save_state()
                if max_cycles is not None and cycle >= max_cycles:
                    break
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            self._save_state()
            self.display.warn("Lab live loop stopped. Open paper positions were left as they are.")

    def _market_open(self) -> bool:
        if self.broker is not None:
            try:
                return self.broker.is_market_open()
            except BrokerError:
                pass
        now = pd.Timestamp.now(tz=ET)
        m = now.hour * 60 + now.minute
        return now.dayofweek < 5 and 570 <= m < 960

    def _cycle(self, cycle: int) -> None:
        now = pd.Timestamp.now(tz=ET).tz_localize(None)
        if not self._market_open():
            self.display.lab_live_idle(cycle, now)
            return
        data = fetch_live_data(list(self.ensemble_symbols()))
        fs = build_features(data)
        fs.quantiles = self.quantiles
        forming = now.floor("5min")
        t = int(np.searchsorted(data.index.to_numpy(), np.datetime64(forming), side="left")) - 1
        if t < 1:
            self.display.warn("Not enough completed bars yet today.")
            return
        bar_time = str(data.index[t])
        minute = int(data.minute_of_day[t])
        trend = fs.values[FEATURE_INDEX["spy_d1_trend"], 0, t]
        regime = 0 if trend > 0.01 else (2 if trend < -0.01 else 1)
        votes = self._votes(self.ensemble, fs, data, t, regime)
        prices = {v.symbol: v.price for v in votes}
        self._last_prices = prices

        equity, day_pnl = self._account(prices)
        self._track_day(now.date().isoformat(), equity)
        rows = []
        clock_minute = now.hour * 60 + now.minute - (9 * 60 + 30)       # real time, not bar time
        flatten_time = clock_minute >= config.LAB_FLATTEN_MINUTE
        loss_hit = day_pnl < -self.ensemble.params["daily_loss"]
        if loss_hit and not self.state.get("loss_day") == now.date().isoformat():
            self.state["loss_day"] = now.date().isoformat()
            self._flatten_all("daily loss limit")
        blocked_today = self.state.get("loss_day") == now.date().isoformat()
        benched = bool(self.state.get("benched"))

        positions = self._positions()
        slot = equity / max(len(votes), 1)
        p = self.ensemble.params
        for v in votes:
            pos = positions.get(v.symbol)
            action, reason, order = "HOLD", "", "-"
            meta = self.state["positions"].get(v.symbol)
            if pos is not None:
                direction = 1 if pos > 0 else -1
                if meta is None:  # position we didn't open (or state lost): adopt it
                    meta = self.state["positions"][v.symbol] = {
                        "dir": direction, "entry": v.price, "stop_d": p["stop_atr"] * v.atr,
                        "tp_d": p["tp_atr"] * v.atr, "trail_d": p["trail_atr"] * v.atr, "extreme": v.price,
                        "opened": bar_time, "bars": 0}
                meta["bars"] = meta.get("bars", 0) + (1 if self.state["acted"].get(v.symbol + ":bar") != bar_time else 0)
                self.state["acted"][v.symbol + ":bar"] = bar_time
                why = self._exit_reason(meta, v, direction, flatten_time, p)
                if why:
                    order = self._close(v.symbol, v.price)
                    action, reason = "CLOSE", why
                else:
                    reason = f"holding {'long' if direction > 0 else 'short'} · stop {meta['entry'] - direction * meta['stop_d']:.2f}"
            else:
                self.state["positions"].pop(v.symbol, None)
                fresh = self.state["acted"].get(v.symbol) != bar_time
                allowed = (not flatten_time and not blocked_today and not benched and fresh
                           and p["start_min"] <= minute <= p["end_min"]
                           and not (self.ensemble.avoid_events and v.event_day))
                side = 1 if v.long else (-1 if v.short else 0)
                if side and allowed and v.atr > 0:
                    stop_d = p["stop_atr"] * v.atr
                    qty = int(min(slot * p["risk_pct"] / stop_d, slot / v.price))
                    if qty >= 1:
                        order = self._open(v.symbol, side, qty, v.price, {
                            "stop_d": stop_d, "tp_d": p["tp_atr"] * v.atr, "trail_d": p["trail_atr"] * v.atr,
                            "extreme": v.price, "opened": bar_time, "bars": 0})
                        action = "BUY" if side > 0 else "SHORT"
                        reason = f"ensemble vote ({'long' if side > 0 else 'short'}) · {qty} shares"
                        self.state["acted"][v.symbol] = bar_time
                    else:
                        reason = "vote, but position would be under 1 share"
                elif side and not allowed:
                    reason = ("benched by drift alarm" if benched else "daily loss limit hit" if blocked_today else
                              "closing time" if flatten_time else "event day" if v.event_day else "outside trading hours"
                              if not p["start_min"] <= minute <= p["end_min"] else "already acted on this bar")
                else:
                    reason = "no vote"
            rows.append((v.symbol, v.price, pos or 0, action, reason, order))

        shadow_line = self._shadow_step(data, fs, t, regime, bar_time, minute, flatten_time) if self.challenger else None
        self.display.lab_live_cycle(cycle, bar_time, ["UP-TREND", "SIDEWAYS", "DOWN-TREND"][regime], equity, day_pnl,
                                    rows, benched, blocked_today, shadow_line, self.dry_run)

    # ------------------------------------------------------------------ #
    def ensemble_symbols(self) -> list[str]:
        return list(config.LAB_UNIVERSE)

    def _votes(self, ens: Ensemble, fs, data: LabData, t: int, regime: int, quantiles=None) -> list[Vote]:
        if quantiles is not None:
            fs.quantiles = quantiles
        ev = TreeEvaluator(fs, t, t + 1)
        sigs = [(ev(m.genome.long_entry), ev(m.genome.short_entry), ev(m.genome.exit_rule)) for m in ens.members]
        long_s, short_s, exit_s = ens.vote(sigs, np.array([regime]))
        # Votes come from the last COMPLETED bar; prices for stops/sizing are the freshest available
        # (the still-forming bar's latest close).
        return [Vote(sym, float(data.close[i, -1]), bool(long_s[i, 0]), bool(short_s[i, 0]), bool(exit_s[i, 0]),
                     float(fs.atr[i, t]), bool(fs.event_block[i, t])) for i, sym in enumerate(data.symbols)]

    @staticmethod
    def _exit_reason(meta: dict, v: Vote, direction: int, flatten_time: bool, p: dict) -> str | None:
        price = v.price
        if direction > 0:
            meta["extreme"] = max(meta["extreme"], price)
            stop = max(meta["entry"] - meta["stop_d"], meta["extreme"] - meta["trail_d"])
            if price <= stop:
                return "stop / trailing stop"
            if price >= meta["entry"] + meta["tp_d"]:
                return "take profit"
            if v.exit or v.short:
                return "ensemble exit vote"
        else:
            meta["extreme"] = min(meta["extreme"], price)
            stop = min(meta["entry"] + meta["stop_d"], meta["extreme"] + meta["trail_d"])
            if price >= stop:
                return "stop / trailing stop"
            if price <= meta["entry"] - meta["tp_d"]:
                return "take profit"
            if v.exit or v.long:
                return "ensemble exit vote"
        if meta.get("bars", 0) >= p["max_hold"]:
            return f"max hold ({int(p['max_hold']) * 5} min)"
        if flatten_time:
            return "flatten before the close"
        return None

    # ------------------------------------------------------------------ #
    # Broker / virtual book plumbing
    # ------------------------------------------------------------------ #
    def _positions(self) -> dict[str, float]:
        if self.dry_run:
            return {s: p["qty"] * p["dir"] for s, p in self.book.s["positions"].items()}
        return {p.symbol: p.qty for p in self.broker.get_positions().values()}

    def _account(self, prices: dict[str, float]) -> tuple[float, float]:
        if self.dry_run:
            eq = self.book.equity(prices)
            start = self.state.get("day_start_equity", eq)
            return eq, eq / start - 1 if start else 0.0
        acct = self.broker.get_account_summary()
        last = getattr(acct, "last_equity", None) or self.state.get("day_start_equity", acct.equity)
        return acct.equity, acct.equity / last - 1 if last else 0.0

    def _open(self, sym: str, side: int, qty: int, price: float, meta: dict) -> str:
        self.state["positions"][sym] = {"dir": side, "entry": price, **meta}
        if self.dry_run:
            self.book.open(sym, side, qty, price, meta)
            return f"DRY RUN: {'buy' if side > 0 else 'short'} {qty}"
        r = self.broker.submit_market_order(sym, "buy" if side > 0 else "sell", qty=qty)
        return f"sent {r.order_id[:8]} ({r.status})"

    def _close(self, sym: str, price: float | None = None) -> str:
        self.state["positions"].pop(sym, None)
        if self.dry_run:
            if sym in self.book.s["positions"]:
                px = price if price is not None else self._last_prices.get(sym, self.book.s["positions"][sym]["entry"])
                pnl = self.book.close(sym, px)
                return f"DRY RUN: closed (${pnl:+,.0f})"
            return "DRY RUN: closed"
        r = self.broker.close_position(sym)
        return f"close sent {r.order_id[:8]}"

    def _flatten_all(self, why: str) -> None:
        for sym in list(self._positions()):
            self._close(sym)
        log.warning("Flattened all positions: %s", why)

    def _track_day(self, day: str, equity: float) -> None:
        if self.state.get("day") != day:
            if self.state.get("day"):
                self.state.setdefault("daily", []).append(self.state.get("last_equity", equity))
            self.state["day"] = day
            self.state["day_start_equity"] = equity
        self.state["last_equity"] = equity
        daily = self.state.get("daily", [])
        n = config.LAB_DRIFT_WINDOW_DAYS
        if self.drift_line is not None and len(daily) > n:
            window_ret = daily[-1] / daily[-1 - n] - 1
            if window_ret < self.drift_line and not self.state.get("benched"):
                self.state["benched"] = True
                self.display.warn(f"DRIFT ALARM: last {n} days {window_ret:+.2%} is worse than the backtest's "
                                  f"worst-5% ({self.drift_line:+.2%}). Benched - no new trades until --unbench.")

    def _shadow_step(self, data, fs, t, regime, bar_time, minute, flatten_time) -> str:
        """Challenger trades a virtual book on the same bars; recommend promotion after enough days."""
        votes = self._votes(self.challenger, fs, data, t, regime, quantiles=self.challenger_q)
        fs.quantiles = self.quantiles
        prices = {v.symbol: v.price for v in votes}
        book = self.shadow
        p = self.challenger.params
        slot = book.equity(prices) / max(len(votes), 1)
        for v in votes:
            pos = book.s["positions"].get(v.symbol)
            if pos:
                pos["bars"] = pos.get("bars", 0) + 1
                if self._exit_reason(pos, v, pos["dir"], flatten_time, p):
                    book.close(v.symbol, v.price)
            elif (v.long or v.short) and not flatten_time and p["start_min"] <= minute <= p["end_min"] and v.atr > 0:
                side = 1 if v.long else -1
                stop_d = p["stop_atr"] * v.atr
                qty = int(min(slot * p["risk_pct"] / stop_d, slot / v.price))
                if qty >= 1:
                    book.open(v.symbol, side, qty, v.price, {"stop_d": stop_d, "tp_d": p["tp_atr"] * v.atr,
                                                            "trail_d": p["trail_atr"] * v.atr, "extreme": v.price,
                                                            "bars": 0})
        sh = self.state.setdefault("shadow_track", {"start_live": None, "start_shadow": None, "days": []})
        eq_shadow = book.equity(prices)
        eq_live = self.state.get("last_equity", 0.0)
        if sh["start_live"] is None:
            sh.update(start_live=eq_live, start_shadow=eq_shadow)
        day = str(data.index[t].date())
        if day not in sh["days"]:
            sh["days"].append(day)
        live_ret = eq_live / sh["start_live"] - 1 if sh["start_live"] else 0.0
        shadow_ret = eq_shadow / sh["start_shadow"] - 1 if sh["start_shadow"] else 0.0
        verdict = ""
        if len(sh["days"]) >= config.LAB_SHADOW_PROMOTE_DAYS:
            verdict = (" → PROMOTE: run  main.py --mode lab-promote" if shadow_ret > live_ret
                       else " → keep the current ensemble")
        return (f"Shadow challenger: {shadow_ret:+.2%} vs live {live_ret:+.2%} over {len(sh['days'])} "
                f"trading day(s){verdict}")


def promote_challenger() -> str:
    """Swap the shadow challenger in as the live ensemble (the old one is kept as a backup)."""
    latest = config.LAB_DIR / "lab_latest.json"
    if not CHALLENGER_FILE.is_file():
        raise BrokerError("No shadow challenger to promote (run --mode reevolve first)")
    backup = config.LAB_DIR / f"lab_latest_backup_{datetime.now():%Y%m%d_%H%M%S}.json"
    if latest.is_file():
        latest.replace(backup)
    CHALLENGER_FILE.replace(latest)
    try:
        state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        state.pop("shadow", None)
        state.pop("shadow_track", None)
        STATE_FILE.write_text(json.dumps(state, indent=1), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return backup.name


def load_deploy(path: Path = config.LAB_DIR / "lab_latest.json") -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BrokerError(f"No Lab checkpoint at {path} - run the Evolution Lab first ({exc})") from exc
    if "deploy" not in data:
        raise BrokerError(f"{path} has no deployable ensemble (was the run interrupted before the end?)")
    return data["deploy"]
