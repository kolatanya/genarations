"""
Fast trade simulator (compiled to machine code with numba).

Every stock gets an equal slice of capital and trades independently:
  * signals are read at a bar's CLOSE and filled at the NEXT bar's open
  * long AND short positions
  * stop-loss / take-profit / trailing stop sized in ATRs (volatility-aware)
  * position size = risk_pct of equity / stop distance (volatility sizing), max 1x
  * stale trades closed after max_hold bars
  * every position is closed on the day's last bar (no overnight gap risk)
  * daily loss limit: after losing daily_loss in a day, that stock stops for the day
  * optional: no new entries on earnings / Fed days
  * every fill pays a cost of base + spread_factor x (bar range), capped
Pessimistic intrabar order: if a stop and a target are both touched, the stop wins.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - slow but correct fallback
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap(args[0]) if args and callable(args[0]) else wrap

import config
from lab.data import LabData
from lab.features import FeatureSet

EXIT_REASONS = ("SIGNAL", "STOP", "TRAIL", "TARGET", "TIME", "CLOSE", "DAILY_LOSS", "END")


@njit(cache=True)
def _simulate(open_, high, low, close, atr, day_id, last_bar, minute, event_block,
              long_sig, short_sig, exit_sig, t0, t1, capital,
              stop_atr, tp_atr, trail_atr, max_hold, risk_pct, start_min, end_min, daily_loss, avoid_events,
              base_cost, spread_factor, max_cost):
    S = open_.shape[0]
    n = t1 - t0
    equity = np.empty((S, n))
    cap_trades = n // 2 + 2
    tr_sym = np.empty(S * cap_trades, np.int32)
    tr_entry = np.empty(S * cap_trades, np.int64)
    tr_exit = np.empty(S * cap_trades, np.int64)
    tr_dir = np.empty(S * cap_trades, np.int8)
    tr_pnl = np.empty(S * cap_trades)
    tr_ret = np.empty(S * cap_trades)
    tr_reason = np.empty(S * cap_trades, np.int8)
    k = 0

    for s in range(S):
        realized = capital
        pos = 0
        qty = 0.0
        entry = 0.0
        entry_t = 0
        stop_d = 0.0
        tp_d = 0.0
        trail_d = 0.0
        extreme = 0.0
        pending = 0          # 1 = buy, -1 = short, 2 = exit
        cur_day = -1
        day_start = capital
        blocked = False
        last_eq = capital

        for t in range(t0, t1):
            i = t - t0
            o = open_[s, t]
            h = high[s, t]
            lo = low[s, t]
            c = close[s, t]
            if day_id[t] != cur_day:
                cur_day = day_id[t]
                day_start = last_eq
                blocked = False
            prev = t - 1 if t > 0 else t
            cost = base_cost + spread_factor * (high[s, prev] - low[s, prev]) / close[s, prev]
            if cost > max_cost:
                cost = max_cost

            exit_px = -1.0
            reason = 0
            # 1) fill yesterday-bar's order at this bar's open
            if pending == 2 and pos != 0:
                exit_px = o
                reason = 0
            elif (pending == 1 or pending == -1) and pos == 0 and not blocked and atr[s, prev] > 0:
                d = pending
                px = o * (1.0 + d * cost)
                sd = stop_atr * atr[s, prev]
                q = (last_eq * risk_pct) / sd
                if q * px > last_eq:
                    q = last_eq / px
                if q > 0:
                    pos = d
                    qty = q
                    entry = px
                    entry_t = t
                    stop_d = sd
                    tp_d = tp_atr * atr[s, prev]
                    trail_d = trail_atr * atr[s, prev]
                    extreme = px
            pending = 0

            # 2) intrabar risk management
            if pos != 0 and exit_px < 0:
                if pos == 1:
                    hard = entry - stop_d
                    trail = extreme - trail_d
                    stop = hard if hard > trail else trail
                    stop_reason = 1 if hard >= trail else 2
                    target = entry + tp_d
                    if o <= stop:
                        exit_px = o
                        reason = stop_reason
                    elif o >= target:
                        exit_px = o
                        reason = 3
                    elif lo <= stop:
                        exit_px = stop
                        reason = stop_reason
                    elif h >= target:
                        exit_px = target
                        reason = 3
                    elif h > extreme:
                        extreme = h
                else:
                    hard = entry + stop_d
                    trail = extreme + trail_d
                    stop = hard if hard < trail else trail
                    stop_reason = 1 if hard <= trail else 2
                    target = entry - tp_d
                    if o >= stop:
                        exit_px = o
                        reason = stop_reason
                    elif o <= target:
                        exit_px = o
                        reason = 3
                    elif h >= stop:
                        exit_px = stop
                        reason = stop_reason
                    elif lo <= target:
                        exit_px = target
                        reason = 3
                    elif lo < extreme:
                        extreme = lo
                # time-based exits at the close
                if exit_px < 0:
                    if t - entry_t >= max_hold:
                        exit_px = c
                        reason = 4
                    elif last_bar[t] or t == t1 - 1:
                        exit_px = c
                        reason = 5 if last_bar[t] else 7

            if exit_px >= 0 and pos != 0:
                fill = exit_px * (1.0 - pos * cost)
                pnl = pos * qty * (fill - entry)
                realized += pnl
                tr_sym[k] = s
                tr_entry[k] = entry_t
                tr_exit[k] = t
                tr_dir[k] = pos
                tr_pnl[k] = pnl
                tr_ret[k] = pnl / (qty * entry)
                tr_reason[k] = reason
                k += 1
                pos = 0

            eq = realized + (pos * qty * (c - entry) if pos != 0 else 0.0)
            # 3) daily loss limit
            if not blocked and eq < day_start * (1.0 - daily_loss):
                blocked = True
                if pos != 0:
                    fill = c * (1.0 - pos * cost)
                    pnl = pos * qty * (fill - entry)
                    realized += pnl
                    tr_sym[k] = s
                    tr_entry[k] = entry_t
                    tr_exit[k] = t
                    tr_dir[k] = pos
                    tr_pnl[k] = pnl
                    tr_ret[k] = pnl / (qty * entry)
                    tr_reason[k] = 6
                    k += 1
                    pos = 0
                    eq = realized
            equity[s, i] = eq
            last_eq = eq

            # 4) read signals at the close -> order for the next bar
            if not last_bar[t] and t < t1 - 1:
                j = t - t0
                if pos != 0:
                    if exit_sig[s, j] or (pos == 1 and short_sig[s, j]) or (pos == -1 and long_sig[s, j]):
                        pending = 2
                elif not blocked and start_min <= minute[t] <= end_min and not (avoid_events and event_block[s, t]):
                    if long_sig[s, j]:
                        pending = 1
                    elif short_sig[s, j]:
                        pending = -1

    return (equity, tr_sym[:k], tr_entry[:k], tr_exit[:k], tr_dir[:k], tr_pnl[:k], tr_ret[:k], tr_reason[:k])


@dataclass
class SimResult:
    t0: int
    t1: int
    equity: np.ndarray        # portfolio equity per bar
    daily_equity: np.ndarray  # equity at each day's close
    day_index: np.ndarray     # which trading day each daily_equity point is
    trades_sym: np.ndarray
    trades_entry: np.ndarray
    trades_exit: np.ndarray
    trades_dir: np.ndarray
    trades_pnl: np.ndarray
    trades_ret: np.ndarray
    trades_reason: np.ndarray
    capital: float

    @property
    def n_trades(self) -> int:
        return len(self.trades_pnl)

    @property
    def total_return(self) -> float:
        return float(self.equity[-1] / self.capital - 1) if len(self.equity) else 0.0

    @property
    def daily_returns(self) -> np.ndarray:
        eq = np.r_[self.capital, self.daily_equity]
        return eq[1:] / eq[:-1] - 1

    @property
    def win_rate(self) -> float:
        return float((self.trades_pnl > 0).mean()) if self.n_trades else 0.0

    @property
    def profit_factor(self) -> float:
        wins = self.trades_pnl[self.trades_pnl > 0].sum()
        losses = -self.trades_pnl[self.trades_pnl < 0].sum()
        if losses <= 0:
            return 10.0 if wins > 0 else 0.0
        return float(min(wins / losses, 10.0))

    @property
    def max_drawdown(self) -> float:
        if not len(self.equity):
            return 0.0
        peak = np.maximum.accumulate(np.r_[self.capital, self.equity])
        return float(np.max(1 - np.r_[self.capital, self.equity] / peak))


def run_signals(data: LabData, fs: FeatureSet, long_sig: np.ndarray, short_sig: np.ndarray, exit_sig: np.ndarray,
                params: dict[str, float], avoid_events: bool, t0: int, t1: int,
                capital: float = config.INITIAL_BALANCE, cost_multiplier: float = 1.0) -> SimResult:
    """Simulate pre-computed signal arrays ([n_symbols, t1-t0] bool) over bars [t0, t1)."""
    per_symbol = capital / data.n_symbols
    out = _simulate(
        data.open, data.high, data.low, data.close, fs.atr, data.day_id, data.last_bar_of_day,
        data.minute_of_day, fs.event_block, long_sig, short_sig, exit_sig, t0, t1, per_symbol,
        params["stop_atr"], params["tp_atr"], params["trail_atr"], int(params["max_hold"]), params["risk_pct"],
        int(params["start_min"]), int(params["end_min"]), params["daily_loss"], avoid_events,
        config.LAB_BASE_COST_BPS / 1e4 * cost_multiplier, config.LAB_SPREAD_FACTOR * cost_multiplier,
        config.LAB_MAX_COST_BPS / 1e4 * cost_multiplier,
    )
    equity_by_sym, *trades = out
    equity = equity_by_sym.sum(axis=0)
    day_end = np.flatnonzero(data.last_bar_of_day[t0:t1])
    if len(day_end) == 0 or day_end[-1] != t1 - t0 - 1:
        day_end = np.r_[day_end, t1 - t0 - 1]
    return SimResult(t0, t1, equity, equity[day_end], data.day_id[t0 + day_end], *trades, capital=capital)
