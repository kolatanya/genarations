"""
TradingBot - a single contestant in the evolutionary arena.

Each bot owns a genome (its DNA) and a private paper portfolio. During a
generation it trades every symbol independently, bar by bar, emitting one of
four actions per symbol per bar:

    BUY             open a long position (signal at close -> fill next open)
    SELL            strategy exit signal on an open long (fill next open)
    HOLD            do nothing
    CLOSE_POSITION  forced risk exit: stop-loss, trailing stop, take-profit,
                    or end-of-window liquidation (fills intrabar)

The bot tracks its own equity curve, open positions, closed trades, stop-loss
triggers and win rate so the evaluator can score it afterwards.
"""

from __future__ import annotations

import itertools
import threading
from collections import Counter
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

import config
from bot.genome import TradingGenome
from bot.indicators import generate_signals
from broker.base import CostModel, MarketData


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_POSITION = "CLOSE_POSITION"


class ExitReason(str, Enum):
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TRAILING_STOP = "TRAILING_STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    END_OF_WINDOW = "END_OF_WINDOW"


class BotRole(str, Enum):
    ALPHA = "ALPHA"
    CLONE = "CLONE"


class BacktestError(RuntimeError):
    """Raised when a bot cannot be evaluated on the requested window."""


# --------------------------------------------------------------------------- #
# Bot ID allocation (global serial numbers -> "Bot #27" keeps its name forever)
# --------------------------------------------------------------------------- #
_id_lock = threading.Lock()
_id_counter = itertools.count(1)


def next_bot_id() -> int:
    with _id_lock:
        return next(_id_counter)


def reserve_bot_ids_above(max_existing_id: int) -> None:
    """After loading a checkpoint, make sure new clones never reuse old IDs."""
    global _id_counter
    with _id_lock:
        _id_counter = itertools.count(max(max_existing_id + 1, 1))


# --------------------------------------------------------------------------- #
# Portfolio records
# --------------------------------------------------------------------------- #
@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_date: pd.Timestamp
    peak_price: float
    entry_commission: float

    @property
    def cost_basis(self) -> float:
        return self.qty * self.entry_price + self.entry_commission


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    return_pct: float
    exit_reason: ExitReason


def check_risk_exit(
    genome: TradingGenome,
    entry_price: float,
    peak_price: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
) -> tuple[ExitReason, float] | None:
    """
    Decide whether a stop or target is hit inside one bar.

    Returns (reason, exit_price) or None. Pessimistic ordering: if a stop and a
    target are both touched in the same bar, the stop is assumed to hit first.
    Gaps through a level fill at the open, not at the level.
    """
    hard_stop = entry_price * (1.0 - genome.stop_loss_pct)
    trail_stop = peak_price * (1.0 - genome.trailing_stop_pct)
    if trail_stop > hard_stop:
        stop_price, stop_reason = trail_stop, ExitReason.TRAILING_STOP
    else:
        stop_price, stop_reason = hard_stop, ExitReason.STOP_LOSS
    target = entry_price * (1.0 + genome.take_profit_pct)

    if bar_open <= stop_price:
        return stop_reason, bar_open
    if bar_open >= target:
        return ExitReason.TAKE_PROFIT, bar_open
    if bar_low <= stop_price:
        return stop_reason, stop_price
    if bar_high >= target:
        return ExitReason.TAKE_PROFIT, target
    return None


@dataclass
class _SymbolTape:
    """Pre-computed numpy arrays for the fast backtest loop."""

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    buy: np.ndarray
    sell: np.ndarray
    index: pd.DatetimeIndex


@dataclass(frozen=True)
class FoldResult:
    """One evaluation period ("fold"): traded from a fresh balance."""

    start: pd.Timestamp
    end: pd.Timestamp
    equity_curve: pd.Series
    trade_count: int


PROFIT_FACTOR_CAP = 10.0  # a bot with no losing trades would otherwise score infinity


class TradingBot:
    """One genome + one paper portfolio."""

    def __init__(
        self,
        genome: TradingGenome,
        *,
        bot_id: int | None = None,
        role: BotRole = BotRole.CLONE,
        parent_id: int | None = None,
        generation_born: int = 1,
        initial_balance: float = config.INITIAL_BALANCE,
    ) -> None:
        if initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        self.genome = genome
        self.bot_id = bot_id if bot_id is not None else next_bot_id()
        self.role = role
        self.parent_id = parent_id
        self.generation_born = generation_born
        self.generations_survived = 0
        self.initial_balance = float(initial_balance)
        self.reset()

    # ------------------------------------------------------------------ #
    # Identity / lifecycle
    # ------------------------------------------------------------------ #
    @property
    def name(self) -> str:
        return f"Bot #{self.bot_id}"

    def __repr__(self) -> str:
        return f"<TradingBot {self.name} {self.role.value} dna={self.genome.fingerprint()}>"

    def reset(self) -> None:
        """Wipe everything so the bot can be evaluated from scratch."""
        self._reset_portfolio()
        self.trades: list[Trade] = []
        self.folds: list[FoldResult] = []
        self.action_counts: Counter[Action] = Counter()
        self.risk_triggers: Counter[ExitReason] = Counter()
        self.orders_filled: int = 0

    def _reset_portfolio(self) -> None:
        """Fresh balance and no positions (trade history is kept)."""
        self.cash: float = self.initial_balance
        self.positions: dict[str, Position] = {}
        self.equity_curve: pd.Series = pd.Series(dtype=float)

    def promote_to_alpha(self) -> None:
        """Called on the sole survivor of a generation."""
        self.role = BotRole.ALPHA
        self.generations_survived += 1

    def spawn_clone(self, mutation_rate: float, generation: int, rng) -> "TradingBot":
        """Replicate: a new bot with a Gaussian-mutated copy of this bot's DNA."""
        return TradingBot(
            self.genome.mutate(mutation_rate, rng),
            role=BotRole.CLONE,
            parent_id=self.bot_id,
            generation_born=generation,
            initial_balance=self.initial_balance,
        )

    def dispose(self) -> None:
        """Release every large structure (called when the bot is eliminated)."""
        self.positions.clear()
        self.trades.clear()
        self.folds.clear()
        self.equity_curve = pd.Series(dtype=float)
        self.action_counts.clear()
        self.risk_triggers.clear()

    # ------------------------------------------------------------------ #
    # Decision logic (shared with live trading)
    # ------------------------------------------------------------------ #
    @staticmethod
    def decide(buy_signal: bool, sell_signal: bool, in_position: bool) -> Action:
        """Translate the genome's signals into an action for one symbol."""
        if in_position:
            return Action.SELL if sell_signal else Action.HOLD
        return Action.BUY if buy_signal else Action.HOLD

    def check_risk(self, entry_price: float, peak_price: float, price: float) -> ExitReason | None:
        """Live-mode helper: does the current price breach a stop or target?"""
        hit = check_risk_exit(self.genome, entry_price, peak_price, price, price, price)
        return hit[0] if hit else None

    # ------------------------------------------------------------------ #
    # Backtest
    # ------------------------------------------------------------------ #
    def run(
        self,
        market: MarketData,
        start: pd.Timestamp,
        end: pd.Timestamp,
        costs: CostModel | None = None,
    ) -> "TradingBot":
        """
        Trade every symbol from `start` to `end` (inclusive) on daily bars.

        Indicators are computed on the FULL history so they are already warm
        when the window opens; trading only happens inside the window. Signals
        are generated at each bar's close and filled at the next bar's open
        (no look-ahead). All positions are liquidated on the final bar.
        """
        return self.run_folds(market, [(start, end)], costs)

    def run_folds(
        self,
        market: MarketData,
        windows: list[tuple[pd.Timestamp, pd.Timestamp]],
        costs: CostModel | None = None,
    ) -> "TradingBot":
        """
        Trade several separate periods ("folds"), each from a fresh balance.

        Trades, stop triggers and action counts accumulate across folds (so win
        rate and profit factor cover every period); each fold's equity curve is
        stored in `self.folds`. `self.equity_curve` holds the last fold.
        """
        if not windows:
            raise BacktestError(f"{self.name}: no evaluation windows given")
        costs = costs or CostModel.from_config()
        self.reset()

        # Indicators depend only on the genome + full history: compute them once.
        tapes: dict[str, _SymbolTape] = {}
        for symbol, bars in market.bars.items():
            signals = generate_signals(bars, self.genome)
            tapes[symbol] = _SymbolTape(
                open=bars["open"].to_numpy(dtype=float),
                high=bars["high"].to_numpy(dtype=float),
                low=bars["low"].to_numpy(dtype=float),
                close=bars["close"].to_numpy(dtype=float),
                buy=signals["buy"].to_numpy(dtype=bool),
                sell=signals["sell"].to_numpy(dtype=bool),
                index=bars.index,
            )

        for start, end in windows:
            self._reset_portfolio()
            trades_before = len(self.trades)
            self._simulate(market, tapes, start, end, costs)
            self.folds.append(FoldResult(start, end, self.equity_curve, len(self.trades) - trades_before))
        return self

    def _simulate(
        self,
        market: MarketData,
        tapes: dict[str, _SymbolTape],
        start: pd.Timestamp,
        end: pd.Timestamp,
        costs: CostModel,
    ) -> None:
        """Bar-by-bar trading of one window from the current (fresh) portfolio."""
        calendar = market.calendar[(market.calendar >= start) & (market.calendar <= end)]
        if len(calendar) < 2:
            raise BacktestError(f"{self.name}: evaluation window {start.date()} -> {end.date()} has < 2 bars")
        # calendar position -> row in each symbol's bars (-1 = no bar that day)
        rows = {symbol: tape.index.get_indexer(calendar) for symbol, tape in tapes.items()}

        last_close: dict[str, float] = {}
        pending: dict[str, Action] = {}
        equity = np.empty(len(calendar), dtype=float)
        genome = self.genome

        for t, day in enumerate(calendar):
            for symbol, tape in tapes.items():
                i = rows[symbol][t]
                if i < 0:
                    continue  # market closed for this symbol today (e.g. stocks on weekends)
                o, h, l, c = tape.open[i], tape.high[i], tape.low[i], tape.close[i]

                # 1) Fill yesterday's signal at today's open.
                order = pending.pop(symbol, None)
                if order is Action.BUY and symbol not in self.positions:
                    self._open_position(symbol, day, o, self._mark_to_market(last_close), costs)
                elif order is Action.SELL and symbol in self.positions:
                    self._close_position(symbol, day, o, ExitReason.SIGNAL, costs)

                # 2) Intrabar risk management on any open position.
                pos = self.positions.get(symbol)
                if pos is not None:
                    hit = check_risk_exit(genome, pos.entry_price, pos.peak_price, o, h, l)
                    if hit is not None:
                        reason, exit_price = hit
                        self.action_counts[Action.CLOSE_POSITION] += 1
                        self.risk_triggers[reason] += 1
                        self._close_position(symbol, day, exit_price, reason, costs)
                    else:
                        pos.peak_price = max(pos.peak_price, h)

                # 3) Read the genome's signal at the close -> order for tomorrow.
                action = self.decide(bool(tape.buy[i]), bool(tape.sell[i]), symbol in self.positions)
                self.action_counts[action] += 1
                if action is not Action.HOLD:
                    pending[symbol] = action
                last_close[symbol] = c

            equity[t] = self._mark_to_market(last_close)

        # 4) Evaluation window is over - flatten everything at the last close.
        for symbol in list(self.positions):
            self.action_counts[Action.CLOSE_POSITION] += 1
            self._close_position(symbol, calendar[-1], last_close[symbol], ExitReason.END_OF_WINDOW, costs)
        equity[-1] = self.cash

        self.equity_curve = pd.Series(equity, index=calendar, name=self.name)

    # ------------------------------------------------------------------ #
    # Portfolio accounting
    # ------------------------------------------------------------------ #
    def _mark_to_market(self, last_close: dict[str, float]) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            value += pos.qty * last_close.get(symbol, pos.entry_price)
        return value

    def _open_position(self, symbol: str, day: pd.Timestamp, ref_price: float, equity: float, costs: CostModel) -> None:
        fill = costs.fill_price("buy", ref_price)
        budget = min(equity * self.genome.position_size_pct, self.cash)
        if budget < config.MIN_ORDER_NOTIONAL or fill <= 0:
            return
        qty = budget / (fill * (1.0 + costs.commission_rate(symbol)))
        commission = costs.commission(qty * fill, symbol)
        self.cash -= qty * fill + commission
        self.positions[symbol] = Position(symbol, qty, fill, day, fill, commission)
        self.orders_filled += 1

    def _close_position(self, symbol: str, day: pd.Timestamp, ref_price: float, reason: ExitReason, costs: CostModel) -> None:
        pos = self.positions.pop(symbol)
        fill = costs.fill_price("sell", ref_price)
        proceeds = pos.qty * fill
        commission = costs.commission(proceeds, symbol)
        self.cash += proceeds - commission
        pnl = proceeds - commission - pos.cost_basis
        self.trades.append(
            Trade(
                symbol=symbol,
                entry_date=pos.entry_date,
                exit_date=day,
                entry_price=pos.entry_price,
                exit_price=fill,
                qty=pos.qty,
                pnl=pnl,
                return_pct=pnl / pos.cost_basis * 100.0 if pos.cost_basis else 0.0,
                exit_reason=reason,
            )
        )
        self.orders_filled += 1

    # ------------------------------------------------------------------ #
    # Quick stats
    # ------------------------------------------------------------------ #
    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.initial_balance

    @property
    def net_profit_pct(self) -> float:
        return (self.final_equity / self.initial_balance - 1.0) * 100.0

    @property
    def total_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        """Fraction of closed trades that made money (0.0 - 1.0)."""
        if not self.trades:
            return 0.0
        return sum(1 for t in self.trades if t.pnl > 0) / len(self.trades)

    @property
    def profit_factor(self) -> float:
        """Money won on winning trades / money lost on losing trades (capped)."""
        gross_win = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = -sum(t.pnl for t in self.trades if t.pnl < 0)
        if gross_loss <= 0:
            return PROFIT_FACTOR_CAP if gross_win > 0 else 0.0
        return min(gross_win / gross_loss, PROFIT_FACTOR_CAP)

    @property
    def stop_loss_hits(self) -> int:
        return self.risk_triggers[ExitReason.STOP_LOSS] + self.risk_triggers[ExitReason.TRAILING_STOP]
