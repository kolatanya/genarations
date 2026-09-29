"""
Evaluator - judges a generation, crowns ONE victor and purges everyone else.

Fitness = Net Profit (%) x Sharpe Ratio x (1 - Max Drawdown)

That formula only makes sense when profit and Sharpe are both positive: a bot
that LOST money with a negative Sharpe would otherwise score negative x
negative = positive and could beat a genuinely profitable bot. So losing bots
are scored separately (always below zero, worse drawdown = worse score).

On top of that base score:
  * it is computed on every evaluation fold and AVERAGED (consistency wins);
  * it is multiplied by a quality bonus for win rate and profit factor;
  * bots with fewer than MIN_TRADES trades are disqualified (ranked last).
"""

from __future__ import annotations

import gc
import math
import weakref
from dataclasses import dataclass, replace

import pandas as pd

import config
from bot.genome import TradingGenome
from bot.trader import BotRole, ExitReason, TradingBot
from broker.base import CostModel, MarketData


class EvolutionIntegrityError(RuntimeError):
    """The survival rules were violated (wrong population size, failed purge...)."""


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def periods_per_year(index: pd.DatetimeIndex) -> float:
    """Infer bar frequency from the index (365 when crypto adds weekends, ~252 otherwise)."""
    if len(index) < 2:
        return 252.0
    years = (index[-1] - index[0]).days / 365.25
    return len(index) / years if years > 0 else 252.0


def sharpe_ratio(equity: pd.Series) -> float:
    """Annualised Sharpe of bar-to-bar returns (risk-free rate = 0)."""
    returns = equity.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if not math.isfinite(std) or std < 1e-12:
        return 0.0
    value = float(returns.mean()) / std * math.sqrt(periods_per_year(equity.index))
    return value if math.isfinite(value) else 0.0


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough fall as a fraction (0.25 = -25%)."""
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    dd = 1.0 - equity / running_peak
    value = float(dd.max())
    return min(max(value, 0.0), 1.0) if math.isfinite(value) else 0.0


def compute_fitness(net_profit_pct: float, sharpe: float, max_dd: float) -> float:
    """
    Fitness = Net Profit (%) x Sharpe x (1 - Max Drawdown)   for profitable bots.

    Losing bots get  Net Profit (%) x (1 + Max Drawdown)  - always negative, so
    they can never outrank a winner, and deeper drawdowns rank lower.
    Flat bots (never traded) score exactly 0.
    """
    max_dd = min(max(max_dd, 0.0), 1.0)
    if net_profit_pct > 0 and sharpe > 0:
        return net_profit_pct * sharpe * (1.0 - max_dd)
    if net_profit_pct < 0:
        return net_profit_pct * (1.0 + max_dd)
    return 0.0


def quality_multiplier(
    win_rate: float,
    profit_factor: float,
    win_rate_weight: float = config.WIN_RATE_WEIGHT,
    profit_factor_weight: float = config.PROFIT_FACTOR_WEIGHT,
) -> float:
    """
    Bonus for winning often AND winning big.

      win part:  1 + weight x (win_rate - 0.5)    -> 40% wins = 0.9x, 60% wins = 1.1x
      PF part:   profit_factor ^ (weight / 2)      -> PF 1.0 = 1.0x, PF 2.0 = 1.41x

    Win rate alone is easy to game (tiny take-profit, huge stop = 90% wins that
    still lose money); profit factor closes that loophole.
    """
    win_part = max(1.0 + win_rate_weight * (win_rate - 0.5), 0.1)
    pf = min(max(profit_factor, 0.25), 4.0)
    return win_part * pf ** (profit_factor_weight / 2.0)


def frequency_multiplier(
    minutes_per_trade: float | None,
    target_minutes: float | None = config.TARGET_MINUTES_PER_TRADE,
    weight: float = config.TRADE_FREQUENCY_WEIGHT,
) -> float:
    """
    Nudge bots toward trading at the target pace (e.g. one trade every 20 minutes).

    1.0 at exactly the target; trading 2x too fast OR 2x too slow costs the same.
    With weight 0.75: 2x off keeps 59% of the score, 4x off keeps 35%.
    """
    if target_minutes is None or weight <= 0:
        return 1.0
    if minutes_per_trade is None or not math.isfinite(minutes_per_trade) or minutes_per_trade <= 0:
        ratio = 50.0  # never traded = as slow as it gets
    else:
        ratio = min(max(minutes_per_trade / target_minutes, 1 / 50), 50.0)
    return math.exp(-weight * abs(math.log(ratio)))


def apply_quality(base_fitness: float, multiplier: float) -> float:
    """Better quality always improves the score: winners are multiplied, losers are shrunk toward 0."""
    return base_fitness * multiplier if base_fitness >= 0 else base_fitness / multiplier


@dataclass(frozen=True)
class CurveStats:
    net_profit_pct: float
    sharpe: float
    max_drawdown: float
    fitness: float
    final_equity: float
    win_rate: float | None = None


def summarize_curve(equity: pd.Series, initial_balance: float) -> CurveStats:
    final = float(equity.iloc[-1]) if len(equity) else initial_balance
    net = (final / initial_balance - 1.0) * 100.0
    s = sharpe_ratio(equity)
    dd = max_drawdown(equity)
    return CurveStats(net, s, dd, compute_fitness(net, s, dd), final)


def buy_and_hold_curve(
    market: MarketData,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_balance: float,
    costs: CostModel | None = None,
) -> pd.Series:
    """Equal-weight buy & hold of every symbol - the benchmark every bot must beat."""
    costs = costs or CostModel()
    calendar = market.calendar[(market.calendar >= start) & (market.calendar <= end)]
    per_symbol = initial_balance / len(market.bars)
    total = pd.Series(0.0, index=calendar)
    for symbol, bars in market.bars.items():
        window = bars.loc[(bars.index >= start) & (bars.index <= end)]
        if window.empty:
            total += per_symbol
            continue
        fill = costs.fill_price("buy", float(window["open"].iloc[0]))
        shares = per_symbol / (fill * (1.0 + costs.commission_rate(symbol)))
        value = (window["close"] * shares).reindex(calendar).ffill().fillna(per_symbol)
        total += value
    return total.rename("Buy & Hold")


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FitnessReport:
    """Immutable scorecard. Holds NO reference to the bot, so bots can be purged."""

    bot_id: int
    role: BotRole
    parent_id: int | None
    generation_born: int
    generations_survived: int
    genome: TradingGenome
    net_profit_pct: float
    sharpe: float
    max_drawdown: float
    fitness: float
    total_trades: int
    win_rate: float
    stop_loss_hits: int
    take_profit_hits: int
    final_equity: float
    profit_factor: float = 0.0
    base_fitness: float = 0.0          # fold-averaged score before the quality bonus
    folds_profitable: int = 0
    n_folds: int = 1
    disqualified: bool = False
    minutes_per_trade: float | None = None   # average time between trades (None = no trades)
    pace_error: float = 0.0                  # |ln(actual pace / target)|, 0 = spot on (inf = never traded)
    on_pace: bool = True                     # inside the pace band (always True with no target)
    rank: int = 0

    @property
    def name(self) -> str:
        return f"Bot #{self.bot_id}"

    @property
    def trade_pace(self) -> str:
        """Human-friendly 'one trade every ...' label."""
        m = self.minutes_per_trade
        if m is None:
            return "never"
        if m < 120:
            return f"{m:.0f}m"
        if m < 2880:
            return f"{m / 60:.1f}h"
        return f"{m / 1440:.1f}d"

    def to_dict(self) -> dict:
        return {
            "bot_id": self.bot_id,
            "role": self.role.value,
            "parent_id": self.parent_id,
            "generation_born": self.generation_born,
            "generations_survived": self.generations_survived,
            "rank": self.rank,
            "fitness": round(self.fitness, 6),
            "net_profit_pct": round(self.net_profit_pct, 4),
            "sharpe": round(self.sharpe, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "base_fitness": round(self.base_fitness, 6),
            "folds_profitable": self.folds_profitable,
            "n_folds": self.n_folds,
            "disqualified": self.disqualified,
            "minutes_per_trade": round(self.minutes_per_trade, 2) if self.minutes_per_trade else None,
            "on_pace": self.on_pace,
            "stop_loss_hits": self.stop_loss_hits,
            "take_profit_hits": self.take_profit_hits,
            "final_equity": round(self.final_equity, 2),
            "genome": self.genome.to_dict(),
        }


@dataclass
class SelectionResult:
    generation: int
    victor: TradingBot
    victor_report: FitnessReport
    leaderboard: list[FitnessReport]
    eliminated: list[FitnessReport]
    purged_count: int
    lingering_refs: int

    @property
    def purge_verified(self) -> bool:
        """True when every eliminated bot has actually been garbage-collected."""
        return self.lingering_refs == 0 and self.purged_count == len(self.eliminated)

    @property
    def incumbent_retained(self) -> bool:
        return self.victor_report.role is BotRole.ALPHA


# --------------------------------------------------------------------------- #
# Evaluator
# --------------------------------------------------------------------------- #
class Evaluator:
    """Scores bots and enforces the one-survivor rule."""

    def __init__(
        self,
        min_trades: int = config.MIN_TRADES,
        win_rate_weight: float = config.WIN_RATE_WEIGHT,
        profit_factor_weight: float = config.PROFIT_FACTOR_WEIGHT,
        target_minutes_per_trade: float | None = config.TARGET_MINUTES_PER_TRADE,
        frequency_weight: float = config.TRADE_FREQUENCY_WEIGHT,
        pace_tolerance: float | None = config.PACE_TOLERANCE,
    ) -> None:
        self.pace_tolerance = pace_tolerance
        self.min_trades = min_trades
        self.win_rate_weight = win_rate_weight
        self.profit_factor_weight = profit_factor_weight
        self.target_minutes_per_trade = target_minutes_per_trade
        self.frequency_weight = frequency_weight

    def score(self, bot: TradingBot) -> FitnessReport:
        """
        Average the base fitness over every fold, then apply the quality bonus.
        Displayed PnL / Sharpe are fold averages; Max DD is the worst fold.
        """
        if not bot.folds:
            raise EvolutionIntegrityError(f"{bot.name} has not traded yet - call bot.run() first")
        fold_stats = [summarize_curve(f.equity_curve, bot.initial_balance) for f in bot.folds]
        n = len(fold_stats)
        base = sum(s.fitness for s in fold_stats) / n
        # Pace is measured over MARKET-OPEN time: every bar in the calendar is a slice
        # of time when at least one of the bot's markets was trading.
        window_minutes = sum(len(f.equity_curve) for f in bot.folds) * config.BAR_MINUTES
        minutes_per_trade = window_minutes / bot.total_trades if bot.total_trades else None
        multiplier = quality_multiplier(bot.win_rate, bot.profit_factor,
                                        self.win_rate_weight, self.profit_factor_weight)
        multiplier *= frequency_multiplier(minutes_per_trade, self.target_minutes_per_trade, self.frequency_weight)
        return FitnessReport(
            bot_id=bot.bot_id,
            role=bot.role,
            parent_id=bot.parent_id,
            generation_born=bot.generation_born,
            generations_survived=bot.generations_survived,
            genome=bot.genome,
            net_profit_pct=sum(s.net_profit_pct for s in fold_stats) / n,
            sharpe=sum(s.sharpe for s in fold_stats) / n,
            max_drawdown=max(s.max_drawdown for s in fold_stats),
            fitness=apply_quality(base, multiplier),
            total_trades=bot.total_trades,
            win_rate=bot.win_rate,
            stop_loss_hits=bot.stop_loss_hits,
            take_profit_hits=bot.risk_triggers[ExitReason.TAKE_PROFIT],
            final_equity=sum(s.final_equity for s in fold_stats) / n,
            profit_factor=bot.profit_factor,
            base_fitness=base,
            folds_profitable=sum(1 for s in fold_stats if s.net_profit_pct > 0),
            n_folds=n,
            disqualified=bot.total_trades < self.min_trades,
            minutes_per_trade=minutes_per_trade,
            **self._pace(minutes_per_trade),
        )

    def _pace(self, minutes_per_trade: float | None) -> dict:
        target = self.target_minutes_per_trade
        if target is None:
            return {"pace_error": 0.0, "on_pace": True}
        error = abs(math.log(minutes_per_trade / target)) if minutes_per_trade else math.inf
        on_pace = self.pace_tolerance is None or error <= math.log(self.pace_tolerance) + 1e-9
        return {"pace_error": error, "on_pace": on_pace}

    @staticmethod
    def rank(reports: list[FitnessReport]) -> list[FitnessReport]:
        """
        1. Qualified bots (enough trades) before disqualified ones.
        2. On-pace bots before off-pace ones; among off-pace bots the one
           CLOSEST to the target pace ranks higher.
        3. Highest fitness. Ties break on net profit, then Sharpe, and finally
           in favour of the incumbent Alpha - a challenger must strictly beat
           the reigning champion to take the crown.
        """
        def key(r: FitnessReport) -> tuple:
            pace_rank = 0.0 if r.on_pace else -r.pace_error
            return (not r.disqualified, r.on_pace, pace_rank, r.fitness, r.net_profit_pct, r.sharpe,
                    r.role is BotRole.ALPHA)

        ordered = sorted(reports, key=key, reverse=True)
        return [replace(r, rank=i + 1) for i, r in enumerate(ordered)]

    def select_and_purge(self, population: list[TradingBot], generation: int) -> SelectionResult:
        """
        Rank the population, keep exactly ONE victor and permanently delete the rest.

        The caller's `population` list is emptied IN PLACE. Eliminated bots are
        disposed, dereferenced and garbage-collected; weak references confirm
        that none of them survive in memory.
        """
        if len(population) < 2:
            raise EvolutionIntegrityError(f"Need at least 2 bots to hold a selection, got {len(population)}")
        ids = [b.bot_id for b in population]
        if len(set(ids)) != len(ids):
            raise EvolutionIntegrityError(f"Duplicate bot IDs in population: {ids}")

        leaderboard = self.rank([self.score(b) for b in population])
        victor_report = leaderboard[0]
        eliminated = leaderboard[1:]
        victor = next(b for b in population if b.bot_id == victor_report.bot_id)

        purged, lingering = self._purge(population, keep=victor)
        if population:
            raise EvolutionIntegrityError("Population list was not emptied during the purge")

        return SelectionResult(
            generation=generation,
            victor=victor,
            victor_report=victor_report,
            leaderboard=leaderboard,
            eliminated=eliminated,
            purged_count=purged,
            lingering_refs=lingering,
        )

    @staticmethod
    def _purge(population: list[TradingBot], keep: TradingBot) -> tuple[int, int]:
        """EXTINCTION: dispose + drop every bot except `keep`; return (purged, still_alive)."""
        doomed = [b for b in population if b is not keep]
        tombstones = [weakref.ref(b) for b in doomed]
        population.clear()
        while doomed:
            doomed.pop().dispose()  # pop() leaves no lingering local reference
        gc.collect()
        still_alive = sum(1 for ref in tombstones if ref() is not None)
        return len(tombstones), still_alive


__all__ = [
    "CurveStats",
    "EvolutionIntegrityError",
    "Evaluator",
    "FitnessReport",
    "SelectionResult",
    "apply_quality",
    "frequency_multiplier",
    "buy_and_hold_curve",
    "compute_fitness",
    "max_drawdown",
    "quality_multiplier",
    "sharpe_ratio",
    "summarize_curve",
]
