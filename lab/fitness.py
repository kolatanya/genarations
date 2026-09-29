"""
Lab scoring: how good is a bot, and how sure can we be it's skill, not luck?

Fitness (for positive Sharpe):
    Sharpe x (0.5 + 0.5 x share of profitable eras) x regime factor
           x (1 - worst drawdown) x complexity factor x pace factor
  * Sharpe          - annualised, from daily returns across the eras traded
  * regime factor   - (1 + number of market types it profits in) / 4:
                      up-trend, down-trend and sideways days (labelled at the
                      open from SPY's trend, so no hindsight)
  * complexity      - 1 / (1 + 0.02 x rule nodes): simple rules generalise better
  * pace            - nudges toward ~1 trade every 20 minutes (portfolio-wide)
Negative Sharpe is divided by the same factors, so better behaviour always helps.

Luck detectors:
  * plateau()        - wobble the genes; fragile bots collapse
  * monte_carlo()    - bootstrap daily returns into thousands of alternate histories
  * deflated_sharpe()- probability the best Sharpe isn't just the luckiest of N tries
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np

import config
from engine.evaluator import frequency_multiplier
from lab.data import LabData
from lab.features import FEATURE_INDEX, FeatureSet
from lab.genome import LabGenome
from lab.gp import TreeEvaluator
from lab.simulator import SimResult, run_signals

REGIMES = ("UP", "SIDEWAYS", "DOWN")
_N = NormalDist()


@dataclass(frozen=True)
class LabScore:
    fitness: float
    sharpe: float
    avg_return: float          # average return per era (fraction)
    max_dd: float              # worst era drawdown
    trades: int
    win_rate: float
    profit_factor: float
    minutes_per_trade: float | None
    eras_profitable: int
    n_eras: int
    regime_returns: tuple[float, float, float]   # mean daily return on UP / SIDEWAYS / DOWN days
    complexity: int
    disqualified: bool
    on_pace: bool
    pace_error: float
    daily_returns: np.ndarray = field(repr=False, compare=False, default_factory=lambda: np.zeros(0))

    @property
    def regimes_profitable(self) -> int:
        return sum(r > 0 for r in self.regime_returns)

    @property
    def trade_pace(self) -> str:
        m = self.minutes_per_trade
        if m is None:
            return "never"
        return f"{m:.0f}m" if m < 120 else (f"{m / 60:.1f}h" if m < 2880 else f"{m / 1440:.1f}d")

    def rank_key(self, is_alpha: bool = False) -> tuple:
        return (not self.disqualified, self.on_pace, 0.0 if self.on_pace else -self.pace_error,
                self.fitness, self.sharpe, is_alpha)


class Scorer:
    """Evaluates genomes on bar windows, caching rule evaluations per window."""

    def __init__(self, data: LabData, fs: FeatureSet) -> None:
        self.data = data
        self.fs = fs
        self._evaluators: dict[tuple[int, int], TreeEvaluator] = {}
        first_bars = data.day_bounds()[:, 0]
        trend = fs.values[FEATURE_INDEX["spy_d1_trend"], 0, first_bars]
        # Regime of each day, known at the open: SPY above/below its 20-day average by >1%
        self.day_regime = np.where(trend > 0.01, 0, np.where(trend < -0.01, 2, 1)).astype(np.int64)
        self.bar_regime = self.day_regime[data.day_id]
        self.trial_sharpes = _Welford()   # every Sharpe ever measured (for the deflated Sharpe)

    def evaluator(self, t0: int, t1: int) -> TreeEvaluator:
        ev = self._evaluators.get((t0, t1))
        if ev is None:
            if len(self._evaluators) > 24:
                self._evaluators.clear()
            ev = self._evaluators[(t0, t1)] = TreeEvaluator(self.fs, t0, t1)
        return ev

    def simulate(self, genome: LabGenome, t0: int, t1: int, cost_multiplier: float = 1.0) -> SimResult:
        ev = self.evaluator(t0, t1)
        return run_signals(self.data, self.fs, ev(genome.long_entry), ev(genome.short_entry), ev(genome.exit_rule),
                           genome.params, genome.avoid_events, t0, t1, cost_multiplier=cost_multiplier)

    def score(self, genome: LabGenome, windows: list[tuple[int, int]], cost_multiplier: float = 1.0,
              count_trial: bool = True) -> LabScore:
        return self.score_results([self.simulate(genome, a, b, cost_multiplier) for a, b in windows],
                                  genome.complexity, count_trial)

    def score_results(self, results: list[SimResult], complexity: int, count_trial: bool = True) -> LabScore:
        daily = np.concatenate([r.daily_returns for r in results]) if results else np.zeros(0)
        days = np.concatenate([r.day_index for r in results]) if results else np.zeros(0, dtype=np.int64)
        sharpe = _sharpe(daily)
        if count_trial and len(daily) > 5:
            self.trial_sharpes.add(sharpe / math.sqrt(252))
        trades = sum(r.n_trades for r in results)
        pnl = np.concatenate([r.trades_pnl for r in results]) if results else np.zeros(0)
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        pf = float(min(wins / losses, 10.0)) if losses > 0 else (10.0 if wins > 0 else 0.0)
        era_returns = [r.total_return for r in results]
        regime = tuple(float(daily[self.day_regime[days] == g].mean()) if np.any(self.day_regime[days] == g) else 0.0
                       for g in range(3))
        minutes = sum(r.t1 - r.t0 for r in results) * config.BAR_MINUTES_LAB
        mpt = minutes / trades if trades else None
        target = config.LAB_PACE_TARGET_MINUTES
        pace_err = abs(math.log(mpt / target)) if (target and mpt) else (0.0 if not target else math.inf)
        tol = config.PACE_TOLERANCE
        on_pace = target is None or tol is None or pace_err <= math.log(tol) + 1e-9
        max_dd = max((r.max_drawdown for r in results), default=0.0)
        eras_pos = sum(x > 0 for x in era_returns)

        factor = ((0.5 + 0.5 * eras_pos / max(len(results), 1))
                  * (1 + sum(g > 0 for g in regime)) / 4
                  * (1 - min(max_dd, 0.99))
                  / (1 + config.LAB_COMPLEXITY_PENALTY * complexity)
                  * frequency_multiplier(mpt, target, config.TRADE_FREQUENCY_WEIGHT))
        factor = max(factor, 1e-6)
        fitness = sharpe * factor if sharpe >= 0 else sharpe / factor
        return LabScore(
            fitness=float(fitness), sharpe=float(sharpe),
            avg_return=float(np.mean(era_returns)) if era_returns else 0.0, max_dd=float(max_dd),
            trades=int(trades), win_rate=float((pnl > 0).mean()) if trades else 0.0, profit_factor=pf,
            minutes_per_trade=mpt, eras_profitable=int(eras_pos), n_eras=len(results), regime_returns=regime,
            complexity=complexity, disqualified=trades < config.MIN_TRADES, on_pace=on_pace, pace_error=pace_err,
            daily_returns=daily,
        )

    def regime_weights(self, genome: LabGenome, windows: list[tuple[int, int]]) -> tuple[float, float, float]:
        """How well a bot does in each market type - used to weight its ensemble vote."""
        s = self.score(genome, windows, count_trial=False)
        return tuple(max(r, 0.0) * 1e4 + 1e-3 for r in s.regime_returns)

    # ------------------------------------------------------------------ #
    def plateau(self, genome: LabGenome, windows: list[tuple[int, int]], rng: random.Random,
                variants: int = config.LAB_WOBBLE_VARIANTS, pct: float = config.LAB_WOBBLE_PCT,
                cost_multiplier: float = 1.0) -> float:
        """Median fitness of slightly-wobbled copies. Real patterns sit on plateaus, flukes on needles."""
        scores = [self.score(genome.wobble(rng, pct), windows, cost_multiplier, count_trial=False).fitness
                  for _ in range(variants)]
        return float(np.median(scores))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def _sharpe(daily: np.ndarray) -> float:
    if len(daily) < 3:
        return 0.0
    sd = daily.std(ddof=1)
    if not np.isfinite(sd) or sd < 1e-12:
        return 0.0
    return float(daily.mean() / sd * math.sqrt(252))


class _Welford:
    def __init__(self) -> None:
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x: float) -> None:
        if not math.isfinite(x):
            return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.n - 1) if self.n > 1 else 0.0


def deflated_sharpe(daily: np.ndarray, n_trials: int, trial_sharpe_variance: float) -> float:
    """
    Bailey & Lopez de Prado's Deflated Sharpe Ratio: the probability that the
    selected strategy's true Sharpe is above zero AFTER accounting for how many
    strategies were tried. Near 1.0 = convincing, near 0.5 or below = likely luck.
    """
    t = len(daily)
    if t < 10 or n_trials < 2:
        return float("nan")
    sd = daily.std(ddof=1)
    if sd <= 0:
        return float("nan")
    sr = daily.mean() / sd                         # per-day Sharpe
    z = (daily - daily.mean()) / sd
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    gamma = 0.5772156649
    e = math.e
    sr0 = math.sqrt(max(trial_sharpe_variance, 1e-12)) * (
        (1 - gamma) * _N.inv_cdf(1 - 1 / n_trials) + gamma * _N.inv_cdf(1 - 1 / (n_trials * e)))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return float(_N.cdf((sr - sr0) * math.sqrt(t - 1) / denom))


@dataclass(frozen=True)
class MonteCarlo:
    p5: float
    p50: float
    p95: float
    prob_profit: float
    five_day_p5: float      # 5th percentile of any 5-day stretch - the live drift-alarm line


def monte_carlo(daily: np.ndarray, runs: int = config.LAB_MONTE_CARLO_RUNS, block: int = 5,
                seed: int = 0) -> MonteCarlo | None:
    """Block-bootstrap the daily returns into `runs` alternate histories of the same length."""
    n = len(daily)
    if n < 10:
        return None
    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block)
    starts = rng.integers(0, n - block + 1, size=(runs, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(runs, -1)[:, :n]
    totals = np.prod(1 + daily[idx], axis=1) - 1
    five = np.array([np.prod(1 + daily[i:i + 5]) - 1 for i in range(n - 4)])
    return MonteCarlo(float(np.percentile(totals, 5)), float(np.percentile(totals, 50)),
                      float(np.percentile(totals, 95)), float((totals > 0).mean()),
                      float(np.percentile(five, 5)) if len(five) else float("nan"))
