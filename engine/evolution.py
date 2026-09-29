"""
EvolutionEngine - the endless cycle of trade, judge, purge, replicate.

    Generation N:  [Alpha] + [10 mutated clones]  ->  all trade the same folds
                   -> Evaluator ranks them -> 1 victor, the rest PERMANENTLY deleted
                   -> victor saved to checkpoints/alpha_gen_N.json
                   -> mutation rate adapts (stuck = mutate harder, progress = fine-tune)
                   -> victor replicates Gaussian-mutated clones
    Generation N+1 begins.

Integrity checks run every generation (population size, lineage, unique IDs,
verified memory purge) and raise EvolutionIntegrityError if a rule is broken,
so a video never shows a bug pretending to be natural selection.
"""

from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

import config
from bot.genome import TradingGenome
from bot.trader import BotRole, TradingBot, reserve_bot_ids_above
from broker.base import CostModel, MarketData, Windows
from engine.evaluator import (
    CurveStats,
    EvolutionIntegrityError,
    Evaluator,
    FitnessReport,
    buy_and_hold_curve,
    summarize_curve,
)
from utils.charts import render_evolution_report
from utils.display import Display

log = logging.getLogger(__name__)

CHECKPOINT_SCHEMA_VERSION = 1
HISTORY_FILE = "evolution_history.jsonl"
CHART_FILE = "evolution_report.png"


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SeedAlpha:
    """Where a run starts: a fresh genome or a resumed checkpoint."""

    genome: TradingGenome
    origin_genome: TradingGenome
    bot_id: int | None = None
    start_generation: int = 1
    generations_survived: int = 0
    mutation_rate: float | None = None      # resume with the adapted rate
    origin_label: str = "Origin Genome (Gen 1)"


@dataclass(frozen=True)
class GenerationRecord:
    generation: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    victor: FitnessReport
    incumbent_retained: bool
    eliminated: list[FitnessReport]
    purge_verified: bool
    drift_from_previous: dict[str, tuple[float | int, float | int]]
    checkpoint_path: Path
    mutation_rate: float = 0.0              # rate used to create this generation's clones


@dataclass
class EvolutionSummary:
    run_id: str
    history: list[GenerationRecord]
    origin_genome: TradingGenome
    champion_genome: TradingGenome
    origin_label: str = "Origin Genome (Gen 1)"
    showdown: dict[str, dict[str, CurveStats | None]] = field(default_factory=dict)
    has_test_window: bool = False
    checkpoint_label: str = ""
    chart_label: str | None = None
    history_label: str = ""
    interrupted: bool = False

    def reigns(self) -> list[dict[str, Any]]:
        """Collapse the history into consecutive reigns of the same Alpha."""
        out: list[dict[str, Any]] = []
        for h in self.history:
            v = h.victor
            if out and out[-1]["bot_id"] == v.bot_id:
                reign = out[-1]
                reign["end"] = h.generation
                reign["best_fitness"] = max(reign["best_fitness"], v.fitness)
                reign["best_pnl"] = max(reign["best_pnl"], v.net_profit_pct)
            else:
                out.append({"bot_id": v.bot_id, "start": h.generation, "end": h.generation,
                            "best_fitness": v.fitness, "best_pnl": v.net_profit_pct})
        return out


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def _relative_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read and sanity-check an alpha checkpoint file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvolutionIntegrityError(f"Checkpoint not found: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise EvolutionIntegrityError(f"Checkpoint {path} is unreadable: {exc}") from exc
    for key in ("generation", "genome", "survivor"):
        if key not in data:
            raise EvolutionIntegrityError(f"Checkpoint {path} is missing '{key}'")
    return data


def check_timeframe(data: dict[str, Any], path: Path) -> None:
    """A genome evolved on daily bars means something different on 5-minute bars (and vice versa)."""
    saved = data.get("timeframe", "1d")  # checkpoints from before the timeframe switch were daily
    if saved != config.TIMEFRAME:
        raise EvolutionIntegrityError(
            f"{Path(path).name} was evolved on {saved} bars but TIMEFRAME is {config.TIMEFRAME}. "
            f"Start a fresh lineage (answer 'n' to continue), or set TIMEFRAME={saved} in .env / config.py."
        )


def seed_from_checkpoint(path: Path) -> SeedAlpha:
    """Resume the lineage: the saved Alpha starts the next generation."""
    data = load_checkpoint(path)
    check_timeframe(data, path)
    genome = TradingGenome.from_dict(data["genome"])
    survivor = data["survivor"]
    if data.get("origin_genome"):
        origin, origin_label = TradingGenome.from_dict(data["origin_genome"]), "Origin Genome (Gen 1)"
    else:  # checkpoints from older versions didn't store the true ancestor
        origin, origin_label = genome, f"Resumed Alpha (Gen {data['generation']})"
    reserve_bot_ids_above(int(data.get("max_bot_id", survivor["bot_id"])))
    rate = data.get("mutation_rate")
    return SeedAlpha(
        genome=genome,
        origin_genome=origin,
        bot_id=int(survivor["bot_id"]),
        start_generation=int(data["generation"]) + 1,
        generations_survived=int(data.get("generations_survived", 1)),
        mutation_rate=float(rate) if rate else None,
        origin_label=origin_label,
    )


def archive_previous_run(checkpoint_dir: Path = config.CHECKPOINT_DIR) -> Path | None:
    """Move an old run's checkpoints into checkpoints/archive/run_<timestamp>/ (never deletes)."""
    candidates = sorted(checkpoint_dir.glob("alpha_gen_*.json"))
    for name in (config.LATEST_ALPHA_FILE, CHART_FILE):
        if (checkpoint_dir / name).is_file():
            candidates.append(checkpoint_dir / name)
    if not candidates:
        return None
    dest = checkpoint_dir / "archive" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    dest.mkdir(parents=True, exist_ok=True)
    for file in candidates:
        file.replace(dest / file.name)
    return dest


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class EvolutionEngine:
    def __init__(
        self,
        market: MarketData,
        windows: Windows,
        display: Display,
        *,
        evaluator: Evaluator | None = None,
        costs: CostModel | None = None,
        population_size: int = config.POPULATION_SIZE,
        mutation_rate: float = config.MUTATION_RATE,
        adaptive_mutation: bool = config.ADAPTIVE_MUTATION,
        eval_folds: int = config.EVAL_FOLDS,
        initial_balance: float = config.INITIAL_BALANCE,
        checkpoint_dir: Path = config.CHECKPOINT_DIR,
        log_dir: Path = config.LOG_DIR,
        rng: random.Random | None = None,
    ) -> None:
        if population_size < 2:
            raise EvolutionIntegrityError("population_size must be >= 2")
        self.market = market
        self.windows = windows
        self.display = display
        self.evaluator = evaluator or Evaluator()
        self.costs = costs or CostModel.from_config()
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.adaptive_mutation = adaptive_mutation
        self.initial_balance = initial_balance
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.rng = rng or random.Random()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folds = self._make_folds(eval_folds)
        self.origin_genome: TradingGenome | None = None

    @property
    def clone_count(self) -> int:
        return self.population_size - 1

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, generations: int, seed: SeedAlpha) -> EvolutionSummary:
        if generations < 1:
            raise ValueError("generations must be >= 1")
        first = seed.start_generation
        last = first + generations - 1
        self.origin_genome = seed.origin_genome
        if seed.mutation_rate is not None and self.adaptive_mutation:
            self.mutation_rate = min(max(seed.mutation_rate, config.MUTATION_RATE_MIN), config.MUTATION_RATE_MAX)
        history: list[GenerationRecord] = []
        interrupted = False

        population = self._genesis(seed, first)
        champion: TradingBot | None = None
        try:
            for generation in range(first, last + 1):
                champion = None  # drop our handle on the old Alpha so it can die if dethroned
                champion, record = self._run_generation(population, generation, last)
                history.append(record)
                if generation < last:
                    population.extend([champion, *self._replicate(champion, generation + 1)])
                    self._show_mutations(population, generation + 1)
        except KeyboardInterrupt:
            interrupted = True
            population.clear()

        champion_genome = history[-1].victor.genome if history else seed.genome
        champion_id = history[-1].victor.bot_id if history else None
        summary = EvolutionSummary(
            run_id=self.run_id,
            history=history,
            origin_genome=seed.origin_genome,
            champion_genome=champion_genome,
            origin_label=seed.origin_label,
            has_test_window=self.windows.has_test,
            checkpoint_label=_relative_label(self.checkpoint_dir / config.LATEST_ALPHA_FILE),
            history_label=_relative_label(self.log_dir / HISTORY_FILE),
            interrupted=interrupted,
        )
        if history:
            summary.showdown, curves, curves_title = self._showdown(champion_genome, champion_id, seed.origin_genome, seed.origin_label)
            chart = render_evolution_report(
                generations=[h.generation for h in history],
                fitness=[h.victor.fitness for h in history],
                crown_changes=[h.generation for h in history if not h.incumbent_retained],
                curves=curves,
                curves_title=curves_title,
                output_path=self.checkpoint_dir / CHART_FILE,
            )
            summary.chart_label = _relative_label(chart) if chart else None
        return summary

    # ------------------------------------------------------------------ #
    # One generation
    # ------------------------------------------------------------------ #
    def _genesis(self, seed: SeedAlpha, generation: int) -> list[TradingBot]:
        alpha = TradingBot(
            seed.genome,
            bot_id=seed.bot_id,
            role=BotRole.ALPHA,
            generation_born=generation,
            initial_balance=self.initial_balance,
        )
        alpha.generations_survived = seed.generations_survived
        population = [alpha, *self._replicate(alpha, generation)]
        self._show_mutations(population, generation)
        return population

    def _run_generation(self, population: list[TradingBot], generation: int, last: int) -> tuple[TradingBot, GenerationRecord]:
        """Trade -> judge -> purge -> crown. Empties `population` in place and returns the victor."""
        self._check_population(population, generation)
        start, end = self.windows.train_start, self.windows.train_end
        clone_rate = self.mutation_rate  # the rate that produced this generation's clones
        # Remember the incumbent by value only - a live reference would keep it
        # alive through the purge if a clone dethrones it.
        previous_genome = population[0].genome

        self.display.generation_header(generation, last, self.folds, len(self.market.bars))
        self._evaluate(population)

        result = self.evaluator.select_and_purge(population, generation)
        self.display.leaderboard(result)
        self.display.extinction_log(result)
        self.display.purge_verification(result)
        if not result.purge_verified:
            raise EvolutionIntegrityError(
                f"Generation {generation}: {result.lingering_refs} eliminated bot(s) survived the purge"
            )

        victor = result.victor
        will_replicate = generation < last
        self.display.victory_banner(result, self.clone_count if will_replicate else 0)
        victor.promote_to_alpha()
        self._adapt_mutation(result.incumbent_retained, show=will_replicate)

        record = GenerationRecord(
            generation=generation,
            window_start=start,
            window_end=end,
            victor=result.victor_report,
            incumbent_retained=result.incumbent_retained,
            eliminated=result.eliminated,
            purge_verified=result.purge_verified,
            drift_from_previous=previous_genome.diff(victor.genome),
            checkpoint_path=self.checkpoint_dir / f"alpha_gen_{generation}.json",
            mutation_rate=clone_rate,
        )
        self._save_checkpoint(record, victor, result.leaderboard)
        self._append_history({
            "event": "selection",
            "run_id": self.run_id,
            "generation": generation,
            "window": [str(start.date()), str(end.date())],
            "victor": result.victor_report.to_dict(),
            "incumbent_retained": result.incumbent_retained,
            "eliminated": [
                {"bot_id": r.bot_id, "net_profit_pct": round(r.net_profit_pct, 4), "fitness": round(r.fitness, 6)}
                for r in result.eliminated
            ],
            "genetic_drift": {g: list(v) for g, v in record.drift_from_previous.items()},
            "mutation_rate_used": round(clone_rate, 4),
            "next_mutation_rate": round(self.mutation_rate, 4),
            "purge_verified": result.purge_verified,
        })
        self.display.checkpoint_saved(_relative_label(record.checkpoint_path))
        return victor, record

    def _evaluate(self, population: list[TradingBot]) -> None:
        """Every bot independently trades every fold."""
        with self.display.arena_progress(len(population)) as advance:
            for bot in population:
                bot.run_folds(self.market, self.folds, self.costs)
                advance(bot.name)

    def _make_folds(self, n_folds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Cut the training window into consecutive periods of (roughly) equal length."""
        w = self.windows
        cal = self.market.calendar
        train = cal[(cal >= w.train_start) & (cal <= w.train_end)]
        n = max(1, min(n_folds, len(train) // 90))  # every fold gets at least ~3 months
        bounds = [round(i * len(train) / n) for i in range(n + 1)]
        return [(train[bounds[i]], train[bounds[i + 1] - 1]) for i in range(n)]

    def _adapt_mutation(self, incumbent_retained: bool, show: bool) -> None:
        """Stuck -> explore harder. New champion -> fine-tune gently."""
        if not self.adaptive_mutation:
            return
        old = self.mutation_rate
        if incumbent_retained:
            new = min(config.MUTATION_RATE_MAX, old * config.MUTATION_STUCK_BOOST)
            reason = "Alpha defended - evolution is stuck, mutating harder to explore"
        else:
            new = max(config.MUTATION_RATE_MIN, old * config.MUTATION_PROGRESS_DECAY)
            reason = "New champion - fine-tuning with gentler mutations"
        self.mutation_rate = new
        if show:
            self.display.mutation_rate_update(old, new, reason)

    def _replicate(self, parent: TradingBot, generation: int) -> list[TradingBot]:
        """The survivor clones itself with Gaussian mutations."""
        clones = [parent.spawn_clone(self.mutation_rate, generation, self.rng) for _ in range(self.clone_count)]
        if any(c.parent_id != parent.bot_id or c.role is not BotRole.CLONE for c in clones):
            raise EvolutionIntegrityError("Clone lineage corrupted during replication")
        self._append_history({
            "event": "replication",
            "run_id": self.run_id,
            "generation": generation,
            "parent_id": parent.bot_id,
            "clones": [
                {"bot_id": c.bot_id, "mutations": {g: list(v) for g, v in parent.genome.diff(c.genome).items()}}
                for c in clones
            ],
        })
        return clones

    def _show_mutations(self, population: list[TradingBot], generation: int) -> None:
        alpha = population[0]
        self.display.mutation_table(generation, alpha.bot_id, alpha.genome,
                                    [(c.bot_id, c.genome) for c in population[1:]], self.mutation_rate)

    def _check_population(self, population: list[TradingBot], generation: int) -> None:
        """Enforce: exactly POPULATION_SIZE bots, one Alpha first, clones descend from it, unique IDs."""
        if len(population) != self.population_size:
            raise EvolutionIntegrityError(
                f"Generation {generation} has {len(population)} bots, expected {self.population_size}"
            )
        alpha, clones = population[0], population[1:]
        if alpha.role is not BotRole.ALPHA:
            raise EvolutionIntegrityError(f"Generation {generation}: first bot is not the Alpha")
        if any(c.role is not BotRole.CLONE or c.parent_id != alpha.bot_id for c in clones):
            raise EvolutionIntegrityError(f"Generation {generation}: every clone must descend from the Alpha")
        if len({b.bot_id for b in population}) != len(population):
            raise EvolutionIntegrityError(f"Generation {generation}: duplicate bot IDs")

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _save_checkpoint(self, record: GenerationRecord, victor: TradingBot, leaderboard: list[FitnessReport]) -> None:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation": record.generation,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "genome": victor.genome.to_dict(),
            "timeframe": config.TIMEFRAME,
            "dna": victor.genome.fingerprint(),
            "generations_survived": victor.generations_survived,
            "origin_genome": self.origin_genome.to_dict() if self.origin_genome else None,
            "mutation_rate": round(self.mutation_rate, 6),
            "survivor": record.victor.to_dict(),
            "incumbent_retained": record.incumbent_retained,
            "eliminated": [r.to_dict() for r in record.eliminated],
            "max_bot_id": max(r.bot_id for r in leaderboard),
            "evaluation_window": [str(record.window_start.date()), str(record.window_end.date())],
            "evaluation_folds": [[str(a.date()), str(b.date())] for a, b in self.folds],
            "market": {"source": self.market.source, "symbols": self.market.symbols},
            "settings": {
                "population_size": self.population_size,
                "adaptive_mutation": self.adaptive_mutation,
                "min_trades": self.evaluator.min_trades,
                "initial_balance": self.initial_balance,
                "commission_pct": self.costs.commission_pct,
                "crypto_commission_pct": self.costs.commission_rate("BTC-USD"),
                "slippage_pct": self.costs.slippage_pct,
            },
        }
        _atomic_write_json(record.checkpoint_path, payload)
        _atomic_write_json(self.checkpoint_dir / config.LATEST_ALPHA_FILE, payload)

    def _append_history(self, entry: dict[str, Any]) -> None:
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            with (self.log_dir / HISTORY_FILE).open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, default=str) + "\n")
        except OSError as exc:  # logging must never kill an evolution run
            log.warning("Could not append evolution history: %s", exc)

    # ------------------------------------------------------------------ #
    # Final showdown
    # ------------------------------------------------------------------ #
    def _showdown(
        self, champion: TradingGenome, champion_id: int | None, origin: TradingGenome, origin_label: str
    ) -> tuple[dict[str, dict[str, CurveStats | None]], dict[str, pd.Series], str]:
        """Champion vs origin genome vs buy & hold, in-sample and on the hold-out window."""
        w = self.windows
        contestants = {f"Evolved Champion (Bot #{champion_id})": champion, origin_label: origin}
        results: dict[str, dict[str, CurveStats | None]] = {}
        oos_curves: dict[str, pd.Series] = {}
        ins_curves: dict[str, pd.Series] = {}

        for label, genome in contestants.items():
            bot = TradingBot(genome, bot_id=0, initial_balance=self.initial_balance)
            ins_curve = bot.run(self.market, w.train_start, w.train_end, self.costs).equity_curve
            ins_curves[label] = ins_curve
            row: dict[str, CurveStats | None] = {
                "in_sample": replace(summarize_curve(ins_curve, self.initial_balance), win_rate=bot.win_rate),
                "out_of_sample": None,
            }
            if w.has_test:
                oos_curve = bot.run(self.market, w.test_start, w.test_end, self.costs).equity_curve
                oos_curves[label] = oos_curve
                row["out_of_sample"] = replace(summarize_curve(oos_curve, self.initial_balance), win_rate=bot.win_rate)
            results[label] = row

        bh_label = "Buy & Hold (equal weight)"
        ins_bh = buy_and_hold_curve(self.market, w.train_start, w.train_end, self.initial_balance, self.costs)
        ins_curves[bh_label] = ins_bh
        results[bh_label] = {"in_sample": summarize_curve(ins_bh, self.initial_balance), "out_of_sample": None}
        if w.has_test:
            oos_bh = buy_and_hold_curve(self.market, w.test_start, w.test_end, self.initial_balance, self.costs)
            oos_curves[bh_label] = oos_bh
            results[bh_label]["out_of_sample"] = summarize_curve(oos_bh, self.initial_balance)
            return results, oos_curves, f"Out-of-sample equity ({w.test_start.date()} → {w.test_end.date()})"
        return results, ins_curves, f"In-sample equity ({w.train_start.date()} → {w.train_end.date()})"
