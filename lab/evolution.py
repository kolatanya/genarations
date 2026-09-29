"""
LabEngine - island evolution with robustness gates.

Each generation, on EVERY island:
  1 Alpha + 5 challengers: 3 mutants, 1 crossover child (Alpha x Hall-of-Famer),
  1 wildcard (a migrant from the next island every 5 gens, a returning
  Hall-of-Fame veteran, a brand-new random bot, or a wild explorer).
  All six trade a random 4 of the 6 training eras (with jittered costs - noise
  that punishes memorising). They're ranked, and a challenger only takes the
  crown if it also passes:
    * the WOBBLE test  - its genes nudged +/-10% must still beat the Alpha's
    * the VALIDATION gate - it must beat the Alpha on data it never trained on
  Everyone else is purged (weak-reference verified). One survivor per island.

Across islands:
  * a Hall of Fame keeps the 10 best champions ever (ranked on validation)
  * if two islands evolve the SAME bot, or an Alpha reigns 20 generations,
    a METEOR wipes that island and a random newcomer starts over
  * a Pareto archive keeps bots that are best at different trade-offs

The final 20% of history is never touched until the very end, when the
champion, the ensemble, the textbook origin bot and buy & hold face it.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import weakref
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

import config
from engine.evaluator import EvolutionIntegrityError
from lab.data import LabData
from lab.ensemble import Ensemble, Member
from lab.features import FeatureSet
from lab.fitness import LabScore, MonteCarlo, Scorer, deflated_sharpe, monte_carlo
from lab.genome import LabGenome
from lab.simulator import SimResult

log = logging.getLogger(__name__)

WARMUP_DAYS = 25            # daily-trend features need ~20 days of history
STAGNATION_LIMIT = 20       # an Alpha reigning this long gets a meteor
SCHEMA = 1


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class LabBot:
    bot_id: int
    genome: LabGenome
    island: int
    role: str                      # ALPHA / MUTANT / CROSSOVER / MIGRANT / VETERAN / RANDOM / EXPLORER
    parents: tuple[int, ...]
    born: int
    score: LabScore | None = None


@dataclass(frozen=True)
class Tombstone:
    """What remains of a purged bot: numbers only, no reference to the bot."""
    bot_id: int
    role: str
    score: LabScore


@dataclass
class Island:
    idx: int
    alpha: LabBot
    mutation_rate: float = config.MUTATION_RATE
    reign: int = 0


@dataclass
class IslandResult:
    island: int
    generation: int
    winner_id: int
    winner_role: str
    winner_score: LabScore
    status: str                    # DEFENDED / USURPED / BLOCKED_WOBBLE / BLOCKED_VALIDATION
    challenger_id: int | None
    eliminated: list[Tombstone]
    rate_before: float
    rate_after: float
    reign: int
    wobble: tuple[float, float] | None = None       # (challenger plateau, alpha plateau)
    validation: tuple[float, float] | None = None   # (challenger val Sharpe, alpha val Sharpe)
    purge_verified: bool = False


@dataclass
class HallEntry:
    bot_id: int
    genome: LabGenome
    island: int
    generation: int
    val_sharpe: float
    val_return: float
    train_fitness: float
    weights: tuple[float, float, float]

    def to_json(self) -> dict:
        return {"bot_id": self.bot_id, "genome": self.genome.to_json(), "island": self.island,
                "generation": self.generation, "val_sharpe": self.val_sharpe, "val_return": self.val_return,
                "train_fitness": self.train_fitness, "weights": list(self.weights)}

    @classmethod
    def from_json(cls, o: dict) -> "HallEntry":
        return cls(int(o["bot_id"]), LabGenome.from_json(o["genome"]), int(o["island"]), int(o["generation"]),
                   float(o["val_sharpe"]), float(o["val_return"]), float(o["train_fitness"]), tuple(o["weights"]))


@dataclass
class Showdown:
    name: str
    result: SimResult | None
    total_return: float
    sharpe: float
    max_dd: float
    trades: int
    win_rate: float
    profit_factor: float
    daily: np.ndarray
    equity_daily: np.ndarray
    monte_carlo: MonteCarlo | None = None


@dataclass
class LabSummary:
    generations: list[int]
    island_fitness: list[list[float]]           # per generation, per island Alpha fitness
    hall: list[HallEntry]
    champion: HallEntry | None
    ensemble: Ensemble | None
    showdown: list[Showdown]
    deflated_sharpe: float
    trials: int
    pareto: list[dict]
    windows: dict[str, str]
    checkpoint: str = ""
    chart: str | None = None
    interrupted: bool = False
    data_source: str = ""


@dataclass
class LabSeed:
    islands: list[dict]
    hall: list[dict]
    generation: int
    next_bot_id: int


# --------------------------------------------------------------------------- #
class LabEngine:
    def __init__(
        self,
        data: LabData,
        fs: FeatureSet,
        display,
        *,
        rng: random.Random | None = None,
        n_islands: int = config.LAB_ISLANDS,
        bots_per_island: int = config.LAB_BOTS_PER_ISLAND,
        split: tuple[float, float, float] = config.LAB_SPLIT,
        day_ranges: dict[str, tuple[int, int]] | None = None,
        n_folds: int = config.LAB_TRAIN_FOLDS,
        folds_per_gen: int = config.LAB_FOLDS_PER_GEN,
        out_dir: Path = config.LAB_DIR,
        save: bool = True,
        latest_name: str = "lab_latest.json",
    ) -> None:
        if bots_per_island < 3:
            raise EvolutionIntegrityError("bots_per_island must be >= 3")
        self.data, self.fs, self.display = data, fs, display
        self.rng = rng or random.Random()
        self.n_islands, self.bots_per_island = n_islands, bots_per_island
        self.out_dir, self.save = Path(out_dir), save
        self.latest_name = latest_name
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        n_days = len(data.day_bounds())
        if day_ranges is None:
            usable = n_days - WARMUP_DAYS
            a = WARMUP_DAYS
            b = a + int(usable * split[0])
            c = b + int(usable * split[1])
            day_ranges = {"train": (a, b), "validation": (b, c), "test": (c, n_days)}
        self.day_ranges = day_ranges
        tr = day_ranges["train"]
        fold_edges = np.linspace(tr[0], tr[1], n_folds + 1).astype(int)
        self.train_folds = [data.slice_days(int(fold_edges[i]), int(fold_edges[i + 1])) for i in range(n_folds)
                            if fold_edges[i + 1] > fold_edges[i]]
        self.folds_per_gen = min(folds_per_gen, len(self.train_folds))
        self.train_window = data.slice_days(*tr)
        self.val_window = data.slice_days(*day_ranges["validation"])
        self.test_window = data.slice_days(*day_ranges["test"]) if day_ranges["test"][1] > day_ranges["test"][0] else None
        fs.fit_quantiles(*self.train_window)            # thresholds from TRAINING data only
        self.scorer = Scorer(data, fs)

        self.islands: list[Island] = []
        self.hall: list[HallEntry] = []
        self.pareto: list[dict] = []
        self._val_cache: dict[str, LabScore] = {}
        self._next_id = 1
        self.generation = 0
        self.island_fitness: list[list[float]] = []
        self.generations_run: list[int] = []

    # ------------------------------------------------------------------ #
    def window_dates(self) -> dict[str, str]:
        def fmt(r):
            a, b = r
            if b <= a:
                return "-"
            return f"{self.data.unique_days[a]} → {self.data.unique_days[b - 1]}"
        return {k: fmt(v) for k, v in self.day_ranges.items()}

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def _bot(self, genome: LabGenome, island: int, role: str, parents: tuple[int, ...], gen: int) -> LabBot:
        return LabBot(self._new_id(), genome, island, role, parents, gen)

    # ------------------------------------------------------------------ #
    # Setup / resume
    # ------------------------------------------------------------------ #
    def seed(self, seed: LabSeed | None) -> None:
        if seed is not None:
            self._next_id = seed.next_bot_id
            self.generation = seed.generation
            self.islands = [Island(int(o["idx"]), LabBot(int(o["bot_id"]), LabGenome.from_json(o["genome"]),
                                                         int(o["idx"]), "ALPHA", (), int(o.get("born", 0))),
                                   float(o.get("mutation_rate", config.MUTATION_RATE)), int(o.get("reign", 0)))
                            for o in seed.islands][: self.n_islands]
            self.hall = [HallEntry.from_json(h) for h in seed.hall]
        while len(self.islands) < self.n_islands:
            i = len(self.islands)
            genome = LabGenome.textbook() if i == 0 else LabGenome.random(self.rng)
            self.islands.append(Island(i, self._bot(genome, i, "ALPHA", (), self.generation + 1)))

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self, generations: int, seed: LabSeed | None = None) -> LabSummary:
        self.seed(seed)
        first = self.generation + 1
        last = first + generations - 1
        interrupted = False
        try:
            for gen in range(first, last + 1):
                self.generation = gen
                self._run_generation(gen, last)
        except KeyboardInterrupt:
            interrupted = True
        summary = self.finalize()
        summary.interrupted = interrupted
        return summary

    def _run_generation(self, gen: int, last: int) -> None:
        folds = sorted(self.rng.sample(range(len(self.train_folds)), self.folds_per_gen))
        windows = [self.train_folds[i] for i in folds]
        cost_mult = self.rng.uniform(0.8, 1.3)           # noise injection: costs vary per generation
        self.display.lab_generation_header(gen, last, folds, len(self.train_folds), cost_mult)

        results: list[IslandResult] = []
        tombstones: list[weakref.ref] = []
        with self.display.lab_progress(len(self.islands) * self.bots_per_island) as tick:
            for isl in self.islands:
                res, dead = self._island_generation(isl, gen, windows, cost_mult, tick)
                results.append(res)
                tombstones += dead
        gc.collect()
        lingering = sum(1 for r in tombstones if r() is not None)
        for res in results:
            res.purge_verified = lingering == 0
        if lingering:
            raise EvolutionIntegrityError(f"Generation {gen}: {lingering} purged bots still in memory")

        events = self._update_hall(gen) + self._meteors(gen)
        self.island_fitness.append([isl.alpha.score.fitness if isl.alpha.score else 0.0 for isl in self.islands])
        self.generations_run.append(gen)
        self.display.lab_generation(gen, results, events, self.hall, self.scorer.trial_sharpes.n)
        if self.save:
            self._checkpoint(gen)

    def _island_generation(self, isl: Island, gen: int, windows, cost_mult: float, tick):
        alpha = isl.alpha
        bots = [alpha, *self._spawn(isl, gen)]
        for b in bots:
            b.score = self.scorer.score(b.genome, windows, cost_mult)
            tick(f"Island {isl.idx + 1}: Bot #{b.bot_id}")
        ranked = sorted(bots, key=lambda b: b.score.rank_key(b is alpha), reverse=True)
        top = ranked[0]
        status, wobble, val = "DEFENDED", None, None
        if top is not alpha:
            status = "USURPED"
            same_class = top.score.rank_key()[:3] == alpha.score.rank_key()[:3]
            if same_class:
                ch_p = self.scorer.plateau(top.genome, windows, self.rng, cost_multiplier=cost_mult)
                al_p = self.scorer.plateau(alpha.genome, windows, self.rng, cost_multiplier=cost_mult)
                wobble = (ch_p, al_p)
                if 0.5 * top.score.fitness + 0.5 * ch_p <= 0.5 * alpha.score.fitness + 0.5 * al_p:
                    status = "BLOCKED_WOBBLE"
            if status == "USURPED" and same_class:
                v_ch, v_al = self.val_score(top.genome).sharpe, self.val_score(alpha.genome).sharpe
                val = (v_ch, v_al)
                if v_ch < v_al:
                    status = "BLOCKED_VALIDATION"
        winner = top if status == "USURPED" else alpha

        # Adaptive mutation (per island): stuck -> explore, progress -> fine-tune
        before = isl.mutation_rate
        if winner is alpha:
            isl.mutation_rate = min(config.MUTATION_RATE_MAX, before * config.MUTATION_STUCK_BOOST)
            isl.reign += 1
        else:
            isl.mutation_rate = max(config.MUTATION_RATE_MIN, before * config.MUTATION_PROGRESS_DECAY)
            isl.reign = 1

        eliminated = [Tombstone(b.bot_id, b.role, b.score) for b in ranked if b is not winner]
        challenger_id = top.bot_id if top is not alpha else None
        self._add_pareto(bots)
        # EXTINCTION: drop every loser; weak references prove they're gone
        dead = [weakref.ref(b) for b in bots if b is not winner]
        winner_role = winner.role if winner is not alpha else "ALPHA"
        winner.role = "ALPHA"
        isl.alpha = winner
        bots.clear()
        ranked.clear()
        del top, alpha
        return IslandResult(isl.idx, gen, winner.bot_id, winner_role, winner.score, status, challenger_id,
                            eliminated, before, isl.mutation_rate, isl.reign, wobble, val), dead

    def _spawn(self, isl: Island, gen: int) -> list[LabBot]:
        alpha = isl.alpha
        rate = isl.mutation_rate
        seen = {alpha.genome.key()}
        kids: list[LabBot] = []

        def add(genome: LabGenome, role: str, parents: tuple[int, ...]) -> None:
            tries = 0
            while genome.key() in seen and tries < 4:
                genome = genome.mutate(self.rng, max(rate, 0.1))
                tries += 1
            seen.add(genome.key())
            kids.append(self._bot(genome, isl.idx, role, parents, gen))

        n_mutants = self.bots_per_island - 3
        for _ in range(n_mutants):
            add(alpha.genome.mutate(self.rng, rate), "MUTANT", (alpha.bot_id,))

        others = [i.alpha for i in self.islands if i is not isl]
        if self.hall and (self.rng.random() < 0.6 or not others):
            partner = self.rng.choice(self.hall)
            add(LabGenome.crossover(alpha.genome, partner.genome, self.rng), "CROSSOVER", (alpha.bot_id, partner.bot_id))
        elif others:
            partner_bot = self.rng.choice(others)
            add(LabGenome.crossover(alpha.genome, partner_bot.genome, self.rng), "CROSSOVER",
                (alpha.bot_id, partner_bot.bot_id))
        else:
            add(alpha.genome.mutate(self.rng, rate * 2), "MUTANT", (alpha.bot_id,))

        roll = self.rng.random()
        if gen % config.LAB_MIGRATION_EVERY == 0 and len(self.islands) > 1:
            neighbour = self.islands[(isl.idx + 1) % len(self.islands)].alpha
            add(neighbour.genome, "MIGRANT", (neighbour.bot_id,))
        elif self.hall and roll < 0.35:
            vet = self.rng.choice(self.hall)
            add(vet.genome, "VETERAN", (vet.bot_id,))
        elif roll < 0.5:
            add(LabGenome.random(self.rng), "RANDOM", ())
        else:
            add(alpha.genome.mutate(self.rng, min(0.6, rate * 3)), "EXPLORER", (alpha.bot_id,))
        return kids

    # ------------------------------------------------------------------ #
    def val_score(self, genome: LabGenome) -> LabScore:
        k = genome.key()
        hit = self._val_cache.get(k)
        if hit is None:
            hit = self._val_cache[k] = self.scorer.score(genome, [self.val_window], count_trial=False)
            if len(self._val_cache) > 2000:
                self._val_cache.clear()
        return hit

    def _update_hall(self, gen: int) -> list[str]:
        events: list[str] = []
        known = {h.genome.key() for h in self.hall}
        for isl in self.islands:
            a = isl.alpha
            if a.genome.key() in known or a.score is None or a.score.disqualified:
                continue
            v = self.val_score(a.genome)
            if v.trades < 5:
                continue
            weights = self.scorer.regime_weights(a.genome, [self.train_window, self.val_window])
            entry = HallEntry(a.bot_id, a.genome, isl.idx, gen, v.sharpe, v.avg_return, a.score.fitness, weights)
            self.hall.append(entry)
            known.add(a.genome.key())
            self.hall.sort(key=lambda h: (h.val_sharpe, h.train_fitness), reverse=True)
            if len(self.hall) > config.LAB_HALL_OF_FAME:
                dropped = self.hall.pop()
                if dropped is entry:
                    continue
            rank = self.hall.index(entry) + 1
            events.append(f"🏛  Bot #{a.bot_id} (Island {isl.idx + 1}) enters the Hall of Fame at #{rank} "
                          f"(validation Sharpe {v.sharpe:.2f})")
        return events

    def _meteors(self, gen: int) -> list[str]:
        events: list[str] = []
        seen: dict[str, Island] = {}
        for isl in self.islands:
            k = isl.alpha.genome.key()
            reason = None
            if k in seen:
                reason = f"evolved the same bot as Island {seen[k].idx + 1}"
            elif isl.reign >= STAGNATION_LIMIT:
                reason = f"Bot #{isl.alpha.bot_id} reigned {isl.reign} generations (stagnation)"
            seen.setdefault(k, isl)
            if reason:
                newcomer = self._bot(LabGenome.random(self.rng), isl.idx, "ALPHA", (), gen + 1)
                events.append(f"☄  METEOR STRIKE on Island {isl.idx + 1}: {reason}. "
                              f"Bot #{newcomer.bot_id} starts a new dynasty (the old Alpha lives on in the Hall of Fame).")
                isl.alpha = newcomer
                isl.reign = 0
                isl.mutation_rate = config.MUTATION_RATE
        return events

    def _add_pareto(self, bots: list[LabBot]) -> None:
        """Keep bots that are best at SOME trade-off of return, drawdown and consistency."""
        for b in bots:
            s = b.score
            if s is None or s.disqualified:
                continue
            p = {"bot_id": b.bot_id, "return": s.avg_return, "max_dd": s.max_dd,
                 "consistency": s.eras_profitable / max(s.n_eras, 1), "sharpe": s.sharpe,
                 "genome": b.genome.to_json()}

            def dominates(x, y):
                ge = x["return"] >= y["return"] and x["max_dd"] <= y["max_dd"] and x["consistency"] >= y["consistency"]
                gt = x["return"] > y["return"] or x["max_dd"] < y["max_dd"] or x["consistency"] > y["consistency"]
                return ge and gt

            if any(dominates(q, p) for q in self.pareto):
                continue
            self.pareto = [q for q in self.pareto if not dominates(p, q)] + [p]
        if len(self.pareto) > 40:
            self.pareto = sorted(self.pareto, key=lambda q: q["sharpe"], reverse=True)[:40]

    # ------------------------------------------------------------------ #
    # Final: the untouched test
    # ------------------------------------------------------------------ #
    def build_ensemble(self) -> Ensemble | None:
        members = [Member(h.bot_id, h.genome, h.weights, h.val_sharpe) for h in self.hall[:config.LAB_ENSEMBLE_SIZE]]
        return Ensemble(members) if members else None

    def finalize(self) -> LabSummary:
        champion = self.hall[0] if self.hall else None
        ensemble = self.build_ensemble()
        showdown: list[Showdown] = []
        window = self.test_window or self.val_window
        if champion is not None:
            showdown.append(self._show(f"Champion (Bot #{champion.bot_id})", self.scorer.simulate(champion.genome, *window)))
        if ensemble is not None and len(ensemble.members) > 1:
            showdown.append(self._show(f"Ensemble vote ({len(ensemble.members)} bots)", ensemble.simulate(self.scorer, *window)))
        showdown.append(self._show("Textbook origin bot", self.scorer.simulate(LabGenome.textbook(), *window)))
        showdown.append(self._buy_and_hold(*window))

        dsr = float("nan")
        if champion is not None:
            train_daily = self.scorer.simulate(champion.genome, *self.train_window).daily_returns
            dsr = deflated_sharpe(train_daily, self.scorer.trial_sharpes.n, self.scorer.trial_sharpes.variance)
        summary = LabSummary(self.generations_run, self.island_fitness, list(self.hall), champion, ensemble, showdown,
                             dsr, self.scorer.trial_sharpes.n, list(self.pareto), self.window_dates(),
                             data_source=self.data.source)
        if self.save:
            summary.checkpoint = self._save_latest(summary)
            summary.chart = self._chart(summary)
        return summary

    def _show(self, name: str, r: SimResult) -> Showdown:
        daily = r.daily_returns
        from lab.fitness import _sharpe
        return Showdown(name, r, r.total_return, _sharpe(daily), r.max_drawdown, r.n_trades, r.win_rate,
                        r.profit_factor, daily, np.r_[r.capital, r.daily_equity], monte_carlo(daily))

    def _buy_and_hold(self, t0: int, t1: int) -> Showdown:
        d = self.data
        cap = config.INITIAL_BALANCE
        shares = cap / d.n_symbols / d.open[:, t0]
        day_end = np.flatnonzero(d.last_bar_of_day[t0:t1]) + t0
        eq = (shares[:, None] * d.close[:, day_end]).sum(axis=0)
        daily = np.r_[cap, eq][1:] / np.r_[cap, eq][:-1] - 1
        from lab.fitness import _sharpe
        peak = np.maximum.accumulate(np.r_[cap, eq])
        return Showdown(f"Buy & hold (all {d.n_symbols})", None, float(eq[-1] / cap - 1), _sharpe(daily),
                        float(np.max(1 - np.r_[cap, eq] / peak)), d.n_symbols, float("nan"), float("nan"),
                        daily, np.r_[cap, eq], monte_carlo(daily))

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def _state(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA, "run_id": self.run_id, "generation": self.generation, "next_bot_id": self._next_id,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "islands": [{"idx": i.idx, "bot_id": i.alpha.bot_id, "genome": i.alpha.genome.to_json(),
                         "mutation_rate": i.mutation_rate, "reign": i.reign, "born": i.alpha.born}
                        for i in self.islands],
            "hall": [h.to_json() for h in self.hall],
            "universe": self.data.symbols, "windows": self.window_dates(), "data_source": self.data.source,
            "trials": {"n": self.scorer.trial_sharpes.n, "mean": self.scorer.trial_sharpes.mean,
                       "m2": self.scorer.trial_sharpes.m2},
        }

    def _checkpoint(self, gen: int) -> None:
        if self.latest_name != "lab_latest.json":
            return  # re-evolution runs only save their final challenger
        _atomic_json(self.out_dir / f"lab_gen_{gen}.json", self._state())

    def _save_latest(self, summary: LabSummary) -> str:
        state = self._state()
        ens = summary.ensemble
        oos = next((s for s in summary.showdown if s.name.startswith("Ensemble")), None) or \
            next((s for s in summary.showdown if s.name.startswith("Champion")), None)
        state["deploy"] = {
            "ensemble": ens.to_json() if ens else None,
            "champion": summary.champion.to_json() if summary.champion else None,
            "quantiles": self.fs.quantiles.tolist(),
            "feature_names": self.fs.names,
            "drift_five_day_p5": oos.monte_carlo.five_day_p5 if oos and oos.monte_carlo else None,
            "test_return": oos.total_return if oos else None,
        }
        state["showdown"] = [{"name": s.name, "return": s.total_return, "sharpe": s.sharpe, "max_dd": s.max_dd,
                              "trades": s.trades} for s in summary.showdown]
        state["deflated_sharpe"] = summary.deflated_sharpe
        state["pareto"] = summary.pareto
        path = self.out_dir / self.latest_name
        _atomic_json(path, state)
        return _label(path)

    def _chart(self, summary: LabSummary) -> str | None:
        from utils.charts import render_lab_report
        out = render_lab_report(summary, self.data, self.out_dir / "lab_report.png")
        return _label(out) if out else None


def load_lab_seed(path: Path = config.LAB_DIR / "lab_latest.json") -> LabSeed:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvolutionIntegrityError(f"Cannot read Lab checkpoint {path}: {exc}") from exc
    return LabSeed(data["islands"], data.get("hall", []), int(data["generation"]), int(data["next_bot_id"]))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, path)


def _label(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(config.BASE_DIR)).replace("\\", "/")
    except ValueError:
        return str(path)
