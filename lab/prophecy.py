"""
PROPHECY LEAGUE - bots predict the real market; the most accurate live and reproduce.

Two leagues run side by side, each with 6 prophets (1 Alpha + 5 offspring):
  DIRECTION  "Will this stock close UP tomorrow?"          (for ~100 stocks)
  VOLATILITY "Will tomorrow's move be BIGGER than usual?"

Every trading day each prophet makes one call per stock for TOMORROW.
The next day reality reveals who was right. Every ROUND_DAYS trading days the
prophet with the most correct calls in that round SURVIVES; the other five are
purged, and the survivor has five children (four mutants and one crossover with
the runner-up). Ties go to the reigning Alpha.

Two ways to run it:
  LIVE    - run once each evening after the US close: yesterday's calls are
            scored against what really happened, then tomorrow's calls are made.
            Nobody can peek at the future, so every score is genuine.
  REPLAY  - the same tournament stepped day by day through the last year of
            history. At every step prophets only use information available that
            day, so the reigning Alpha's calls are genuinely out-of-sample.

The honest scoreboard is the REIGNING ALPHA's accuracy (it was crowned before
the days it's judged on) versus the naive guess (always the more common answer).
"""

from __future__ import annotations

import gc
import json
import os
import random
import weakref
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from engine.evaluator import EvolutionIntegrityError
from lab import gp
from lab.features import FeatureSet
from lab.forecast import DAILY_BINARY, DAILY_NAMES, PLAIN, TARGETS, WARMUP_DAYS, Answers, textbook_tree

ROUND_DAYS = 5
LEAGUE_SIZE = 6
STATE_FILE = config.LAB_DIR / "prophecy_live.json"
MAX_DEPTH = config.LAB_MAX_TREE_DEPTH


@dataclass
class Prophet:
    bot_id: int
    tree: dict                  # rule tree as JSON
    parent: int | None
    born: str
    role: str = "CHILD"
    correct: int = 0            # this round
    total: int = 0
    life_correct: int = 0       # whole life
    life_total: int = 0
    note: str = ""              # Claude's reason for designing this child (empty for random offspring)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def life_accuracy(self) -> float:
        return self.life_correct / self.life_total if self.life_total else 0.0


@dataclass
class League:
    target: str
    prophets: list[Prophet]
    next_id: int = 1
    round_no: int = 0
    round_days: int = 0
    rate: float = config.MUTATION_RATE
    history: list[dict] = field(default_factory=list)   # one entry per completed round
    alpha_correct: int = 0      # reigning Alpha's calls, all rounds (the honest score)
    alpha_total: int = 0
    naive_correct: int = 0
    naive_total: int = 0

    @property
    def alpha(self) -> Prophet:
        return next(p for p in self.prophets if p.role == "ALPHA")

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id - 1


# --------------------------------------------------------------------------- #
# Founding and reproduction
# --------------------------------------------------------------------------- #
def founding_tree(target: str) -> gp.Node:
    """Start from the Forecaster Arena champion if there is one, else the textbook rule."""
    path = config.LAB_DIR / f"forecast_{target}.json"
    try:
        hall = json.loads(path.read_text(encoding="utf-8")).get("hall") or []
        if hall:
            return gp.from_json(hall[0]["tree"])
    except (OSError, ValueError, KeyError):
        pass
    return textbook_tree(target)


def found_league(target: str, rng: random.Random, today: str, breeder=None) -> League:
    league = League(target, [])
    alpha_tree = founding_tree(target)
    league.prophets.append(Prophet(league.new_id(), gp.to_json(alpha_tree), None, today, "ALPHA"))
    designed = breeder.design(target, alpha_tree, [], []) if breeder else None
    league.prophets += _offspring(league, alpha_tree, alpha_tree, rng, today, designed)
    return league


def _offspring(league: League, alpha: gp.Node, runner_up: gp.Node, rng: random.Random, today: str,
               designed: list[tuple[gp.Node, str]] | None = None) -> list[Prophet]:
    """Five children: Claude's designs first (if any), then random mutants, and always one crossover.

    At least one random mutant and the crossover are kept, so Claude's children always race a blind control."""
    kids, seen = [], {gp.key(alpha)}
    parent = league.prophets[0].bot_id if league.prophets else None
    for tree, why in (designed or [])[:LEAGUE_SIZE - 3]:
        if gp.key(tree) in seen:
            continue
        seen.add(gp.key(tree))
        kids.append(Prophet(league.new_id(), gp.to_json(tree), parent, today, "CLAUDE", note=why))
    while len(kids) < LEAGUE_SIZE - 1:
        cross = len(kids) == LEAGUE_SIZE - 2
        tree = gp.crossover(alpha, runner_up, rng, MAX_DEPTH) if cross else gp.mutate(alpha, rng, league.rate, MAX_DEPTH)
        for _ in range(5):
            if gp.key(tree) not in seen:
                break
            tree = gp.mutate(tree, rng, max(league.rate, 0.2), MAX_DEPTH)
        seen.add(gp.key(tree))
        kids.append(Prophet(league.new_id(), gp.to_json(tree), parent, today, "CROSSOVER" if cross else "MUTANT"))
    return kids


def judge_round(league: League, rng: random.Random, today: str, dates: str, breeder=None) -> dict:
    """Most correct calls survives; the rest are purged; the survivor has five children."""
    alpha = league.alpha
    ranked = sorted(league.prophets, key=lambda p: (p.accuracy, p is alpha), reverse=True)
    survivor, runner_up = ranked[0], ranked[1]
    dethroned = survivor is not alpha
    record = {
        "round": league.round_no + 1, "dates": dates, "survivor": survivor.bot_id,
        "survivor_role": survivor.role, "survivor_accuracy": survivor.accuracy,
        "alpha_before": alpha.bot_id, "alpha_accuracy": alpha.accuracy, "dethroned": dethroned,
        "leaderboard": [(p.bot_id, p.role, p.accuracy, p.correct, p.total) for p in ranked],
        "rule": survivor.tree,
    }
    survivor_tree, runner_tree = gp.from_json(survivor.tree), gp.from_json(runner_up.tree)
    last_round = [{"accuracy": p.accuracy, "role": p.role, "rule": gp.describe(gp.from_json(p.tree), breeder.fs)}
                  for p in ranked] if breeder is not None else []   # what Claude sees, captured before the purge
    # EXTINCTION: everyone except the survivor is deleted - verified with weak references
    tombstones = [weakref.ref(p) for p in league.prophets if p is not survivor]
    league.prophets = [survivor]
    del ranked, runner_up, alpha
    gc.collect()
    lingering = sum(r() is not None for r in tombstones)
    if lingering:
        raise EvolutionIntegrityError(f"{lingering} purged prophets still in memory")
    record["purged"] = len(tombstones)

    league.rate = (max(config.MUTATION_RATE_MIN, league.rate * config.MUTATION_PROGRESS_DECAY) if dethroned
                   else min(config.MUTATION_RATE_MAX, league.rate * config.MUTATION_STUCK_BOOST))
    survivor.role = "ALPHA"
    survivor.correct = survivor.total = 0
    designed = breeder.design(league.target, survivor_tree, last_round, league.history) if breeder else None
    league.prophets += _offspring(league, survivor_tree, runner_tree, rng, today, designed)
    record["claude_children"] = sum(p.role == "CLAUDE" for p in league.prophets)
    for p in league.prophets[1:]:
        p.parent = survivor.bot_id
    league.round_no += 1
    league.round_days = 0
    league.history.append(record)
    return record


def score_day(league: League, calls: dict[int, np.ndarray], label: np.ndarray, valid: np.ndarray,
              naive_call: bool) -> dict[int, float]:
    """Add one resolved day of calls to every prophet's tally. Returns each prophet's accuracy that day."""
    out = {}
    n = int(valid.sum())
    for p in league.prophets:
        pred = calls.get(p.bot_id)
        if pred is None or n == 0:
            continue
        c = int((pred[valid] == label[valid]).sum())
        p.correct += c
        p.total += n
        p.life_correct += c
        p.life_total += n
        out[p.bot_id] = c / n
        if p.role == "ALPHA":
            league.alpha_correct += c
            league.alpha_total += n
    league.naive_correct += int((label[valid] == naive_call).sum())
    league.naive_total += n
    league.round_days += 1
    return out


def league_calls(league: League, fs: FeatureSet, t: int) -> dict[int, np.ndarray]:
    ev = gp.TreeEvaluator(fs, t, t + 1)
    return {p.bot_id: ev(gp.from_json(p.tree))[:, 0] for p in league.prophets}


def _labels(ans: Answers, target: str) -> tuple[np.ndarray, np.ndarray]:
    if target == "direction":
        return ans.up, np.isfinite(ans.next_ret)
    return ans.big, ans.valid


def naive_call_for(ans: Answers, target: str, t: int, lookback: int = 250) -> bool:
    """The naive guess: whichever answer was more common over the past year (known at day t)."""
    label, valid = _labels(ans, target)
    a = max(0, t - lookback)
    ok = valid[:, a:t]
    return bool(label[:, a:t][ok].mean() >= 0.5) if ok.any() else True


# --------------------------------------------------------------------------- #
# REPLAY - the tournament stepped through history, day by day
# --------------------------------------------------------------------------- #
@dataclass
class ReplaySummary:
    target: str
    league: League
    days: list[str]
    alpha_daily: list[float]          # reigning Alpha's accuracy each day
    naive_daily: list[float]
    alpha_accuracy: float
    naive_accuracy: float
    ci: tuple[float, float]           # 90% bootstrap interval of (Alpha - naive), percentage points
    chart: str | None = None


def replay(fs: FeatureSet, ans: Answers, index: pd.DatetimeIndex, target: str, display, *,
           days: int = 250, seed: int | None = None, save_chart: bool = True, breeder=None) -> ReplaySummary:
    rng = random.Random(seed)
    T = len(index)
    end = T - 1                                       # the last day has no "tomorrow" yet
    start = max(WARMUP_DAYS + 250, end - days)
    fs.fit_quantiles(WARMUP_DAYS, start)             # thresholds only from BEFORE the replay starts
    label, valid = _labels(ans, target)
    if breeder is not None:
        breeder.fs, breeder.ans, breeder.t = fs, ans, start
    with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
        league = found_league(target, rng, str(index[start].date()), breeder)
        display.prophecy_replay_intro(league, index[start].date(), index[end - 1].date(), days)
        alpha_daily, naive_daily, day_labels = [], [], []
        correct_matrix = []                          # per day: (alpha correct, naive correct, n) for the bootstrap
        round_start = start
        for t in range(start, end):
            calls = league_calls(league, fs, t)
            naive = naive_call_for(ans, target, t)
            alpha_id = league.alpha.bot_id
            day_acc = score_day(league, calls, label[:, t], valid[:, t], naive)
            n = int(valid[:, t].sum())
            if n:
                a_c = int((calls[alpha_id][valid[:, t]] == label[valid[:, t], t]).sum())
                n_c = int((label[valid[:, t], t] == naive).sum())
                correct_matrix.append((a_c, n_c, n))
                alpha_daily.append(day_acc.get(alpha_id, np.nan))
                naive_daily.append(n_c / n)
                day_labels.append(str(index[t].date()))
            if league.round_days >= ROUND_DAYS:
                if breeder is not None:
                    breeder.t = t + 1          # day t's answer is known at t+1's close, when the next calls are made
                rec = judge_round(league, rng, str(index[t].date()),
                                  f"{index[round_start].date()} → {index[t].date()}", breeder)
                display.prophecy_round(league, rec, naive_daily[-ROUND_DAYS:])
                round_start = t + 1
    cm = np.array(correct_matrix, dtype=float)
    diff_ci = _bootstrap_ci(cm, rng=np.random.default_rng(seed or 0))
    summary = ReplaySummary(target, league, day_labels, alpha_daily, naive_daily,
                            league.alpha_correct / max(league.alpha_total, 1),
                            league.naive_correct / max(league.naive_total, 1), diff_ci)
    if save_chart:
        from utils.charts import render_prophecy_report
        out = render_prophecy_report(summary, config.LAB_DIR / f"prophecy_replay_{target}.png")
        summary.chart = str(out.relative_to(config.BASE_DIR)).replace("\\", "/") if out else None
    return summary


def _bootstrap_ci(cm: np.ndarray, rng: np.random.Generator, runs: int = 2000) -> tuple[float, float]:
    """90% interval for (Alpha accuracy - naive accuracy), resampling whole DAYS (calls within a day move together)."""
    if len(cm) < 5:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(cm), size=(runs, len(cm)))
    a = cm[idx, 0].sum(axis=1) / cm[idx, 2].sum(axis=1)
    b = cm[idx, 1].sum(axis=1) / cm[idx, 2].sum(axis=1)
    d = (a - b) * 100
    return float(np.percentile(d, 5)), float(np.percentile(d, 95))


# --------------------------------------------------------------------------- #
# LIVE - one step per evening
# --------------------------------------------------------------------------- #
def _drop_unfinished_today(index: pd.DatetimeIndex) -> int:
    """Index of the last COMPLETED daily bar (today's bar is still forming until ~16:15 ET)."""
    now = pd.Timestamp.now(tz="America/New_York")
    last = len(index) - 1
    if index[last].date() == now.date() and now.hour * 60 + now.minute < 16 * 60 + 15:
        last -= 1
    return last


def load_state() -> dict | None:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1, default=float), encoding="utf-8")
    os.replace(tmp, STATE_FILE)


def _league_to_json(lg: League) -> dict:
    d = asdict(lg)
    return d


def _league_from_json(d: dict) -> League:
    prophets = [Prophet(**p) for p in d.pop("prophets")]
    return League(prophets=prophets, **d)


def live_step(fs: FeatureSet, ans: Answers, index: pd.DatetimeIndex, symbols: list[str], display,
              seed: int | None = None, breeder=None) -> dict:
    """Score every call whose answer is now known, run selections, then make calls for the next day."""
    rng = random.Random(seed)
    last = _drop_unfinished_today(index)
    today = str(index[last].date())
    state = load_state()
    if breeder is not None:
        breeder.fs, breeder.ans, breeder.t = fs, ans, last
    with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
        upgraded = state is not None and (state.get("symbols") != symbols or state.get("features") != DAILY_NAMES)
        if upgraded:
            # The stock list or the bots' senses changed: old calls can't be scored against the new setup.
            archive = STATE_FILE.with_name(f"prophecy_live_archived_{datetime.now():%Y%m%d_%H%M%S}.json")
            STATE_FILE.replace(archive)
            display.warn(f"League upgraded ({len(state.get('symbols', []))} -> {len(symbols)} stocks, "
                         f"{len(DAILY_NAMES)} senses). The previous league is archived as {archive.name}; "
                         "a new one is founded with the upgraded bots.")
            state = None
        if state is None:
            fs.fit_quantiles(WARMUP_DAYS, last - 60)
            state = {"founded": today, "symbols": symbols, "features": DAILY_NAMES,
                     "quantiles": fs.quantiles.tolist(), "pending": {},
                     "leagues": {t: _league_to_json(found_league(t, rng, today, breeder)) for t in TARGETS},
                     "log": []}
            display.prophecy_live_founded(today)
        fs.quantiles = np.array(state["quantiles"])
        leagues = {t: _league_from_json(dict(d)) for t, d in state["leagues"].items()}
        date_pos = {str(d.date()): i for i, d in enumerate(index)}
        reveals: list[dict] = []
        rounds: list[tuple[str, dict]] = []

        # 1) Reveal: score every pending call whose next trading day has closed
        for day in sorted(state["pending"]):
            i = date_pos.get(day)
            if i is None or i + 1 > last:
                continue                                      # tomorrow hasn't happened yet
            for target, lg in leagues.items():
                label, valid = _labels(ans, target)
                calls = {int(k): np.array(v, dtype=bool) for k, v in state["pending"][day].get(target, {}).items()}
                naive = naive_call_for(ans, target, i)
                accs = score_day(lg, calls, label[:, i], valid[:, i], naive)
                n = int(valid[:, i].sum())
                reveals.append({"target": target, "day": day, "result_day": str(index[i + 1].date()), "accs": accs,
                                "naive": int((label[valid[:, i], i] == naive).sum()) / n if n else float("nan"),
                                "alpha": lg.alpha.bot_id, "up_share": float(label[valid[:, i], i].mean()) if n else 0})
                if lg.round_days >= ROUND_DAYS:
                    rounds.append((target, judge_round(lg, rng, today, f"round ending {day}", breeder)))
            del state["pending"][day]

        # 2) Prophesy: calls for the next trading day, based on today's close
        calls_today = {}
        if today not in state["pending"]:
            calls_today = {t: {str(k): v.astype(bool).tolist() for k, v in league_calls(lg, fs, last).items()}
                           for t, lg in leagues.items()}
            state["pending"][today] = calls_today
        state["leagues"] = {t: _league_to_json(lg) for t, lg in leagues.items()}
        state["log"].append({"run_at": datetime.now().isoformat(timespec="seconds"), "data_through": today,
                             "revealed": len(reveals), "rounds": len(rounds)})
        save_state(state)
        display.prophecy_live(today, symbols, leagues, reveals, rounds, state["pending"].get(today, {}), fs)
    return state
