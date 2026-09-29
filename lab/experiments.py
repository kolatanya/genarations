"""
The two most honest experiments you can run on an evolved trading bot.

MONKEY TEST (luck detector)
    Run the exact same evolution on "monkey markets": every stock's trading
    days shuffled independently, so real day-to-day and stock-to-stock
    patterns no longer exist. If bots evolved on monkey markets score as well
    on the final test as bots evolved on the real market, the real results
    were luck.

WALK-FORWARD (how the bot would really be used)
    Evolve on a stretch of the past, trade the NEXT month with the winner,
    slide forward a month, repeat. Every traded month is genuinely unseen.
"""

from __future__ import annotations

import random
from typing import Callable

import numpy as np

import config
from lab.data import LabData, shuffle_days
from lab.evolution import LabEngine
from lab.features import build_features
from lab.fitness import _sharpe
from utils.lab_display import LabDisplay


def _quiet() -> LabDisplay:
    return LabDisplay(delay=0, quiet=True)


def _headline(summary) -> dict:
    best = summary.showdown[0] if summary.showdown else None
    return {"return": best.total_return if best else 0.0, "sharpe": best.sharpe if best else 0.0,
            "val": summary.champion.val_sharpe if summary.champion else float("nan")}


def monkey_test(data: LabData, generations: int, runs: int = config.LAB_MONKEY_RUNS, seed: int = 0,
                say: Callable[[str], None] = lambda m: None) -> tuple[dict, list[dict]]:
    """Real-market evolution vs `runs` monkey-market evolutions with the same budget."""
    say("Evolving on the REAL market...")
    fs = build_features(data)
    real = LabEngine(data, fs, _quiet(), rng=random.Random(seed), save=False).run(generations)
    monkeys = []
    for i in range(runs):
        say(f"Evolving on MONKEY market {i + 1}/{runs} (shuffled days)...")
        m_data = shuffle_days(data, np.random.default_rng(seed + 100 + i))
        m_fs = build_features(m_data)
        s = LabEngine(m_data, m_fs, _quiet(), rng=random.Random(seed), save=False).run(generations)
        monkeys.append(_headline(s))
    return _headline(real), monkeys


def walk_forward(data: LabData, generations: int = 10, train_days: int = 90, val_days: int = 20,
                 trade_days: int = 21, seed: int = 0,
                 say: Callable[[str], None] = lambda m: None) -> tuple[list[dict], dict, dict]:
    """Roll forward: evolve on [train+val], trade the next `trade_days` with the champion."""
    fs = build_features(data)
    n_days = len(data.day_bounds())
    start = 25
    steps: list[dict] = []
    all_daily: list[np.ndarray] = []
    bench_daily: list[np.ndarray] = []
    cap = config.INITIAL_BALANCE
    while start + train_days + val_days + trade_days <= n_days:
        a, b = start, start + train_days
        c, d = b + val_days, b + val_days + trade_days
        say(f"Walk-forward step {len(steps) + 1}: evolving on {data.unique_days[a]} → {data.unique_days[c - 1]}...")
        eng = LabEngine(data, fs, _quiet(), rng=random.Random(seed + start), save=False,
                        n_islands=2, day_ranges={"train": (a, b), "validation": (b, c), "test": (c, d)},
                        n_folds=3, folds_per_gen=2)
        s = eng.run(generations)
        champ = s.showdown[0]
        bh = s.showdown[-1]
        steps.append({"train": f"{data.unique_days[a]} → {data.unique_days[c - 1]}",
                      "trade": f"{data.unique_days[c]} → {data.unique_days[d - 1]}",
                      "bot": champ.name.replace("Champion (", "").rstrip(")"), "return": champ.total_return,
                      "trades": champ.trades, "bench": bh.total_return})
        all_daily.append(champ.daily)
        bench_daily.append(bh.daily)
        start += trade_days

    def summarise(parts: list[np.ndarray]) -> dict:
        if not parts:
            return {"return": 0.0, "sharpe": 0.0, "max_dd": 0.0}
        daily = np.concatenate(parts)
        eq = cap * np.cumprod(1 + daily)
        peak = np.maximum.accumulate(np.r_[cap, eq])
        return {"return": float(eq[-1] / cap - 1), "sharpe": _sharpe(daily),
                "max_dd": float(np.max(1 - np.r_[cap, eq] / peak))}

    return steps, summarise(all_daily), summarise(bench_daily)
