"""
Evolution Lab (v2) invariants - offline, synthetic data.

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TIMEFRAME", "1d")

import numpy as np  # noqa: E402

import config  # noqa: E402
from lab import gp  # noqa: E402
from lab.data import synthetic_lab_data, shuffle_days  # noqa: E402
from lab.ensemble import Ensemble, Member  # noqa: E402
from lab.evolution import LabEngine, load_lab_seed  # noqa: E402
from lab.features import FEATURE_NAMES, build_features  # noqa: E402
from lab.fitness import Scorer, deflated_sharpe, monte_carlo  # noqa: E402
from lab.genome import PARAM_SPECS, LabGenome  # noqa: E402
from utils.lab_display import LabDisplay  # noqa: E402

SYMS = ["AAA", "BBB", "CCC", "DDD"]


def make(days: int = 150, seed: int = 1):
    data = synthetic_lab_data(SYMS, days=days, seed=seed)
    fs = build_features(data)
    fs.fit_quantiles(0, data.n_bars // 2)
    return data, fs


class RuleTreeTests(unittest.TestCase):
    def test_variation_respects_depth_and_round_trips(self) -> None:
        rng = random.Random(0)
        for _ in range(300):
            a = gp.random_tree(rng, config.LAB_MAX_TREE_DEPTH)
            b = gp.random_tree(rng, config.LAB_MAX_TREE_DEPTH)
            for t in (gp.mutate(a, rng, 0.4, config.LAB_MAX_TREE_DEPTH),
                      gp.crossover(a, b, rng, config.LAB_MAX_TREE_DEPTH)):
                self.assertLessEqual(gp.depth(t), config.LAB_MAX_TREE_DEPTH)
                self.assertEqual(gp.key(gp.from_json(gp.to_json(t))), gp.key(t))
                self.assertTrue(gp.describe(t))

    def test_tree_evaluation_logic(self) -> None:
        data, fs = make()
        ev = gp.TreeEvaluator(fs)
        a, b = gp.cond("rsi_14", False, 0.3), gp.cond("vwap_dev", True, 0.5)
        np.testing.assert_array_equal(ev(gp.And(a, b)), ev(a) & ev(b))
        np.testing.assert_array_equal(ev(gp.Or(a, b)), ev(a) | ev(b))
        np.testing.assert_array_equal(ev(gp.Not(a)), ~ev(a))


class GenomeTests(unittest.TestCase):
    def test_mutation_and_crossover_stay_valid(self) -> None:
        rng = random.Random(1)
        g = LabGenome.textbook()
        for _ in range(300):
            g = LabGenome.crossover(g.mutate(rng, 0.5), LabGenome.random(rng), rng)
            for k, (lo, hi, _, _) in PARAM_SPECS.items():
                self.assertTrue(lo <= g.params[k] <= hi, k)
            self.assertLessEqual(g.params["start_min"] + 30, g.params["end_min"])
            self.assertTrue(g.long_entry is not None or g.short_entry is not None)
            self.assertEqual(LabGenome.from_json(g.to_json()).key(), g.key())


class NoLookAheadTests(unittest.TestCase):
    def test_features_ignore_the_future(self) -> None:
        """Scramble everything after bar `cut`: no feature at or before `cut` may change."""
        data = synthetic_lab_data(SYMS, days=120, seed=3)
        cut = data.n_bars - 400
        before = build_features(data).values[:, :, :cut + 1]
        rng = np.random.default_rng(9)
        for arr in (data.open, data.high, data.low, data.close, data.volume):
            arr[:, cut + 1:] *= rng.uniform(0.5, 1.5, arr[:, cut + 1:].shape)
        for df in data.context.values():
            df.iloc[cut + 1:] = df.iloc[cut + 1:] * 1.3
        after = build_features(data).values[:, :, :cut + 1]
        for i, name in enumerate(FEATURE_NAMES):
            np.testing.assert_allclose(after[i], before[i], rtol=1e-5, atol=1e-7, equal_nan=True,
                                       err_msg=f"feature {name} peeked at the future")


class SimulatorTests(unittest.TestCase):
    def test_accounting_and_no_overnight_positions(self) -> None:
        data, fs = make()
        rng = random.Random(4)
        sc = Scorer(data, fs)
        for _ in range(15):
            g = LabGenome.random(rng)
            r = sc.simulate(g, 500, data.n_bars)
            self.assertAlmostEqual(r.equity[-1] - r.capital, r.trades_pnl.sum(), places=5)
            if r.n_trades:
                np.testing.assert_array_equal(data.day_id[r.trades_entry], data.day_id[r.trades_exit])
                self.assertTrue(np.all(r.trades_exit >= r.trades_entry))

    def test_shorts_happen_when_short_rule_exists(self) -> None:
        data, fs = make()
        g = LabGenome(None, gp.cond("rsi_14", True, 0.7), gp.cond("rsi_14", False, 0.4),
                      {k: v[2] for k, v in PARAM_SPECS.items()}, {})._repaired()
        r = Scorer(data, fs).simulate(g, 500, data.n_bars)
        self.assertGreater(r.n_trades, 0)
        self.assertTrue(np.all(r.trades_dir == -1))


class StatsTests(unittest.TestCase):
    def test_deflated_sharpe_penalises_many_trials(self) -> None:
        daily = np.random.default_rng(0).normal(0.001, 0.01, 250)
        self.assertGreater(deflated_sharpe(daily, 10, 0.0004), deflated_sharpe(daily, 10_000, 0.0004))

    def test_monte_carlo_ordering(self) -> None:
        mc = monte_carlo(np.random.default_rng(1).normal(0.0005, 0.01, 200))
        self.assertLess(mc.p5, mc.p50)
        self.assertLess(mc.p50, mc.p95)
        self.assertTrue(0 <= mc.prob_profit <= 1)


class EnsembleTests(unittest.TestCase):
    def test_single_member_ensemble_equals_member(self) -> None:
        data, fs = make()
        sc = Scorer(data, fs)
        g = LabGenome.textbook()
        solo = Ensemble([Member(1, g, (1.0, 1.0, 1.0))], quorum=0.5)
        ev = sc.evaluator(500, data.n_bars)
        long_s, _, exit_s = solo.vote([(ev(g.long_entry), ev(g.short_entry), ev(g.exit_rule))],
                                      sc.bar_regime[500:data.n_bars])
        np.testing.assert_array_equal(long_s, ev(g.long_entry))
        np.testing.assert_array_equal(exit_s, ev(g.exit_rule))


class MonkeyTests(unittest.TestCase):
    def test_shuffle_keeps_shape_and_total_move(self) -> None:
        data, _ = make()
        m = shuffle_days(data, np.random.default_rng(0))
        self.assertEqual(m.close.shape, data.close.shape)
        np.testing.assert_allclose(m.close[:, -1], data.close[:, -1], rtol=1e-9)   # same moves, new order
        self.assertFalse(np.allclose(m.close, data.close))


class LabEngineTests(unittest.TestCase):
    def test_islands_purge_hall_and_resume(self) -> None:
        data = synthetic_lab_data(SYMS, days=200, seed=5)
        fs = build_features(data)
        with tempfile.TemporaryDirectory() as tmp:
            eng = LabEngine(data, fs, LabDisplay(delay=0, quiet=True), rng=random.Random(2), n_islands=3,
                            out_dir=Path(tmp))
            s = eng.run(3)
            self.assertEqual(s.generations, [1, 2, 3])
            self.assertEqual(len(eng.islands), 3)
            self.assertTrue((Path(tmp) / "lab_latest.json").is_file())
            self.assertTrue((Path(tmp) / "lab_gen_3.json").is_file())
            self.assertTrue(any(sd.name.startswith("Buy & hold") for sd in s.showdown))
            seed = load_lab_seed(Path(tmp) / "lab_latest.json")
            eng2 = LabEngine(data, fs, LabDisplay(delay=0, quiet=True), rng=random.Random(3), n_islands=3,
                             out_dir=Path(tmp))
            s2 = eng2.run(1, seed)
            self.assertEqual(s2.generations, [4])
            ids = {i.alpha.bot_id for i in eng2.islands}
            self.assertEqual(len(ids), 3)


if __name__ == "__main__":
    unittest.main()
