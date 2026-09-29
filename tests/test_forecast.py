"""Forecaster Arena invariants - offline, synthetic daily data."""

from __future__ import annotations

import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TIMEFRAME", "1d")

import numpy as np  # noqa: E402

from lab import gp  # noqa: E402
from lab.forecast import (  # noqa: E402
    DAILY_BINARY, DAILY_NAMES, PLAIN, ForecastEngine, ForecastScorer, _synthetic_daily, build_daily_features, textbook_tree,
)
from utils.lab_display import LabDisplay  # noqa: E402


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _synthetic_daily([f"S{i}" for i in range(12)], seed=3)
        cls.fs, cls.ans = build_daily_features(cls.data)

    def test_daily_features_ignore_the_future(self) -> None:
        data = _synthetic_daily([f"S{i}" for i in range(6)], seed=4)
        cut = data.n_days - 200
        before, _ = build_daily_features(data)
        for arr in (data.open, data.high, data.low, data.close, data.volume):
            arr[:, cut + 1:] *= 1.37
        data.spy.iloc[cut + 1:] *= 0.8
        data.vix.iloc[cut + 1:] *= 2
        after, _ = build_daily_features(data)
        for i, name in enumerate(DAILY_NAMES):
            np.testing.assert_allclose(after.values[i, :, :cut + 1], before.values[i, :, :cut + 1], rtol=1e-5,
                                       atol=1e-7, equal_nan=True, err_msg=f"{name} peeked at the future")

    def test_earnings_flag_the_move_they_affect(self) -> None:
        import pandas as pd

        from lab.daily_extra import earnings_features

        idx = pd.bdate_range("2026-01-05", periods=10)
        ny = "America/New_York"
        flags, to, since = earnings_features([pd.Timestamp("2026-01-07 16:05", tz=ny),    # after close
                                              pd.Timestamp("2026-01-13 07:00", tz=ny)], idx)  # before open
        self.assertEqual([str(d.date()) for d in idx[flags > 0]], ["2026-01-07", "2026-01-12"])
        self.assertEqual(to[0], 2)       # Monday: 2 trading days until the Wednesday flag
        self.assertEqual(since[3], 1)    # Thursday: 1 day after

    def test_answers_are_tomorrow(self) -> None:
        c = self.data.close
        np.testing.assert_allclose(self.ans.next_ret[:, :-1], c[:, 1:] / c[:, :-1] - 1, rtol=1e-12)
        self.assertTrue(np.all(np.isnan(self.ans.next_ret[:, -1])))

    def test_scoring_against_naive_baseline(self) -> None:
        self.fs.fit_quantiles(260, 1500)
        sc = ForecastScorer(self.fs, self.ans, "direction")
        acc, base, n, _ = sc.window_stats(np.ones_like(sc.label[:, 300:900]), 300, 900)
        self.assertAlmostEqual(acc, sc.label[:, 300:900][sc.valid[:, 300:900]].mean())   # "always UP"
        self.assertGreaterEqual(base, 0.5)
        self.assertGreater(n, 0)
        p = sc.luck_test(np.ones_like(sc.label[:, 300:900]), 300, 900, runs=50)
        self.assertTrue(0 < p <= 1)

    def test_engine_runs_and_purges(self) -> None:
        for target in ("direction", "volatility"):
            eng = ForecastEngine(self.fs, self.ans, self.data.index, target, LabDisplay(delay=0, quiet=True),
                                 rng=random.Random(1), n_islands=2, save=False)
            s = eng.run(3)
            self.assertEqual(s.generations, [1, 2, 3])
            self.assertTrue(s.hall)
            self.assertTrue(any(r.name.startswith("Coin") for r in s.results))
            with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
                self.assertTrue(gp.describe(textbook_tree(target)))


if __name__ == "__main__":
    unittest.main()
