"""Prophecy League invariants - offline, synthetic daily data."""

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

import lab.prophecy as P  # noqa: E402
from lab import gp  # noqa: E402
from lab.forecast import DAILY_BINARY, DAILY_NAMES, PLAIN, _synthetic_daily, build_daily_features  # noqa: E402
from utils.lab_display import LabDisplay  # noqa: E402


class ProphecyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _synthetic_daily([f"S{i}" for i in range(10)], seed=5)
        cls.fs, cls.ans = build_daily_features(cls.data)

    def test_judge_keeps_one_survivor_and_breeds_five(self) -> None:
        rng = random.Random(0)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            lg = P.found_league("volatility", rng, "2025-01-01")
            for i, p in enumerate(lg.prophets):
                p.correct, p.total = 10 + i, 20
            del p
            best = lg.prophets[-1].bot_id
            rec = P.judge_round(lg, rng, "2025-01-08", "test")
        self.assertEqual(rec["survivor"], best)
        self.assertEqual(rec["purged"], P.LEAGUE_SIZE - 1)
        self.assertEqual(len(lg.prophets), P.LEAGUE_SIZE)
        self.assertEqual(sum(p.role == "ALPHA" for p in lg.prophets), 1)
        self.assertTrue(all(p.parent == best for p in lg.prophets[1:]))

    def test_ties_go_to_the_alpha(self) -> None:
        rng = random.Random(1)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            lg = P.found_league("direction", rng, "2025-01-01")
            for p in lg.prophets:
                p.correct, p.total = 10, 20
            del p  # a lingering loop variable would (rightly) fail the purge check
            alpha = lg.alpha.bot_id
            rec = P.judge_round(lg, rng, "2025-01-08", "test")
        self.assertEqual(rec["survivor"], alpha)
        self.assertFalse(rec["dethroned"])

    def test_replay_runs_rounds_out_of_sample(self) -> None:
        s = P.replay(self.fs, self.ans, self.data.index, "volatility", LabDisplay(delay=0, quiet=True),
                     days=60, seed=2, save_chart=False)
        self.assertEqual(len(s.league.history), 60 // P.ROUND_DAYS)
        self.assertEqual(len(s.days), 60)
        self.assertTrue(0 <= s.alpha_accuracy <= 1)

    def test_live_state_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_file, old_drop = P.STATE_FILE, P._drop_unfinished_today
            P.STATE_FILE = Path(tmp) / "state.json"
            P._drop_unfinished_today = lambda index: len(index) - 1
            try:
                T = self.data.n_days
                for k in (3, 2):
                    end = T - k
                    fs = build_daily_features(self.data)[0]
                    fs.values = fs.values[:, :, :end]
                    ans = P.Answers(self.ans.up[:, :end], self.ans.big[:, :end], self.ans.valid[:, :end].copy(),
                                    self.ans.next_ret[:, :end].copy())
                    ans.valid[:, -1] = False
                    ans.next_ret[:, -1] = np.nan
                    st = P.live_step(fs, ans, self.data.index[:end], self.data.symbols,
                                     LabDisplay(delay=0, quiet=True), seed=1)
                self.assertEqual(list(st["pending"]), [str(self.data.index[T - 3].date())])
                self.assertEqual(st["leagues"]["direction"]["round_days"], 1)
            finally:
                P.STATE_FILE, P._drop_unfinished_today = old_file, old_drop


if __name__ == "__main__":
    unittest.main()
