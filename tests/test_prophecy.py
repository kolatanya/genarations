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

    def test_generations_from_a_fresh_start(self) -> None:
        from lab.forecast import textbook_tree

        gens = 20
        s = P.replay(self.fs, self.ans, self.data.index, "direction", LabDisplay(delay=0, quiet=True),
                     days=gens * P.ROUND_DAYS, seed=3, save_chart=False, fresh=True)
        self.assertEqual(len(s.league.history), gens)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            founder = s.league.history[0]["leaderboard"]
            self.assertEqual(len(founder), P.LEAGUE_SIZE)
            self.assertEqual(gp.key(gp.from_json(P.found_league("direction", random.Random(0), "x", fresh=True)
                                                 .alpha.tree)), gp.key(textbook_tree("direction")))
        eras = s.eras()
        self.assertEqual(len(eras), 5)
        self.assertEqual(eras[0]["first_gen"], 1)
        self.assertEqual(eras[-1]["last_gen"], gens)
        pooled = sum(r["alpha"] for r in eras) / 5
        self.assertAlmostEqual(pooled, s.alpha_accuracy, delta=0.02)
        self.assertGreater(P.max_generations(self.data.n_days), 100)

    def test_every_call_has_a_reason_that_reproduces_it(self) -> None:
        from lab.explain import playbook, reason

        self.fs.fit_quantiles(260, 1200)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            a, b = gp.cond("move_size", True, 0.7), gp.cond("rsi_2", False, 0.2)
            self.assertEqual(gp.key(gp.simplify(gp.Or(gp.Or(a, b), a))), gp.key(gp.Or(a, b)))
            self.assertEqual(gp.key(gp.simplify(gp.Not(gp.Not(a)))), gp.key(a))
            tree = gp.Or(gp.And(a, gp.Not(b)), gp.cond("vol_z", True, 0.9))
            t = 1500
            calls = gp.TreeEvaluator(self.fs, t, t + 1)(tree)[:, 0]
            for s in range(self.fs.values.shape[1]):
                hit, why = reason(tree, self.fs, s, t)
                self.assertEqual(hit, bool(calls[s]))            # the reason gives exactly the call
                self.assertTrue(why)
                self.assertTrue(all(("✓" in w) == hit for w in why) or not hit)
            pb = playbook(tree, self.fs, self.ans.big, self.ans.valid, t)
            self.assertEqual(len(pb["leaves"]), 3)
            self.assertTrue(0 <= pb["accuracy"] <= 1)

    def test_lucky_week_cannot_steal_the_crown(self) -> None:
        rng = random.Random(4)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            lg = P.found_league("volatility", rng, "2025-01-01")
            alpha, challenger = lg.prophets[0].bot_id, lg.prophets[1].bot_id
            scores = {p.bot_id: 0.50 for p in lg.prophets}
            scores[challenger] = 0.50 + P.crown_margin_points() / 2           # better, but not by enough
            rec = P.judge_round(lg, rng, "2025-01-08", "t", scores=scores, judge_days=P.JUDGE_DAYS)
            self.assertEqual(rec["survivor"], alpha)
            alpha, challenger = lg.alpha.bot_id, lg.prophets[1].bot_id
            scores = {p.bot_id: 0.50 for p in lg.prophets}
            scores[challenger] = 0.50 + P.crown_margin_points() * 2           # clearly better
            rec = P.judge_round(lg, rng, "2025-01-15", "t", scores=scores, judge_days=P.JUDGE_DAYS)
            self.assertEqual(rec["survivor"], challenger)

    def test_ancestors_are_kept_and_can_return(self) -> None:
        rng = random.Random(6)
        self.fs.fit_quantiles(260, 1200)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            lg = P.found_league("volatility", rng, "2025-01-01")
            lg.alpha.tree = gp.to_json(gp.cond("move_size", True, 0.5))   # a rule with a real reason
            old_alpha = lg.alpha
            old_id, old_tree = old_alpha.bot_id, old_alpha.tree
            del old_alpha
            scores = {p.bot_id: 0.0 for p in lg.prophets}
            scores[lg.prophets[1].bot_id] = 5.0                      # the Alpha is dethroned...
            scores[lg.prophets[2].bot_id] = 4.0                      # (runner-up isn't the old Alpha)
            P.judge_round(lg, rng, "2025-01-08", "t", scores=scores, judge_days=P.JUDGE_DAYS)
            self.assertEqual([h["bot_id"] for h in lg.hall], [old_id])   # ...and enshrined
            ghost = P.summon_ancestor(lg, self.fs, self.ans, 1500)
            self.assertIsNotNone(ghost)
            self.assertEqual((ghost.bot_id, ghost.tree, ghost.role), (old_id, old_tree, "ANCESTOR"))
            scores = {p.bot_id: 0.0 for p in lg.prophets}
            scores[old_id] = 9.0                                     # the ancestor fits the market best
            del ghost
            rec = P.judge_round(lg, rng, "2025-01-15", "t", scores=scores, judge_days=P.JUDGE_DAYS)
            self.assertEqual((rec["survivor"], rec["survivor_role"]), (old_id, "ANCESTOR"))
            self.assertEqual(lg.alpha.bot_id, old_id)
            self.assertEqual(len(lg.prophets), P.LEAGUE_SIZE)

    def test_confidence_matches_the_rule(self) -> None:
        from lab.explain import ConfidenceTable

        self.fs.fit_quantiles(260, 1200)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            tree = gp.Or(gp.cond("move_size", True, 0.7), gp.cond("earn_expected", True, 0.97))
            table = ConfidenceTable(tree, self.fs, self.ans.big, self.ans.valid, 1500)
            calls, conf = table.calls(self.fs, 1500)
            np.testing.assert_array_equal(calls, gp.TreeEvaluator(self.fs, 1500, 1501)(tree)[:, 0])
            self.assertTrue(np.all((conf >= 0) & (conf <= 1)))

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
