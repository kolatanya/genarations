"""Claude breeder - offline tests with a fake Claude (no network, no API key needed)."""

from __future__ import annotations

import json
import os
import random
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("TIMEFRAME", "1d")

import numpy as np  # noqa: E402

from lab import gp  # noqa: E402
from lab.claude_breeder import ClaudeBreeder, parse_children  # noqa: E402
from lab.forecast import DAILY_BINARY, DAILY_NAMES, PLAIN, _synthetic_daily, build_daily_features, textbook_tree  # noqa: E402
from lab.prophecy import LEAGUE_SIZE, found_league, replay  # noqa: E402
from utils.lab_display import LabDisplay  # noqa: E402

REPLY = json.dumps({"children": [
    {"why": "add a stock-specific filter", "rule": {"and": [{"f": "vol_ratio", "gt": True, "q": 0.7},
                                                            {"f": "move_size", "gt": True, "q": 0.6}]}},
    {"why": "made-up sense, must be dropped", "rule": {"f": "astrology", "gt": True, "q": 0.5}},
    {"why": "earnings only", "rule": {"f": "earn_next", "gt": True, "q": 0.9}},
    {"why": "three-way OR", "rule": {"or": [{"f": "gap", "gt": True, "q": 0.95}, {"f": "vol_z", "gt": True, "q": 0.9},
                                             {"f": "rsi_2", "gt": False, "q": 0.05}]}},
]})


class FakeClient:
    def __init__(self, reply: str | None = REPLY, fail: bool = False) -> None:
        self.reply, self.fail, self.prompts = reply, fail, []
        self.messages = self

    def create(self, **kw):
        if self.fail:
            raise ConnectionError("no internet")
        self.prompts.append(kw["messages"][0]["content"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Here you go:\n" + self.reply)])


class ClaudeBreederTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _synthetic_daily([f"S{i}" for i in range(10)], seed=5)
        cls.fs, cls.ans = build_daily_features(cls.data)
        cls.fs.fit_quantiles(260, 1200)

    def breeder(self, **kw) -> ClaudeBreeder:
        b = ClaudeBreeder(model="fake", children=3, client=FakeClient(**kw))
        b.fs, b.ans, b.t = self.fs, self.ans, 1500
        return b

    def test_reply_is_parsed_safely(self) -> None:
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            survivor = textbook_tree("volatility")
            kids = parse_children("junk " + REPLY + " junk", survivor, 3)
            self.assertEqual(len(kids), 3)                       # the made-up sense was dropped
            self.assertEqual(kids[0][1], "add a stock-specific filter")
            self.assertTrue(all(gp.depth(t) <= 4 for t, _ in kids))
            self.assertEqual(parse_children("no json here", survivor, 3), [])
            self.assertEqual(parse_children(json.dumps({"children": [{"rule": gp.to_json(survivor)}]}), survivor, 3), [])

    def test_claude_children_join_and_control_group_stays(self) -> None:
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            lg = found_league("volatility", random.Random(1), "2020-01-01", self.breeder())
        roles = [p.role for p in lg.prophets]
        self.assertEqual(len(roles), LEAGUE_SIZE)
        self.assertEqual(roles.count("CLAUDE"), 3)
        self.assertEqual(roles.count("MUTANT"), 1)
        self.assertEqual(roles.count("CROSSOVER"), 1)
        self.assertTrue(all(p.note for p in lg.prophets if p.role == "CLAUDE"))

    def test_failure_falls_back_to_random(self) -> None:
        b = self.breeder(fail=True)
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            lg = found_league("direction", random.Random(1), "2020-01-01", b)
        self.assertEqual(len(lg.prophets), LEAGUE_SIZE)
        self.assertNotIn("CLAUDE", [p.role for p in lg.prophets])
        self.assertEqual(b.failures, 1)

    def test_evidence_never_includes_unknown_answers(self) -> None:
        b = self.breeder()
        with gp.schema(DAILY_NAMES, DAILY_BINARY, PLAIN):
            tree = textbook_tree("volatility")
            before = b.prompt("volatility", tree, [], [])
            saved = self.ans.big.copy()
            try:
                self.ans.big[:, b.t:] = ~self.ans.big[:, b.t:]  # scramble every answer not yet known on day t
                after = b.prompt("volatility", tree, [], [])
            finally:
                self.ans.big[:] = saved
        self.assertEqual(before, after)

    def test_replay_with_claude_breeder(self) -> None:
        b = self.breeder()
        s = replay(self.fs, self.ans, self.data.index, "volatility", LabDisplay(delay=0, quiet=True),
                   days=30, seed=2, save_chart=False, breeder=b)
        self.assertGreaterEqual(len(s.league.history), 5)
        # the fake always sends the same 3 rules, so one gets dropped as a duplicate when it's the survivor
        self.assertTrue(all(h["claude_children"] >= 2 for h in s.league.history))
        self.assertGreater(b.calls, 5)
        self.assertIn("SENSES", b.client.prompts[-1])
        self.assertTrue(np.isfinite(s.alpha_accuracy))


if __name__ == "__main__":
    unittest.main()
