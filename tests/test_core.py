"""
Core invariants of the arena. Runs offline on synthetic prices:

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import io
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["TIMEFRAME"] = "1d"  # these tests use years of synthetic daily bars

from rich.console import Console  # noqa: E402

import config  # noqa: E402
from bot.genome import GENE_SPECS, GenomeError, TradingGenome  # noqa: E402
from bot.trader import BotRole, ExitReason, TradingBot, check_risk_exit  # noqa: E402
from broker.simulated_broker import SimulatedBroker  # noqa: E402
from dataclasses import replace  # noqa: E402

from engine.evaluator import (  # noqa: E402
    Evaluator,
    EvolutionIntegrityError,
    apply_quality,
    compute_fitness,
    frequency_multiplier,
    quality_multiplier,
)
from engine.evolution import EvolutionEngine, SeedAlpha, seed_from_checkpoint  # noqa: E402
from utils.display import Display  # noqa: E402

SYMBOLS = ["AAPL", "BTC-USD"]


def synthetic_market():
    broker = SimulatedBroker(force_synthetic=True, seed=7, use_cache=False)
    return broker.load_market_data(SYMBOLS, start="2020-01-01", end="2023-12-31")


def quiet_display() -> Display:
    return Display(console=Console(file=io.StringIO(), width=140), delay=0.0)


class GenomeTests(unittest.TestCase):
    def test_mutation_stays_in_bounds_and_valid(self) -> None:
        rng = random.Random(1)
        genome = TradingGenome.textbook()
        for _ in range(2000):
            genome = genome.mutate(0.5, rng)  # aggressive drift to hammer the boundaries
            for name, spec in GENE_SPECS.items():
                self.assertGreaterEqual(getattr(genome, name), spec.low)
                self.assertLessEqual(getattr(genome, name), spec.high)
            self.assertLess(genome.fast_sma, genome.slow_sma)
            self.assertLess(genome.macd_fast, genome.macd_slow)

    def test_mutate_returns_new_object_and_leaves_parent_untouched(self) -> None:
        parent = TradingGenome.textbook()
        child = parent.mutate(0.15, random.Random(3))
        self.assertIsNot(parent, child)
        self.assertEqual(parent, TradingGenome.textbook())
        self.assertTrue(parent.diff(child))

    def test_round_trip_and_validation(self) -> None:
        genome = TradingGenome.random(random.Random(5))
        self.assertEqual(TradingGenome.from_dict(genome.to_dict()), genome)
        with self.assertRaises(GenomeError):
            TradingGenome(fast_sma=50, slow_sma=40)
        with self.assertRaises(GenomeError):
            TradingGenome.from_dict({"rsi_period": 14})


class FitnessTests(unittest.TestCase):
    def test_losers_never_outrank_winners(self) -> None:
        winner = compute_fitness(net_profit_pct=2.0, sharpe=0.3, max_dd=0.2)
        double_negative = compute_fitness(net_profit_pct=-30.0, sharpe=-1.5, max_dd=0.4)
        self.assertGreater(winner, 0)
        self.assertLess(double_negative, 0, "negative PnL x negative Sharpe must not become positive")

    def test_deeper_drawdown_is_worse_for_losers(self) -> None:
        self.assertLess(compute_fitness(-10, 0.1, 0.5), compute_fitness(-10, 0.1, 0.1))

    def test_flat_bot_scores_zero(self) -> None:
        self.assertEqual(compute_fitness(0.0, 0.0, 0.0), 0.0)


class RiskExitTests(unittest.TestCase):
    def test_gap_through_stop_fills_at_open(self) -> None:
        g = TradingGenome()  # daily textbook: 5% stop
        reason, price = check_risk_exit(g, 100.0, 100.0, bar_open=90.0, bar_high=91.0, bar_low=89.0)
        self.assertEqual(reason, ExitReason.STOP_LOSS)
        self.assertEqual(price, 90.0)

    def test_stop_wins_when_both_levels_touched(self) -> None:
        g = TradingGenome()
        reason, _ = check_risk_exit(g, 100.0, 100.0, bar_open=100.0, bar_high=120.0, bar_low=90.0)
        self.assertIn(reason, (ExitReason.STOP_LOSS, ExitReason.TRAILING_STOP))

    def test_trailing_stop_ratchets_above_hard_stop(self) -> None:
        g = TradingGenome()  # 6% trail, 5% stop
        reason, price = check_risk_exit(g, 100.0, 110.0, bar_open=105.0, bar_high=105.0, bar_low=103.0)
        self.assertEqual(reason, ExitReason.TRAILING_STOP)
        self.assertAlmostEqual(price, 110.0 * 0.94)


class TraderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.market = synthetic_market()
        cls.windows = cls.market.split(0.7, config.INDICATOR_WARMUP_BARS)

    def test_backtest_flattens_and_accounts_correctly(self) -> None:
        bot = TradingBot(TradingGenome.textbook()).run(self.market, self.windows.train_start, self.windows.train_end)
        self.assertFalse(bot.positions, "every position must be closed at the end of the window")
        self.assertAlmostEqual(bot.final_equity, bot.cash, places=6)
        realized = sum(t.pnl for t in bot.trades)
        self.assertAlmostEqual(bot.final_equity - bot.initial_balance, realized, places=4)
        self.assertGreater(bot.total_trades, 0)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.market = synthetic_market()
        cls.windows = cls.market.split(0.7, config.INDICATOR_WARMUP_BARS)

    def test_exactly_one_survivor_and_purge_is_verified(self) -> None:
        rng = random.Random(11)
        alpha = TradingBot(TradingGenome.textbook(), role=BotRole.ALPHA)
        population = [alpha, *(alpha.spawn_clone(0.15, 1, rng) for _ in range(5))]
        del alpha
        for bot in population:
            bot.run(self.market, self.windows.train_start, self.windows.train_end)
        del bot  # the loop variable would otherwise keep one bot alive

        result = Evaluator().select_and_purge(population, generation=1)
        self.assertEqual(population, [], "population list must be emptied in place")
        self.assertEqual(len(result.leaderboard), 6)
        self.assertEqual(len(result.eliminated), 5)
        self.assertEqual(result.victor.bot_id, result.leaderboard[0].bot_id)
        self.assertTrue(result.purge_verified, f"{result.lingering_refs} purged bots still in memory")

    def test_selection_needs_a_population(self) -> None:
        with self.assertRaises(EvolutionIntegrityError):
            Evaluator().select_and_purge([TradingBot(TradingGenome.textbook())], generation=1)


class EvolutionEngineTests(unittest.TestCase):
    def test_three_generations_and_resume(self) -> None:
        market = synthetic_market()
        windows = market.split(0.7, config.INDICATOR_WARMUP_BARS)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            engine = EvolutionEngine(market, windows, quiet_display(), rng=random.Random(42),
                                     evaluator=Evaluator(min_trades=0), eval_folds=4, adaptive_mutation=True,
                                     checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs")
            self.assertEqual(len(engine.folds), 4)
            genome = TradingGenome.textbook()
            summary = engine.run(3, SeedAlpha(genome=genome, origin_genome=genome))

            self.assertEqual([h.generation for h in summary.history], [1, 2, 3])
            self.assertTrue(all(h.purge_verified for h in summary.history))
            self.assertTrue(all(len(h.eliminated) == engine.clone_count for h in summary.history))
            self.assertTrue(all(h.victor.n_folds == 4 for h in summary.history))
            for gen in (1, 2, 3):
                self.assertTrue((tmp_path / "ckpt" / f"alpha_gen_{gen}.json").is_file())

            # Fixed folds => the Alpha can only be replaced by a strictly better bot.
            fitness = [h.victor.fitness for h in summary.history]
            self.assertEqual(fitness, sorted(fitness))

            events = [json.loads(line)["event"] for line in (tmp_path / "logs" / "evolution_history.jsonl").read_text().splitlines()]
            self.assertEqual(events.count("selection"), 3)
            self.assertEqual(events.count("replication"), 3)  # genesis + after gens 1 and 2

            seed = seed_from_checkpoint(tmp_path / "ckpt" / config.LATEST_ALPHA_FILE)
            self.assertEqual(seed.start_generation, 4)
            self.assertEqual(seed.genome, summary.champion_genome)
            self.assertEqual(seed.origin_genome, genome, "the true Gen-1 ancestor must survive a resume")
            self.assertAlmostEqual(seed.mutation_rate, engine.mutation_rate, places=5)
            resumed = EvolutionEngine(market, windows, quiet_display(), rng=random.Random(1),
                                      evaluator=Evaluator(min_trades=0), eval_folds=4, adaptive_mutation=True,
                                      checkpoint_dir=tmp_path / "ckpt", log_dir=tmp_path / "logs").run(1, seed)
            self.assertEqual(resumed.history[0].generation, 4)
            self.assertGreaterEqual(resumed.history[0].victor.fitness, fitness[-1])

    def test_adaptive_mutation(self) -> None:
        market = synthetic_market()
        windows = market.split(0.7, config.INDICATOR_WARMUP_BARS)
        engine = EvolutionEngine(market, windows, quiet_display(), mutation_rate=0.15, adaptive_mutation=True)
        engine._adapt_mutation(incumbent_retained=True, show=False)
        self.assertAlmostEqual(engine.mutation_rate, 0.15 * config.MUTATION_STUCK_BOOST)
        engine._adapt_mutation(incumbent_retained=False, show=False)
        self.assertAlmostEqual(engine.mutation_rate, 0.15 * config.MUTATION_STUCK_BOOST * config.MUTATION_PROGRESS_DECAY)
        for _ in range(50):
            engine._adapt_mutation(incumbent_retained=True, show=False)
        self.assertEqual(engine.mutation_rate, config.MUTATION_RATE_MAX)
        for _ in range(50):
            engine._adapt_mutation(incumbent_retained=False, show=False)
        self.assertEqual(engine.mutation_rate, config.MUTATION_RATE_MIN)


class QualityAndQualificationTests(unittest.TestCase):
    def test_quality_rewards_win_rate_and_profit_factor(self) -> None:
        self.assertGreater(quality_multiplier(0.6, 1.5, 1.0, 1.0), quality_multiplier(0.4, 1.5, 1.0, 1.0))
        self.assertGreater(quality_multiplier(0.5, 2.0, 1.0, 1.0), quality_multiplier(0.5, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(quality_multiplier(0.5, 1.0, 1.0, 1.0), 1.0)
        self.assertEqual(quality_multiplier(0.9, 1.0, 0.0, 0.0), 1.0, "weights of 0 switch the bonus off")

    def test_better_quality_always_helps_even_losers(self) -> None:
        good, bad = quality_multiplier(0.6, 2.0, 1.0, 1.0), quality_multiplier(0.3, 0.5, 1.0, 1.0)
        self.assertGreater(apply_quality(10.0, good), apply_quality(10.0, bad))
        self.assertGreater(apply_quality(-10.0, good), apply_quality(-10.0, bad))

    def test_trade_pace_nudge(self) -> None:
        on_target = frequency_multiplier(20, target_minutes=20, weight=0.75)
        self.assertAlmostEqual(on_target, 1.0)
        too_slow, too_fast = frequency_multiplier(40, 20, 0.75), frequency_multiplier(10, 20, 0.75)
        self.assertAlmostEqual(too_slow, too_fast, msg="2x too slow and 2x too fast cost the same")
        self.assertLess(too_slow, on_target)
        self.assertLess(frequency_multiplier(160, 20, 0.75), too_slow, "further off = lower score")
        self.assertLess(frequency_multiplier(None, 20, 0.75), frequency_multiplier(160, 20, 0.75))
        self.assertEqual(frequency_multiplier(500, None, 0.75), 1.0, "no target = no nudge")

    def test_intraday_textbook_uses_tight_stops(self) -> None:
        self.assertLess(TradingGenome.textbook(intraday=True).stop_loss_pct, 0.02)
        self.assertEqual(TradingGenome.textbook(intraday=False), TradingGenome())

    def test_disqualified_bots_rank_last(self) -> None:
        market = synthetic_market()
        windows = market.split(0.7, config.INDICATOR_WARMUP_BARS)
        bots = [TradingBot(TradingGenome.textbook()).run(market, windows.train_start, windows.train_end) for _ in range(2)]
        reports = [Evaluator(min_trades=0).score(bots[0]), Evaluator(min_trades=10_000).score(bots[1])]
        reports[1] = replace(reports[1], fitness=reports[0].fitness + 100)  # DQ bot has the better score...
        ranked = Evaluator.rank(reports)
        self.assertFalse(ranked[0].disqualified, "...but a qualified bot must still win")
        self.assertTrue(ranked[1].disqualified)

    def test_folds_average_and_accumulate(self) -> None:
        market = synthetic_market()
        windows = market.split(0.7, config.INDICATOR_WARMUP_BARS)
        cal = market.calendar[(market.calendar >= windows.train_start) & (market.calendar <= windows.train_end)]
        half = len(cal) // 2
        folds = [(cal[0], cal[half - 1]), (cal[half], cal[-1])]
        bot = TradingBot(TradingGenome.textbook()).run_folds(market, folds)
        self.assertEqual(len(bot.folds), 2)
        self.assertEqual(sum(f.trade_count for f in bot.folds), bot.total_trades)
        for fold in bot.folds:  # each fold starts from a fresh balance
            self.assertLessEqual(abs(fold.equity_curve.iloc[0] - bot.initial_balance), bot.initial_balance * 0.5)
        report = Evaluator(min_trades=0).score(bot)
        self.assertEqual(report.n_folds, 2)


if __name__ == "__main__":
    unittest.main()
