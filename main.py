"""
Evolution Arena - entry point.

    python main.py                                  # interactive menu
    python main.py --mode evolve --generations 30   # Mode 1: fast evolution over history
    python main.py --mode live --dry-run --once     # Mode 2: one live signal check, no orders
    python main.py --mode live                      # Mode 2: trade the champion on Alpaca PAPER
    python main.py --mode lab                       # Evolution Lab v2: islands, invented rules, 2y of data
    python main.py --mode monkey                    # luck detector: real market vs shuffled "monkey" markets
    python main.py --mode walkforward               # evolve on the past, trade the next month, repeat
    python main.py --mode lab-live                  # the Lab ensemble trades your Alpaca PAPER account
    python main.py --mode reevolve                  # weekly: evolve a shadow challenger on the newest data
    python main.py --mode forecast --target both    # Forecaster Arena: bots scored on prediction accuracy
    python main.py --mode predict                   # Prophecy League LIVE: score yesterday, predict tomorrow
    python main.py --mode predict --replay 250      # Prophecy League REPLAY over the last 250 trading days

Run `python main.py --help` for every flag.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import config
from bot.genome import GenomeError, TradingGenome
from broker.base import BrokerError
from broker.simulated_broker import SimulatedBroker
from engine.evaluator import EvolutionIntegrityError
from engine.evolution import (
    EvolutionEngine,
    SeedAlpha,
    archive_previous_run,
    check_timeframe,
    load_checkpoint,
    seed_from_checkpoint,
)
from utils.display import Display
from utils.lab_display import LabDisplay
from utils.logger import setup_logging

MODES = ("evolve", "live", "lab", "monkey", "walkforward", "lab-live", "reevolve", "lab-promote", "forecast", "predict")

log = logging.getLogger("main")

EXIT_OK, EXIT_ERROR, EXIT_INTERRUPTED = 0, 1, 130


def _force_utf8_output() -> None:
    """Windows consoles/pipes default to cp1252, which can't print the arena's emoji."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evolutionary genetic-algorithm trading bots: 1 Alpha + 5 mutants, only one survives.",
    )
    parser.add_argument("--mode", choices=MODES, help="Skip the menu and run this mode directly")
    evo = parser.add_argument_group("Mode 1 - fast evolution")
    evo.add_argument("--generations", type=int, help=f"Generations to run (default {config.DEFAULT_GENERATIONS})")
    evo.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for a reproducible episode")
    evo.add_argument("--resume", action="store_true", help="Continue the lineage from checkpoints/latest_alpha.json")
    evo.add_argument("--random-start", action="store_true", help="Generation 1 Alpha gets random DNA instead of textbook settings")
    evo.add_argument("--synthetic", action="store_true", help="Use synthetic random-walk prices (offline rehearsal)")
    live = parser.add_argument_group("Mode 2 - live paper execution")
    live.add_argument("--checkpoint", type=Path, help="Genome file to deploy (default checkpoints/latest_alpha.json)")
    live.add_argument("--dry-run", action="store_true", help="Compute signals but never send orders (no keys needed)")
    live.add_argument("--once", action="store_true", help="Run a single live cycle and exit")
    live.add_argument("--poll", type=int, default=config.LIVE_POLL_SECONDS, help="Seconds between live cycles")
    lab = parser.add_argument_group("Evolution Lab (v2)")
    lab.add_argument("--unbench", action="store_true", help="Lab live: clear a drift-alarm bench and resume trading")
    lab.add_argument("--days", type=int, default=config.LAB_HISTORY_DAYS, help="Lab: days of 5-minute history to use")
    lab.add_argument("--target", choices=("direction", "volatility", "both"),
                     help="Forecaster: predict UP/DOWN, BIG/CALM moves, or both")
    lab.add_argument("--replay", type=int, help="Prophecy League: replay the last N trading days instead of live")
    lab.add_argument("--no-claude", action="store_true",
                     help="Prophecy League: don't let Claude design children (all random mutation)")
    parser.add_argument("--fast", action="store_true", help="No dramatic pauses (for testing / long runs)")
    args = parser.parse_args(argv)
    if args.generations is not None and args.generations < 1:
        parser.error("--generations must be >= 1")
    if args.poll < 10:
        parser.error("--poll must be >= 10 seconds")
    return args


# --------------------------------------------------------------------------- #
# Interactive menu
# --------------------------------------------------------------------------- #
def interactive_menu(display: Display) -> str | None:
    from rich.prompt import Prompt

    default = "2" if config.EXECUTION_MODE == "alpaca" else "1"
    display.console.print(
        f"[bold]  1[/]  Fast Evolution Loop    [bright_black]- v1: evolve 20-50 generations over historical {config.BAR_INTERVAL} bars (yfinance)[/]\n"
        "[bold]  2[/]  Live Paper Execution   [bright_black]- v1: deploy the evolved Alpha to your Alpaca PAPER account[/]\n"
        "[bold magenta]  3[/]  🧫 Evolution Lab        [bright_black]- v2: islands, bots that invent rules, 2 years x 20 stocks[/]\n"
        "[bold magenta]  4[/]  🐒 Luck detector        [bright_black]- evolve on real vs shuffled 'monkey' markets[/]\n"
        "[bold magenta]  5[/]  🚶 Walk-forward test    [bright_black]- evolve on the past, trade the next month, repeat[/]\n"
        "[bold magenta]  6[/]  🗳  Lab live trading     [bright_black]- the Lab's ensemble trades your Alpaca PAPER account[/]\n"
        "[bold magenta]  7[/]  🔄 Re-evolve challenger [bright_black]- weekly: evolve on the newest data, shadow-test it live[/]\n"
        "[bold cyan]  8[/]  🔮 Forecaster Arena     [bright_black]- bots scored on prediction accuracy: UP/DOWN and BIG/CALM moves[/]\n"
        "[bold cyan]  9[/]  📜 Prophecy League      [bright_black]- bots predict the REAL future daily; most accurate survive[/]\n"
        "[bold]  q[/]  Quit"
    )
    choice = Prompt.ask("Select mode", choices=["1", "2", "3", "4", "5", "6", "7", "8", "9", "q"], default=default,
                        console=display.console)
    return {"1": "evolve", "2": "live", "3": "lab", "4": "monkey", "5": "walkforward", "6": "lab-live",
            "7": "reevolve", "8": "forecast", "9": "predict"}.get(choice)


# --------------------------------------------------------------------------- #
# Mode 1
# --------------------------------------------------------------------------- #
def run_evolution(args: argparse.Namespace, display: Display, interactive: bool) -> int:
    from rich.prompt import Confirm, IntPrompt

    latest = config.CHECKPOINT_DIR / config.LATEST_ALPHA_FILE
    generations = args.generations
    resume = args.resume
    if interactive:
        if generations is None:
            generations = IntPrompt.ask("How many generations? (20-50 recommended)",
                                        default=config.DEFAULT_GENERATIONS, console=display.console)
            generations = max(1, generations)
        if not resume and latest.is_file():
            resume = Confirm.ask("A previous Alpha exists. Continue its lineage?", default=False, console=display.console)
    generations = generations or config.DEFAULT_GENERATIONS

    rng = random.Random(args.seed)
    broker = SimulatedBroker(force_synthetic=args.synthetic, seed=args.seed)
    with display.console.status(f"Loading daily bars for {', '.join(config.TARGET_SYMBOLS)}..."):
        market = broker.load_market_data(config.TARGET_SYMBOLS)
    windows = market.split(config.TRAIN_FRACTION, config.INDICATOR_WARMUP_BARS)
    display.data_summary(market, windows)

    if resume:
        seed = seed_from_checkpoint(latest)
        display.info(f"Resuming lineage: Bot #{seed.bot_id} enters Generation {seed.start_generation} as Alpha.")
    else:
        archived = archive_previous_run()
        if archived:
            display.info(f"Previous run's checkpoints archived to {archived.relative_to(config.BASE_DIR).as_posix()}")
        genome = TradingGenome.random(rng) if args.random_start else TradingGenome.textbook()
        seed = SeedAlpha(genome=genome, origin_genome=genome)
        display.info("Generation 1 Alpha: " + ("random DNA" if args.random_start else
                     "textbook settings (RSI 14/70/30, SMA 20/50, MACD 12/26/9)"))

    engine = EvolutionEngine(market, windows, display, rng=rng)
    summary = engine.run(generations, seed)
    display.final_summary(summary)
    log.info("Run %s finished: %d generations, interrupted=%s", summary.run_id, len(summary.history), summary.interrupted)
    return EXIT_INTERRUPTED if summary.interrupted else EXIT_OK


# --------------------------------------------------------------------------- #
# Mode 2
# --------------------------------------------------------------------------- #
def run_live(args: argparse.Namespace, display: Display, interactive: bool) -> int:
    from rich.prompt import Confirm

    from broker.alpaca_broker import AlpacaBroker
    from engine.live_trader import LiveTrader, describe_checkpoint

    path = args.checkpoint or (config.CHECKPOINT_DIR / config.LATEST_ALPHA_FILE)
    if not path.is_file():
        display.error(f"No evolved genome found at {path}.\nRun Mode 1 (Fast Evolution Loop) first.")
        return EXIT_ERROR
    data = load_checkpoint(path)
    check_timeframe(data, path)
    genome = TradingGenome.from_dict(data["genome"])
    display.genome_panel(genome, f"👑 Deploying {describe_checkpoint(data)}")

    dry_run = args.dry_run
    if interactive and not dry_run:
        dry_run = not Confirm.ask("Send orders to your Alpaca PAPER account? (No = dry run, signals only)",
                                  default=False, console=display.console)

    broker: AlpacaBroker | None = None
    try:
        broker = AlpacaBroker(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    except BrokerError as exc:
        if not dry_run:
            display.error(str(exc))
            return EXIT_ERROR
        display.warn(f"{exc}\nContinuing in dry-run mode with a simulated flat ${config.INITIAL_BALANCE:,.0f} account.")

    if config.LIVE_DATA_SOURCE == "alpaca" and broker is not None:
        data_source, source_label = broker, "Alpaca (IEX feed)"
    else:
        data_source, source_label = SimulatedBroker(use_cache=False, allow_synthetic_fallback=False), "yfinance"

    trader = LiveTrader(genome, broker, data_source, config.TARGET_SYMBOLS, display, dry_run=dry_run)
    display.live_header(trader.account(), dry_run, source_label, config.TARGET_SYMBOLS)
    trader.run(poll_seconds=args.poll, max_cycles=1 if args.once else None)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Evolution Lab (v2)
# --------------------------------------------------------------------------- #
def _load_lab(args: argparse.Namespace, display: LabDisplay):
    from lab.data import load_lab_data
    from lab.features import build_features

    with display.console.status("Loading 5-minute history...") as status:
        data = load_lab_data(days=args.days, synthetic=args.synthetic, seed=args.seed,
                             progress=lambda m: status.update(m))
        status.update("Computing 33 features for every stock and bar...")
        fs = build_features(data)
    return data, fs


def _archive_lab() -> None:
    from datetime import datetime

    files = list(config.LAB_DIR.glob("lab_gen_*.json")) + [
        f for f in (config.LAB_DIR / "lab_latest.json", config.LAB_DIR / "lab_report.png") if f.is_file()]
    if files:
        dest = config.LAB_DIR / "archive" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
        dest.mkdir(parents=True, exist_ok=True)
        for f in files:
            f.replace(dest / f.name)


def run_lab(args: argparse.Namespace, display: LabDisplay, interactive: bool) -> int:
    from rich.prompt import Confirm, IntPrompt

    from lab.evolution import LabEngine, load_lab_seed

    latest = config.LAB_DIR / "lab_latest.json"
    generations, resume = args.generations, args.resume
    if interactive:
        if generations is None:
            generations = max(1, IntPrompt.ask("How many generations? (30-60 recommended)",
                                               default=config.LAB_DEFAULT_GENERATIONS, console=display.console))
        if not resume and latest.is_file():
            resume = Confirm.ask("A previous Lab run exists. Continue it?", default=False, console=display.console)
    generations = generations or config.LAB_DEFAULT_GENERATIONS
    data, fs = _load_lab(args, display)
    seed = load_lab_seed(latest) if resume else None
    if not resume:
        _archive_lab()
    engine = LabEngine(data, fs, display, rng=random.Random(args.seed))
    display.lab_intro(data, engine.window_dates(), generations)
    summary = engine.run(generations, seed)
    display.lab_final(summary, fs)
    return EXIT_INTERRUPTED if summary.interrupted else EXIT_OK


def run_monkey(args: argparse.Namespace, display: LabDisplay) -> int:
    from lab.experiments import monkey_test

    gens = args.generations or 15
    data, _ = _load_lab(args, display)
    display.info(f"Luck detector: 1 real + {config.LAB_MONKEY_RUNS} monkey evolutions x {gens} generations each "
                 "(this takes a while)...")
    with display.console.status("Evolving...") as status:
        real, monkeys = monkey_test(data, gens, seed=args.seed or 0, say=lambda m: status.update(m))
    display.monkey_report(real, monkeys)
    return EXIT_OK


def run_walkforward(args: argparse.Namespace, display: LabDisplay) -> int:
    from lab.experiments import walk_forward

    gens = args.generations or 8
    data, _ = _load_lab(args, display)
    with display.console.status("Walking forward...") as status:
        steps, total, bench = walk_forward(data, generations=gens, seed=args.seed or 0, say=lambda m: status.update(m))
    if not steps:
        display.error("Not enough history for a walk-forward test (need ~150 trading days).")
        return EXIT_ERROR
    display.walk_forward_report(steps, total, bench)
    return EXIT_OK


def run_reevolve(args: argparse.Namespace, display: LabDisplay) -> int:
    from lab.evolution import LabEngine, load_lab_seed

    latest = config.LAB_DIR / "lab_latest.json"
    if not latest.is_file():
        display.error("No Lab run to build on yet - run the Evolution Lab (menu 3) first.")
        return EXIT_ERROR
    gens = args.generations or 15
    data, fs = _load_lab(args, display)
    engine = LabEngine(data, fs, display, rng=random.Random(args.seed), latest_name="lab_challenger.json")
    display.lab_intro(data, engine.window_dates(), gens)
    summary = engine.run(gens, load_lab_seed(latest))
    display.lab_final(summary, fs)
    display.success("Shadow challenger saved. Lab live trading (menu 6) will now paper-trade it virtually "
                    f"alongside the live ensemble and recommend promotion after {config.LAB_SHADOW_PROMOTE_DAYS} days.")
    return EXIT_OK


def run_lab_live(args: argparse.Namespace, display: LabDisplay, interactive: bool) -> int:
    from rich.prompt import Confirm

    from broker.alpaca_broker import AlpacaBroker
    from lab.live import KILL_FILE, LabLive, load_deploy

    deploy = load_deploy()
    dry_run = args.dry_run
    if interactive and not dry_run:
        dry_run = not Confirm.ask("Send orders to your Alpaca PAPER account? (No = dry run)", default=False,
                                  console=display.console)
    broker = None
    try:
        broker = AlpacaBroker(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    except BrokerError as exc:
        if not dry_run:
            display.error(str(exc))
            return EXIT_ERROR
        display.warn(f"{exc} - dry run with a virtual $100,000 book.")
    live = LabLive(deploy, broker, display, dry_run=dry_run, unbench=args.unbench)
    display.console.print(f"[bold]Ensemble:[/] {len(live.ensemble.members)} bots · quorum {live.ensemble.quorum:.0%} · "
                          f"{'shadow challenger active' if live.challenger else 'no shadow challenger'} · "
                          f"kill switch: create the file {KILL_FILE.relative_to(config.BASE_DIR).as_posix()}")
    live.run(poll_seconds=max(args.poll, 30), max_cycles=1 if args.once else None)
    return EXIT_OK


def run_forecast(args: argparse.Namespace, display: LabDisplay, interactive: bool) -> int:
    from rich.prompt import IntPrompt, Prompt

    from lab.forecast import ForecastEngine, build_daily_features, load_daily

    target = args.target
    generations = args.generations
    if interactive:
        if target is None:
            pick = Prompt.ask("Predict what? [1] UP/DOWN direction  [2] BIG/CALM moves  [3] both",
                              choices=["1", "2", "3"], default="3", console=display.console)
            target = {"1": "direction", "2": "volatility", "3": "both"}[pick]
        if generations is None:
            generations = max(1, IntPrompt.ask("How many generations? (30-60 recommended)", default=40,
                                               console=display.console))
    target = target or "both"
    generations = generations or 40
    with display.console.status("Loading daily history...") as status:
        data = load_daily(progress=lambda m: status.update(m), synthetic=args.synthetic, seed=args.seed)
        status.update("Computing features and answers...")
        fs, answers = build_daily_features(data)
    for t in (("direction", "volatility") if target == "both" else (target,)):
        engine = ForecastEngine(fs, answers, data.index, t, display, rng=random.Random(args.seed))
        display.forecast_intro(engine, data.n_symbols, generations, data.source)
        summary = engine.run(generations)
        display.forecast_final(summary, fs)
    return EXIT_OK


def run_prophecy(args: argparse.Namespace, display: LabDisplay, interactive: bool) -> int:
    from rich.prompt import IntPrompt, Prompt

    from lab.claude_breeder import make_breeder
    from lab.forecast import build_daily_features, load_daily
    from lab.prophecy import live_step, replay

    breeder, status_msg = make_breeder(enabled=not args.no_claude)
    (display.success if breeder else display.info)(status_msg)
    days = args.replay
    if interactive and days is None:
        pick = Prompt.ask("[1] LIVE: score yesterday's calls & predict tomorrow (run each evening)   "
                          "[2] REPLAY the last year day by day", choices=["1", "2"], default="1",
                          console=display.console)
        if pick == "2":
            days = max(20, IntPrompt.ask("How many trading days to replay?", default=250, console=display.console))
    with display.console.status("Loading daily history...") as status:
        data = load_daily(progress=lambda m: status.update(m), synthetic=args.synthetic, seed=args.seed)
        status.update("Computing features and answers...")
        fs, answers = build_daily_features(data)
    if days:
        targets = ("direction", "volatility") if (args.target or "both") == "both" else (args.target,)
        for t in targets:
            summary = replay(fs, answers, data.index, t, display, days=days, seed=args.seed, breeder=breeder)
            display.prophecy_replay_summary(summary, fs)
        display.claude_breeder_status(breeder)
        return EXIT_OK
    live_step(fs, answers, data.index, data.symbols, display, seed=args.seed, breeder=breeder)
    display.claude_breeder_status(breeder)
    return EXIT_OK


# --------------------------------------------------------------------------- #
# Entry
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    _force_utf8_output()
    args = parse_args(argv)
    setup_logging()
    display = LabDisplay(delay=0.0 if args.fast else config.DISPLAY_DELAY)

    try:
        config.validate_config()
        display.banner()
        interactive = args.mode is None
        mode = interactive_menu(display) if interactive else args.mode
        if mode is None:
            display.info("Goodbye.")
            return EXIT_OK
        if mode == "evolve":
            return run_evolution(args, display, interactive)
        if mode == "live":
            return run_live(args, display, interactive)
        if mode == "lab":
            return run_lab(args, display, interactive)
        if mode == "monkey":
            return run_monkey(args, display)
        if mode == "walkforward":
            return run_walkforward(args, display)
        if mode == "reevolve":
            return run_reevolve(args, display)
        if mode == "forecast":
            return run_forecast(args, display, interactive)
        if mode == "predict":
            return run_prophecy(args, display, interactive)
        if mode == "lab-promote":
            from lab.live import promote_challenger

            display.success(f"Challenger promoted to live ensemble (previous one backed up as {promote_challenger()}).")
            return EXIT_OK
        return run_lab_live(args, display, interactive)
    except KeyboardInterrupt:
        display.warn("Interrupted.")
        return EXIT_INTERRUPTED
    except EOFError:
        display.error("No interactive input available. Pass --mode (see --help).")
        return EXIT_ERROR
    except (config.ConfigError, BrokerError, GenomeError, EvolutionIntegrityError, ValueError) as exc:
        log.exception("Fatal error")
        display.error(str(exc))
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
