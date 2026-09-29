"""
Global configuration for the Evolutionary Genetic Algorithm Trading Bot.

Every tunable knob lives here so a YouTube viewer can open ONE file and see
exactly how the arena is set up. Values can be overridden with environment
variables (or a local `.env` file) where noted.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Final, Literal, get_args


# ---------------------------------------------------------------------------
# Tiny .env loader (avoids a python-dotenv dependency)
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding real env vars."""
    if not path.is_file():
        return
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        # A broken .env should never stop the bot from starting.
        pass


_load_dotenv(BASE_DIR / ".env")


# ---------------------------------------------------------------------------
# Core arena settings
# ---------------------------------------------------------------------------
INITIAL_BALANCE: float = 10_000.0          # Every bot starts each generation with $10k
POPULATION_SIZE: int = 6                   # 1 Alpha + 5 clones. Try 11 (10 clones) for more tries per generation
MUTATION_RATE: float = 0.15                # Starting Gaussian drift per gene (sigma = 15% of value)

ExecutionMode = Literal["yfinance", "alpaca"]
# "yfinance" -> fast historical backtest evolution (Mode 1 is the default menu choice)
# "alpaca"   -> alpaca-py paper trading API (Mode 2 is the default menu choice)
EXECUTION_MODE: ExecutionMode = os.getenv("EXECUTION_MODE", "yfinance").strip().lower()  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Timeframe: how often bots can trade
# ---------------------------------------------------------------------------
# "5m" -> 5-minute bars; bots are nudged toward ~1 trade every 20 minutes.
#         (yfinance only serves ~60 days of 5-minute history.)
# "1d" -> daily bars; slow swing trading over years of history (the original mode).
Timeframe = Literal["5m", "1d"]
TIMEFRAME: Timeframe = os.getenv("TIMEFRAME", "5m").strip().lower()  # type: ignore[assignment]

_PROFILES: dict[str, dict] = {
    "5m": {
        "bar_minutes": 5,
        "history_start": (date.today() - timedelta(days=58)).isoformat(),
        "commission_pct": 0.0,             # Alpaca stock trades are commission-free
        "crypto_commission_pct": 0.0025,   # Alpaca crypto taker fee (0.25%) - frequent BTC trading is expensive!
        "slippage_pct": 0.0002,            # 2 bps adverse fill on liquid names
        "target_minutes_per_trade": 20.0,
        "live_poll_seconds": 60,
        "live_lookback_days": 10,
        # No BTC here: its 0.25% fee each way is ~15x a typical 20-minute price move,
        # so frequent BTC trading loses on every trade. Add "BTC-USD" back to test that.
        "symbols": ["AAPL", "NVDA", "TSLA"],
    },
    "1d": {
        "bar_minutes": 1440,
        "history_start": "2019-01-01",
        "commission_pct": 0.0005,
        "crypto_commission_pct": 0.0005,
        "slippage_pct": 0.0005,
        "target_minutes_per_trade": None,  # no trade-frequency preference
        "live_poll_seconds": 300,
        "live_lookback_days": 450,
        "symbols": ["AAPL", "NVDA", "TSLA", "BTC-USD"],
    },
}
_P = _PROFILES.get(TIMEFRAME, _PROFILES["5m"])

TARGET_SYMBOLS: list[str] = _P["symbols"]

# Trade-frequency tendency (measured while markets are open): fitness is multiplied by
#   exp(-TRADE_FREQUENCY_WEIGHT x |ln(actual minutes per trade / target)|)
# e.g. trading every 40 min (2x too slow) with weight 0.75 keeps ~59% of the score.
# Set TARGET_MINUTES_PER_TRADE = None to switch it off.
TARGET_MINUTES_PER_TRADE: float | None = _P["target_minutes_per_trade"]
TRADE_FREQUENCY_WEIGHT: float = 0.75
# Pace rule: only bots trading every TARGET/PACE_TOLERANCE .. TARGET*PACE_TOLERANCE minutes
# (10-40 min for a 20 min target) can win the crown. If none manage it, the bot
# CLOSEST to the pace wins - so evolution locks onto the pace first, then profit.
# Set to None to make the pace only a gentle fitness nudge.
PACE_TOLERANCE: float | None = 2.0


# ---------------------------------------------------------------------------
# Historical data / evaluation window
# ---------------------------------------------------------------------------
BAR_INTERVAL: str = TIMEFRAME
BAR_MINUTES: int = _P["bar_minutes"]
IS_INTRADAY: bool = BAR_MINUTES < 1440
HISTORY_START: str = _P["history_start"]   # First bar downloaded (includes indicator warm-up)
HISTORY_END: str | None = None             # None = up to now
TRAIN_FRACTION: float = 0.70               # Evolve on the first 70%, hold out 30% for an honest test
INDICATOR_WARMUP_BARS: int = 220           # Bars reserved before trading so a 200-bar SMA is valid
EVAL_FOLDS: int = 1                        # 1 = one long training period. 4 = cut it into 4 separate
                                           # periods; every bot trades each one from a fresh $10k and
                                           # its fitness is the AVERAGE - winners must work in every era,
                                           # not just get lucky once.

# Execution cost model for the simulated broker (applied on every fill)
COMMISSION_PCT: float = _P["commission_pct"]                 # stocks, per side
CRYPTO_COMMISSION_PCT: float = _P["crypto_commission_pct"]   # crypto, per side
SLIPPAGE_PCT: float = _P["slippage_pct"]
MIN_ORDER_NOTIONAL: float = 10.0           # Ignore dust-sized orders


# ---------------------------------------------------------------------------
# Evolution loop
# ---------------------------------------------------------------------------
DEFAULT_GENERATIONS: int = 25              # Mode 1 default (the menu suggests 20-50)
RANDOM_SEED: int | None = None             # Set an int for a reproducible "episode"

# Fitness quality bonus: fitness is multiplied by
#   (1 + WIN_RATE_WEIGHT x (win_rate - 0.5))  x  profit_factor ^ (PROFIT_FACTOR_WEIGHT / 2)
# so a bot that wins more often AND wins bigger than it loses is preferred.
# Set a weight to 0 to switch that bonus off.
# (Off by default: in testing it lowered out-of-sample profit. Set both to 1.0 to try it.)
WIN_RATE_WEIGHT: float = 0.0
PROFIT_FACTOR_WEIGHT: float = 0.0
MIN_TRADES: int = 20                       # Bots with fewer trades (across all folds) are DISQUALIFIED:
                                           # a handful of trades can't prove skill over luck.

# Adaptive mutation: when the Alpha keeps defending its crown, evolution is
# stuck -> mutate harder to explore. When a clone wins, mutate gently to
# fine-tune around the new champion.
ADAPTIVE_MUTATION: bool = False             # True to switch it on
MUTATION_RATE_MIN: float = 0.05
MUTATION_RATE_MAX: float = 0.40
MUTATION_STUCK_BOOST: float = 1.25         # rate x1.25 each generation the Alpha defends
MUTATION_PROGRESS_DECAY: float = 0.80      # rate x0.80 whenever a new champion appears


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CHECKPOINT_DIR: Final[Path] = BASE_DIR / "checkpoints"
DATA_CACHE_DIR: Final[Path] = BASE_DIR / "data_cache"
LOG_DIR: Final[Path] = BASE_DIR / "logs"
LATEST_ALPHA_FILE: Final[str] = "latest_alpha.json"


# ---------------------------------------------------------------------------
# Display (YouTube pacing)
# ---------------------------------------------------------------------------
DISPLAY_DELAY: float = 0.35                # Dramatic pause between log lines (seconds). --fast sets 0.


# ---------------------------------------------------------------------------
# Alpaca paper trading (Mode 2)
# ---------------------------------------------------------------------------
ALPACA_API_KEY: str | None = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
ALPACA_SECRET_KEY: str | None = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
# NOTE: The Alpaca broker is hard-wired to paper=True. There is intentionally no
# switch for real money in this project.
LiveDataSource = Literal["yfinance", "alpaca"]
LIVE_DATA_SOURCE: LiveDataSource = os.getenv("LIVE_DATA_SOURCE", "yfinance").strip().lower()  # type: ignore[assignment]
LIVE_POLL_SECONDS: int = _P["live_poll_seconds"]    # How often the live loop re-checks signals / stops
LIVE_LOOKBACK_DAYS: int = _P["live_lookback_days"]  # Calendar days of bars fetched for live indicators
LIVE_STATE_FILE: Final[str] = "live_state.json"


# ---------------------------------------------------------------------------
# Claude breeder (Prophecy League): an AI designs some of the children
# ---------------------------------------------------------------------------
ANTHROPIC_API_KEY: str | None = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_BREEDER: bool = os.getenv("CLAUDE_BREEDER", "on").strip().lower() not in ("off", "0", "false", "no")
CLAUDE_MODEL: str = os.getenv("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_CHILDREN: int = 3                    # of the 5 children; 1 random mutant + 1 crossover always stay as a control


# ===========================================================================
# EVOLUTION LAB (v2) - islands, rule-inventing bots, years of data, luck tests
# ===========================================================================
# Traded universe: 20 of the most liquid US stocks (commission-free on Alpaca).
LAB_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "NFLX",
    "JPM", "V", "XOM", "UNH", "COST", "CRM", "ORCL", "INTC", "BAC", "WMT",
]
LAB_CONTEXT: list[str] = ["SPY", "QQQ"]     # market-mood inputs (not traded)
BAR_MINUTES_LAB: int = 5                    # the Lab always trades 5-minute bars
LAB_HISTORY_DAYS: int = 730                 # 2 years of 5-minute bars from Alpaca (SIP feed)
LAB_SPLIT: tuple[float, float, float] = (0.60, 0.20, 0.20)  # train / validation / FINAL test (untouched)
LAB_TRAIN_FOLDS: int = 6                    # training is cut into 6 eras...
LAB_FOLDS_PER_GEN: int = 4                  # ...and each generation sees a random 4 (noise vs memorising)
LAB_NEWS: bool = True                       # headline sentiment from Alpaca's free news feed

# Arena
LAB_ISLANDS: int = 4                        # separate arenas, one Alpha each
LAB_BOTS_PER_ISLAND: int = 6                # 1 Alpha + 5 challengers (3 mutants, 1 crossover child, 1 wildcard)
LAB_MIGRATION_EVERY: int = 5                # every 5 gens the wildcard is a migrant from the next island
LAB_HALL_OF_FAME: int = 10                  # best champions ever (they return as challengers + form the ensemble)
LAB_ENSEMBLE_SIZE: int = 5                  # top Hall-of-Fame bots that vote on live trades
LAB_ENSEMBLE_QUORUM: float = 0.5            # weighted share of votes needed to act
LAB_DEFAULT_GENERATIONS: int = 40

# Rule trees (genetic programming)
LAB_MAX_TREE_DEPTH: int = 4
LAB_COMPLEXITY_PENALTY: float = 0.02        # fitness / (1 + 0.02 x rule nodes): simpler rules generalise better

# Robustness tests
LAB_WOBBLE_VARIANTS: int = 6                # plateau test: nudge a finalist's genes this many times...
LAB_WOBBLE_PCT: float = 0.10                # ...by about +/-10%; fragile bots lose the crown
LAB_MONTE_CARLO_RUNS: int = 2000
LAB_MONKEY_RUNS: int = 3                    # evolutions on shuffled (pattern-free) prices for the luck detector

# Execution realism
LAB_BASE_COST_BPS: float = 1.0              # minimum cost per fill (spread + slippage), basis points
LAB_SPREAD_FACTOR: float = 0.10             # + 10% of the bar's high-low range, capped at LAB_MAX_COST_BPS
LAB_MAX_COST_BPS: float = 20.0
# Trading-pace rule for the Lab. OFF by default: in testing (3 seeds x 40 generations on 2 years of data),
# forcing ~1 trade every 20 minutes lost money on the final test every time (costs eat the tiny moves),
# while letting evolution pick its own pace broke even. Set to 20.0 to force the fast pace anyway.
LAB_PACE_TARGET_MINUTES: float | None = None

# Live (ensemble) trading
LAB_DIR: Final[Path] = CHECKPOINT_DIR / "lab"
LAB_FLATTEN_MINUTE: int = 385               # close everything at 15:55 ET (385 minutes after the open)
LAB_SHADOW_PROMOTE_DAYS: int = 10           # a shadow challenger must out-earn the champion for 10 trading days
LAB_DRIFT_WINDOW_DAYS: int = 5              # drift alarm: rolling 5-day live return vs backtest 5th percentile


class ConfigError(ValueError):
    """Raised when config values are inconsistent."""


def validate_config() -> None:
    """Fail fast with a readable message if someone fat-fingers a setting."""
    problems: list[str] = []
    if INITIAL_BALANCE <= 0:
        problems.append("INITIAL_BALANCE must be positive")
    if POPULATION_SIZE < 2:
        problems.append("POPULATION_SIZE must be >= 2 (1 survivor + at least 1 clone)")
    if not 0.0 < MUTATION_RATE < 1.0:
        problems.append("MUTATION_RATE must be between 0 and 1")
    if not TARGET_SYMBOLS:
        problems.append("TARGET_SYMBOLS cannot be empty")
    if EXECUTION_MODE not in get_args(ExecutionMode):
        problems.append(f"EXECUTION_MODE must be one of {get_args(ExecutionMode)}, got {EXECUTION_MODE!r}")
    if LIVE_DATA_SOURCE not in get_args(LiveDataSource):
        problems.append(f"LIVE_DATA_SOURCE must be one of {get_args(LiveDataSource)}, got {LIVE_DATA_SOURCE!r}")
    if not 0.1 <= TRAIN_FRACTION <= 1.0:
        problems.append("TRAIN_FRACTION must be between 0.1 and 1.0")
    if not 1 <= EVAL_FOLDS <= 12:
        problems.append("EVAL_FOLDS must be between 1 and 12")
    if MIN_TRADES < 0:
        problems.append("MIN_TRADES cannot be negative")
    if WIN_RATE_WEIGHT < 0 or PROFIT_FACTOR_WEIGHT < 0:
        problems.append("WIN_RATE_WEIGHT and PROFIT_FACTOR_WEIGHT cannot be negative")
    if not 0.0 < MUTATION_RATE_MIN <= MUTATION_RATE <= MUTATION_RATE_MAX < 1.0:
        problems.append("Need 0 < MUTATION_RATE_MIN <= MUTATION_RATE <= MUTATION_RATE_MAX < 1")
    if COMMISSION_PCT < 0 or CRYPTO_COMMISSION_PCT < 0 or SLIPPAGE_PCT < 0:
        problems.append("Commission and slippage cannot be negative")
    if TIMEFRAME not in _PROFILES:
        problems.append(f"TIMEFRAME must be one of {list(_PROFILES)}, got {TIMEFRAME!r}")
    if PACE_TOLERANCE is not None and PACE_TOLERANCE < 1:
        problems.append("PACE_TOLERANCE must be >= 1 or None")
    if TARGET_MINUTES_PER_TRADE is not None and TARGET_MINUTES_PER_TRADE <= 0:
        problems.append("TARGET_MINUTES_PER_TRADE must be positive or None")
    if problems:
        raise ConfigError("Invalid configuration:\n  - " + "\n  - ".join(problems))


def clone_count() -> int:
    """Number of mutated clones spawned from each Alpha."""
    return POPULATION_SIZE - 1
