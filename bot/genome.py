"""
TradingGenome - the DNA of a trading bot.

A genome is an immutable bundle of strategy parameters. Bots never edit their
own DNA; reproduction produces a brand-new genome via `mutate()`, which applies
Gaussian drift to every gene and then repairs the result so it always stays
inside realistic trading boundaries.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, Mapping


class GenomeError(ValueError):
    """Raised when a genome is malformed or outside its allowed boundaries."""


@dataclass(frozen=True)
class GeneSpec:
    """Allowed range and type of a single gene."""

    low: float
    high: float
    is_int: bool
    description: str

    def clamp(self, value: float) -> float | int:
        value = min(max(value, self.low), self.high)
        return int(round(value)) if self.is_int else round(float(value), 4)


# Realistic boundaries. Indicator periods are in BARS (days on 1d, 5-minute
# slices on 5m); the risk genes span both intraday (~0.5%) and swing (~10%) sizes.
GENE_SPECS: dict[str, GeneSpec] = {
    "rsi_period":        GeneSpec(5, 30, True, "RSI lookback (bars)"),
    "rsi_overbought":    GeneSpec(55, 90, True, "RSI level that triggers an exit"),
    "rsi_oversold":      GeneSpec(10, 45, True, "RSI level for mean-reversion entries"),
    "fast_sma":          GeneSpec(5, 60, True, "Fast trend SMA (bars)"),
    "slow_sma":          GeneSpec(20, 200, True, "Slow trend SMA (bars)"),
    "macd_fast":         GeneSpec(4, 20, True, "MACD fast EMA (bars)"),
    "macd_slow":         GeneSpec(15, 50, True, "MACD slow EMA (bars)"),
    "macd_signal":       GeneSpec(3, 15, True, "MACD signal EMA (bars)"),
    "stop_loss_pct":     GeneSpec(0.002, 0.15, False, "Hard stop below entry"),
    "take_profit_pct":   GeneSpec(0.002, 0.50, False, "Take-profit above entry"),
    "trailing_stop_pct": GeneSpec(0.002, 0.20, False, "Trailing stop below peak"),
    "position_size_pct": GeneSpec(0.05, 0.50, False, "Fraction of equity per position"),
}

# Minimum gaps that keep paired genes meaningful.
_MIN_SMA_GAP = 5
_MIN_MACD_GAP = 2
_MIN_RSI_GAP = 10


@dataclass(frozen=True)
class TradingGenome:
    """Immutable parameter set that fully defines a bot's trading behaviour."""

    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    fast_sma: int = 20
    slow_sma: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.12
    trailing_stop_pct: float = 0.06
    position_size_pct: float = 0.25

    GENE_NAMES: ClassVar[tuple[str, ...]] = tuple(GENE_SPECS)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        for name, spec in GENE_SPECS.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise GenomeError(f"Gene {name!r} must be numeric, got {value!r}")
            if spec.is_int and int(value) != value:
                raise GenomeError(f"Gene {name!r} must be an integer, got {value!r}")
            if not spec.low <= value <= spec.high:
                raise GenomeError(f"Gene {name!r}={value} outside [{spec.low}, {spec.high}]")
        if self.fast_sma + _MIN_SMA_GAP > self.slow_sma:
            raise GenomeError(f"fast_sma ({self.fast_sma}) must be at least {_MIN_SMA_GAP} below slow_sma ({self.slow_sma})")
        if self.macd_fast + _MIN_MACD_GAP > self.macd_slow:
            raise GenomeError(f"macd_fast ({self.macd_fast}) must be at least {_MIN_MACD_GAP} below macd_slow ({self.macd_slow})")
        if self.rsi_oversold + _MIN_RSI_GAP > self.rsi_overbought:
            raise GenomeError("rsi_oversold must be well below rsi_overbought")

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #
    @classmethod
    def textbook(cls, intraday: bool | None = None) -> "TradingGenome":
        """
        The classic settings every trading course teaches - our Generation 1 Alpha.
        Intraday (5-minute bars) keeps the indicator periods but uses tight,
        scalping-sized risk levels.
        """
        if intraday is None:
            import config

            intraday = config.IS_INTRADAY
        if intraday:
            return cls(stop_loss_pct=0.008, take_profit_pct=0.012, trailing_stop_pct=0.01)
        return cls()

    @classmethod
    def random(cls, rng: random.Random | None = None) -> "TradingGenome":
        """A uniformly random (but valid) genome."""
        rng = rng or random.Random()
        values = {name: rng.uniform(spec.low, spec.high) for name, spec in GENE_SPECS.items()}
        return cls(**_repair(values))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradingGenome":
        """Rebuild a genome from JSON. Unknown keys are ignored, missing keys are an error."""
        missing = [name for name in GENE_SPECS if name not in data]
        if missing:
            raise GenomeError(f"Genome is missing genes: {', '.join(missing)}")
        values: dict[str, Any] = {}
        for name, spec in GENE_SPECS.items():
            try:
                values[name] = int(data[name]) if spec.is_int else float(data[name])
            except (TypeError, ValueError) as exc:
                raise GenomeError(f"Gene {name!r} has invalid value {data[name]!r}") from exc
        return cls(**values)

    # ------------------------------------------------------------------ #
    # Reproduction
    # ------------------------------------------------------------------ #
    def mutate(self, mutation_rate: float, rng: random.Random | None = None) -> "TradingGenome":
        """
        Return a mutated DEEP CLONE of this genome.

        Every gene drifts by Gaussian noise:  new = old * (1 + N(0, mutation_rate)).
        With mutation_rate=0.15 a 20-bar SMA typically lands between 17 and 23.
        Values are then clamped to GENE_SPECS and paired genes are repaired so the
        child is always a valid, tradeable strategy. The parent is never touched.
        """
        if not 0.0 < mutation_rate < 1.0:
            raise GenomeError(f"mutation_rate must be in (0, 1), got {mutation_rate}")
        rng = rng or random.Random()
        parent_values = asdict(self)  # asdict() deep-copies -> parent DNA cannot be aliased
        child_values = {
            name: float(value) * (1.0 + rng.gauss(0.0, mutation_rate))
            for name, value in parent_values.items()
        }
        return TradingGenome(**_repair(child_values))

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def diff(self, other: "TradingGenome") -> dict[str, tuple[float | int, float | int]]:
        """Genes that differ: {gene: (self_value, other_value)}."""
        return {
            f.name: (getattr(self, f.name), getattr(other, f.name))
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        }

    def fingerprint(self) -> str:
        """Short DNA hash - handy for spotting identical genomes on screen."""
        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:6].upper()

    @property
    def warmup_bars(self) -> int:
        """Bars needed before every indicator produces a valid value."""
        return max(self.slow_sma, self.macd_slow + self.macd_signal, self.rsi_period + 1) + 1


def _repair(values: dict[str, float]) -> dict[str, float | int]:
    """Clamp every gene to its bounds and fix paired-gene ordering constraints."""
    fixed: dict[str, float | int] = {name: GENE_SPECS[name].clamp(values[name]) for name in GENE_SPECS}

    # fast SMA must stay meaningfully faster than slow SMA
    if fixed["fast_sma"] + _MIN_SMA_GAP > fixed["slow_sma"]:
        fixed["slow_sma"] = GENE_SPECS["slow_sma"].clamp(fixed["fast_sma"] + _MIN_SMA_GAP)
        fixed["fast_sma"] = GENE_SPECS["fast_sma"].clamp(min(fixed["fast_sma"], fixed["slow_sma"] - _MIN_SMA_GAP))

    # MACD fast EMA must stay faster than MACD slow EMA
    if fixed["macd_fast"] + _MIN_MACD_GAP > fixed["macd_slow"]:
        fixed["macd_slow"] = GENE_SPECS["macd_slow"].clamp(fixed["macd_fast"] + _MIN_MACD_GAP)
        fixed["macd_fast"] = GENE_SPECS["macd_fast"].clamp(min(fixed["macd_fast"], fixed["macd_slow"] - _MIN_MACD_GAP))

    # Oversold must remain below overbought (the bounds already guarantee this,
    # but keep the guard in case someone widens GENE_SPECS).
    if fixed["rsi_oversold"] + _MIN_RSI_GAP > fixed["rsi_overbought"]:
        fixed["rsi_oversold"] = GENE_SPECS["rsi_oversold"].clamp(fixed["rsi_overbought"] - _MIN_RSI_GAP)

    return fixed
