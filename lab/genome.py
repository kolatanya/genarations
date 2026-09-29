"""
LabGenome - the v2 DNA: rules the bot INVENTS plus its risk personality.

  long_entry   rule tree -> buy when true (None = never goes long)
  short_entry  rule tree -> sell short when true (None = never shorts)
  exit_rule    rule tree -> close the position when true
  params       stops/targets in ATRs, risk per trade, trading hours, daily loss limit
  sigmas       each numeric gene's OWN mutation step (self-adaptive evolution)
  avoid_events skip earnings days and Fed days
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field

import config
from lab import gp
from lab.features import FeatureSet

# name: (low, high, textbook value, is_int)
PARAM_SPECS: dict[str, tuple[float, float, float, bool]] = {
    "stop_atr":   (0.5, 6.0, 2.0, False),     # stop-loss distance in ATRs
    "tp_atr":     (0.5, 12.0, 3.0, False),    # take-profit distance in ATRs
    "trail_atr":  (0.5, 8.0, 2.5, False),     # trailing stop distance in ATRs
    "max_hold":   (3, 78, 24, True),          # bars before a stale trade is closed
    "risk_pct":   (0.001, 0.02, 0.005, False),  # equity risked per trade (volatility sizing)
    "start_min":  (0, 300, 5, True),          # first minute after the open it may enter
    "end_min":    (60, 380, 360, True),       # last minute it may enter
    "daily_loss": (0.005, 0.05, 0.02, False),  # stop trading for the day after this loss
}
_TAU = 1 / math.sqrt(2 * len(PARAM_SPECS))


def _clamp(name: str, value: float) -> float:
    lo, hi, _, is_int = PARAM_SPECS[name]
    value = min(max(value, lo), hi)
    return float(round(value)) if is_int else round(value, 5)


@dataclass(frozen=True)
class LabGenome:
    long_entry: gp.Node | None
    short_entry: gp.Node | None
    exit_rule: gp.Node | None
    params: dict[str, float] = field(default_factory=dict)
    sigmas: dict[str, float] = field(default_factory=dict)
    avoid_events: bool = True

    # ------------------------------------------------------------------ #
    @classmethod
    def textbook(cls) -> "LabGenome":
        """Buy oversold dips in an up-trending stock; sell when overbought."""
        return cls(
            long_entry=gp.And(gp.cond("rsi_14", False, 0.10), gp.cond("d1_trend", True, 0.5)),
            short_entry=None,
            exit_rule=gp.cond("rsi_14", True, 0.85),
            params={k: float(v[2]) for k, v in PARAM_SPECS.items()},
            sigmas={k: 0.15 for k in PARAM_SPECS},
        )

    @classmethod
    def random(cls, rng: random.Random) -> "LabGenome":
        depth = config.LAB_MAX_TREE_DEPTH
        style = rng.random()
        long_e = gp.random_tree(rng, depth) if style < 0.85 else None
        short_e = gp.random_tree(rng, depth) if (style > 0.6 or long_e is None) else None
        params = {k: _clamp(k, rng.uniform(lo, hi) if rng.random() < 0.5 else default * math.exp(rng.gauss(0, 0.3)))
                  for k, (lo, hi, default, _) in PARAM_SPECS.items()}
        return cls(long_e, short_e, gp.random_tree(rng, depth - 1), params,
                   {k: 0.15 for k in PARAM_SPECS}, rng.random() < 0.7)._repaired()

    # ------------------------------------------------------------------ #
    def _repaired(self) -> "LabGenome":
        p = dict(self.params)
        if p["end_min"] < p["start_min"] + 30:
            p["end_min"] = _clamp("end_min", p["start_min"] + 30)
            p["start_min"] = _clamp("start_min", min(p["start_min"], p["end_min"] - 30))
        long_e, short_e = self.long_entry, self.short_entry
        if long_e is None and short_e is None:
            long_e = gp.cond("rsi_14", False, 0.1)
        return LabGenome(long_e, short_e, self.exit_rule, p, dict(self.sigmas), self.avoid_events)

    def mutate(self, rng: random.Random, rate: float) -> "LabGenome":
        """Self-adaptive mutation: each gene's step size evolves along with the gene."""
        depth = config.LAB_MAX_TREE_DEPTH
        params, sigmas = {}, {}
        shared = rng.gauss(0, 1)
        for k, v in self.params.items():
            s = self.sigmas.get(k, 0.15) * math.exp(_TAU * shared + _TAU * rng.gauss(0, 1))
            s = min(max(s, 0.01), 0.6)
            sigmas[k] = s
            if rng.random() < min(1.0, rate * 4):
                v = v * math.exp(rng.gauss(0, s * rate / 0.15))
            params[k] = _clamp(k, v)

        def maybe(tree: gp.Node | None, allow_none: bool) -> gp.Node | None:
            if tree is None:
                return gp.random_tree(rng, depth) if (allow_none and rng.random() < 0.05) else None
            if allow_none and rng.random() < 0.03:
                return None  # drop this side entirely
            return gp.mutate(tree, rng, rate, depth)

        avoid = (not self.avoid_events) if rng.random() < 0.05 else self.avoid_events
        return LabGenome(maybe(self.long_entry, True), maybe(self.short_entry, True),
                         maybe(self.exit_rule, False) or gp.random_cond(rng), params, sigmas, avoid)._repaired()

    @staticmethod
    def crossover(a: "LabGenome", b: "LabGenome", rng: random.Random) -> "LabGenome":
        """Child mixes both parents: genes picked from either, rule branches grafted across."""
        depth = config.LAB_MAX_TREE_DEPTH

        def mix(x, y):
            if x is None or y is None:
                return x if rng.random() < 0.5 else y
            return gp.crossover(x, y, rng, depth)

        pick = {k: (a if rng.random() < 0.5 else b) for k in PARAM_SPECS}
        return LabGenome(
            mix(a.long_entry, b.long_entry), mix(a.short_entry, b.short_entry),
            mix(a.exit_rule, b.exit_rule) or a.exit_rule,
            {k: pick[k].params[k] for k in PARAM_SPECS}, {k: pick[k].sigmas.get(k, 0.15) for k in PARAM_SPECS},
            a.avoid_events if rng.random() < 0.5 else b.avoid_events,
        )._repaired()

    def wobble(self, rng: random.Random, pct: float) -> "LabGenome":
        """Plateau test variant: every number nudged ~pct, structure unchanged."""
        params = {k: _clamp(k, v * (1 + rng.gauss(0, pct))) for k, v in self.params.items()}
        w = (lambda t: gp.wobble(t, rng, pct) if t is not None else None)
        return LabGenome(w(self.long_entry), w(self.short_entry), w(self.exit_rule),
                         params, dict(self.sigmas), self.avoid_events)._repaired()

    # ------------------------------------------------------------------ #
    @property
    def complexity(self) -> int:
        return sum(gp.size(t) for t in (self.long_entry, self.short_entry, self.exit_rule) if t is not None)

    @property
    def style(self) -> str:
        if self.long_entry is not None and self.short_entry is not None:
            return "LONG+SHORT"
        return "LONG" if self.long_entry is not None else "SHORT"

    def key(self) -> str:
        p = ",".join(f"{k}={v:.3g}" for k, v in sorted(self.params.items()))
        return f"L{gp.key(self.long_entry)}|S{gp.key(self.short_entry)}|X{gp.key(self.exit_rule)}|{p}|{self.avoid_events}"

    def fingerprint(self) -> str:
        return hashlib.sha1(self.key().encode()).hexdigest()[:6].upper()

    def describe(self, fs: FeatureSet | None = None) -> dict[str, str]:
        p = self.params
        return {
            "BUY when": gp.describe(self.long_entry, fs),
            "SHORT when": gp.describe(self.short_entry, fs),
            "EXIT when": gp.describe(self.exit_rule, fs),
            "Risk": (f"stop {p['stop_atr']:.1f} ATR · target {p['tp_atr']:.1f} ATR · trail {p['trail_atr']:.1f} ATR · "
                     f"max hold {int(p['max_hold']) * 5} min · risk {p['risk_pct']:.2%}/trade · "
                     f"daily loss limit {p['daily_loss']:.1%}"),
            "Hours": (f"enters {_clock(p['start_min'])}-{_clock(p['end_min'])} ET · "
                      f"{'skips' if self.avoid_events else 'trades'} earnings & Fed days"),
        }

    def to_json(self) -> dict:
        return {"long_entry": gp.to_json(self.long_entry), "short_entry": gp.to_json(self.short_entry),
                "exit_rule": gp.to_json(self.exit_rule), "params": self.params, "sigmas": self.sigmas,
                "avoid_events": self.avoid_events, "dna": self.fingerprint()}

    @classmethod
    def from_json(cls, obj: dict) -> "LabGenome":
        params = {k: _clamp(k, float(obj["params"].get(k, v[2]))) for k, v in PARAM_SPECS.items()}
        return cls(gp.from_json(obj.get("long_entry")), gp.from_json(obj.get("short_entry")),
                   gp.from_json(obj.get("exit_rule")), params,
                   {k: float(obj.get("sigmas", {}).get(k, 0.15)) for k in PARAM_SPECS},
                   bool(obj.get("avoid_events", True)))._repaired()

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return json.dumps(self.to_json())


def _clock(minutes_after_open: float) -> str:
    total = 9 * 60 + 30 + int(minutes_after_open)
    return f"{total // 60}:{total % 60:02d}"
