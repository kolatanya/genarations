"""
Ensemble ("team vote") with a market-regime detector.

Every morning the market is labelled UP-trend, SIDEWAYS or DOWN-trend from
SPY's position versus its 20-day average (known at the open - no hindsight).
Each member's vote is weighted by how well that member historically did in
TODAY's regime, so trend specialists lead on trending days and range
specialists lead on choppy days. A trade happens when the weighted votes
reach the quorum (default: half).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from lab.fitness import REGIMES, Scorer
from lab.genome import PARAM_SPECS, LabGenome
from lab.simulator import SimResult, run_signals


@dataclass
class Member:
    bot_id: int
    genome: LabGenome
    weights: tuple[float, float, float]   # vote strength on UP / SIDEWAYS / DOWN days
    val_sharpe: float = 0.0

    def to_json(self) -> dict:
        return {"bot_id": self.bot_id, "genome": self.genome.to_json(), "weights": list(self.weights),
                "val_sharpe": self.val_sharpe}

    @classmethod
    def from_json(cls, obj: dict) -> "Member":
        return cls(int(obj["bot_id"]), LabGenome.from_json(obj["genome"]), tuple(obj["weights"]),
                   float(obj.get("val_sharpe", 0.0)))


class Ensemble:
    def __init__(self, members: list[Member], quorum: float = config.LAB_ENSEMBLE_QUORUM) -> None:
        if not members:
            raise ValueError("An ensemble needs at least one member")
        self.members = members
        self.quorum = quorum
        w = np.array([np.mean(m.weights) for m in members])
        w = w / w.sum()
        # Risk settings: vote-weighted average of the members' genes
        self.params = {k: float(sum(wi * m.genome.params[k] for wi, m in zip(w, members))) for k in PARAM_SPECS}
        for k in ("max_hold", "start_min", "end_min"):
            self.params[k] = float(round(self.params[k]))
        self.avoid_events = sum(wi for wi, m in zip(w, members) if m.genome.avoid_events) >= 0.5

    def vote(self, member_signals: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
             regime: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine per-member (long, short, exit) arrays [S, T] using regime-weighted votes."""
        W = np.array([m.weights for m in self.members], dtype=float)   # [M, 3]
        w_bar = W[:, regime]                                          # [M, T]
        need = self.quorum * w_bar.sum(axis=0)                        # [T]
        out = []
        for j in range(3):
            votes = sum(w_bar[i][None, :] * sig[j] for i, sig in enumerate(member_signals))
            out.append(votes >= need[None, :] - 1e-12)
        long_sig, short_sig, exit_sig = out
        both = long_sig & short_sig          # conflicting majorities -> stand aside
        return long_sig & ~both, short_sig & ~both, exit_sig

    def simulate(self, scorer: Scorer, t0: int, t1: int, cost_multiplier: float = 1.0) -> SimResult:
        ev = scorer.evaluator(t0, t1)
        sigs = [(ev(m.genome.long_entry), ev(m.genome.short_entry), ev(m.genome.exit_rule)) for m in self.members]
        long_sig, short_sig, exit_sig = self.vote(sigs, scorer.bar_regime[t0:t1])
        return run_signals(scorer.data, scorer.fs, long_sig, short_sig, exit_sig, self.params, self.avoid_events,
                           t0, t1, cost_multiplier=cost_multiplier)

    def describe_weights(self) -> list[str]:
        return [f"Bot #{m.bot_id}: " + ", ".join(f"{r.lower()} {w:.1f}" for r, w in zip(REGIMES, m.weights))
                for m in self.members]

    def to_json(self) -> dict:
        return {"quorum": self.quorum, "members": [m.to_json() for m in self.members]}

    @classmethod
    def from_json(cls, obj: dict) -> "Ensemble":
        return cls([Member.from_json(m) for m in obj["members"]], float(obj.get("quorum", config.LAB_ENSEMBLE_QUORUM)))
