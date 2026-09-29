"""
WHY did a prophet make that call?  Every call comes from an explicit rule, so it
can always be explained and replicated by hand:

  reason()    - for one stock on one day: which conditions fired, with the real numbers
                e.g.  BIG because  earn_next ✓  (earnings reaction tomorrow)
  playbook()  - the rule's track record: how often each condition fires, and how often
                the answer really was YES when it did (only already-known answers)
"""

from __future__ import annotations

import numpy as np

from lab import gp
from lab.features import FeatureSet


def _value(name: str, v: float) -> str:
    sch = gp._SCHEMA
    if not np.isfinite(v):
        return "n/a"
    if name in sch.binary:
        return "yes" if v > 0.5 else "no"
    if abs(v) < 0.5 and name not in sch.plain and not name.startswith(("rsi", "rs_rank")):
        return f"{v:.2%}"
    return f"{v:.3g}"


def reason(node: gp.Node, fs: FeatureSet, s: int, t: int) -> tuple[bool, list[str]]:
    """Evaluate the rule for stock s on day t and return (answer, the facts that decided it)."""
    if isinstance(node, gp.Cond):
        name = fs.names[node.feature]
        v = float(fs.values[node.feature, s, t])
        thr = fs.threshold(node.feature, node.q)
        hit = bool(v > thr if node.greater else v < thr) if np.isfinite(v) else False
        mark = "✓" if hit else "✗"
        if name in gp._SCHEMA.binary:
            return hit, [f"{gp.describe(node, fs)} {mark}"]
        return hit, [f"{gp.describe(node, fs)} {mark} (now {_value(name, v)})"]
    if isinstance(node, gp.Not):
        hit, why = reason(node.child, fs, s, t)
        facts = " / ".join(w.replace(" ✓", "").replace(" ✗", "") for w in why)
        return not hit, [f"NOT ({facts}) {'✓' if not hit else '✗'}"]
    (a, wa), (b, wb) = reason(node.left, fs, s, t), reason(node.right, fs, s, t)
    if isinstance(node, gp.And):
        if a and b:
            return True, wa + wb
        return False, (wa if not a else []) + (wb if not b else [])
    if a or b:
        return True, (wa if a else []) + (wb if b else [])
    return False, wa + wb


def playbook(node: gp.Node, fs: FeatureSet, label: np.ndarray, valid: np.ndarray, t: int,
             days: int = 250) -> dict:
    """Track record over the `days` days before t whose answers are already known (columns < t)."""
    a = max(0, t - days)
    ok = valid[:, a:t]
    lab = label[:, a:t]
    ev = gp.TreeEvaluator(fs, a, t)
    pred = ev(node).astype(bool)
    n = int(ok.sum())
    out = {"days": t - a, "n": n, "base": float(lab[ok].mean()) if n else float("nan"), "leaves": []}
    if not n:
        return out
    says = pred & ok
    out["says_yes"] = float(says.sum() / n)
    out["right_yes"] = float(lab[says].mean()) if says.any() else float("nan")
    out["right_no"] = float((~lab[~pred & ok]).mean()) if (~pred & ok).any() else float("nan")
    out["accuracy"] = float((pred == lab)[ok].mean())
    seen = set()
    for c in gp.leaves(node):
        k = gp.key(c)
        if k in seen:
            continue
        seen.add(k)
        fires = ev.leaf(c) & ok
        out["leaves"].append({"text": gp.describe(c, fs), "fires": float(fires.sum() / n),
                              "yes_when_fires": float(lab[fires].mean()) if fires.sum() >= 30 else float("nan"),
                              "cases": int(fires.sum())})
    return out


# --------------------------------------------------------------------------- #
# CONFIDENCE - how sure is the prophet about THIS call?
# --------------------------------------------------------------------------- #
TIERS = (("HIGH", 0.60), ("MEDIUM", 0.54), ("LOW", 0.0))


def tier(conf: float) -> str:
    return next(name for name, floor in TIERS if conf >= floor)


class ConfidenceTable:
    """
    Every call is made for a reason - a particular combination of the rule's conditions being
    true or false. The confidence of a call is how often calls made for exactly that reason were
    right over the past `days` days (answers already known before day t). Rare combinations
    (fewer than `min_cases`) fall back to the rule's overall hit rate for that answer.
    """

    MAX_LEAVES = 12

    def __init__(self, tree: gp.Node, fs: FeatureSet, label: np.ndarray, valid: np.ndarray, t: int,
                 days: int = 250, min_cases: int = 100) -> None:
        self.tree = tree
        uniq = {}
        for c in gp.leaves(tree):
            uniq.setdefault(gp.key(c), c)
        self.leaves = list(uniq.values())[:self.MAX_LEAVES]
        a = max(0, t - days)
        ev = gp.TreeEvaluator(fs, a, t)
        pred = ev(tree).astype(bool)
        ok = valid[:, a:t]
        right = pred == label[:, a:t]
        self.fallback = {True: _mean(right[ok & pred]), False: _mean(right[ok & ~pred])}
        size = 1 << len(self.leaves)
        sig = self._signature(ev)[ok]
        n = np.bincount(sig, minlength=size).astype(float)
        hits = np.bincount(sig, weights=right[ok].astype(float), minlength=size)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.table = np.where(n >= min_cases, hits / n, np.nan)
        self.cases = n

    def _signature(self, ev: gp.TreeEvaluator) -> np.ndarray:
        sig = np.zeros(ev(self.leaves[0]).shape, dtype=np.int64)
        for i, c in enumerate(self.leaves):
            sig |= ev.leaf(c).astype(np.int64) << i
        return sig

    def calls(self, fs: FeatureSet, t: int) -> tuple[np.ndarray, np.ndarray]:
        """(call, confidence) for every stock on day t."""
        ev = gp.TreeEvaluator(fs, t, t + 1)
        pred = ev(self.tree)[:, 0].astype(bool)
        conf = self.table[self._signature(ev)[:, 0]]
        fb = np.where(pred, self.fallback[True], self.fallback[False])
        return pred, np.where(np.isfinite(conf), conf, fb)


def _mean(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else 0.5


def sample_reasons(node: gp.Node, fs: FeatureSet, symbols: list[str], t: int, k: int = 6) -> list[tuple[str, bool, list[str]]]:
    """A few stocks with their call and reason: YES calls first (the interesting ones), then some NO calls."""
    calls = [(sym, *reason(node, fs, s, t)) for s, sym in enumerate(symbols)]
    yes = [c for c in calls if c[1]]
    no = [c for c in calls if not c[1]]
    half = k // 2 if yes and no else k
    return (yes[:half] + no[:k - min(half, len(yes))])[:k]
