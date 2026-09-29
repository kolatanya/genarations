"""
Rule trees - bots that invent their own trading rules (genetic programming).

A rule is a small tree:

        AND
       /   \\
   rsi_14   OR
    < q.12  /  \\
       vwap_dev  NOT
        < q.20    |
               fomc_day > .5

Leaves compare one feature against a threshold (stored as a training-period
quantile, see features.py). Branches combine with AND / OR / NOT.

Evolution can: nudge a threshold, flip a comparison, swap the feature, grow or
prune a branch, and SWAP SUBTREES between two parents (crossover). Each leaf
carries its own mutation step size (self-adaptive), so evolution learns which
thresholds to fine-tune and which to explore.
"""

from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Union

import numpy as np

from lab.features import BINARY_FEATURES, FEATURE_NAMES, FeatureSet

_TAU = 1 / math.sqrt(2)  # learning rate for self-adaptive step sizes


@dataclass(frozen=True)
class Schema:
    """Which features rule leaves can use (the Lab's intraday set by default)."""
    names: list
    binary: frozenset
    plain: frozenset = frozenset()   # features whose thresholds are shown as plain numbers, not %


_SCHEMA = Schema(list(FEATURE_NAMES), frozenset(BINARY_FEATURES),
                 frozenset({"vix", "tod", "vol_z", "news_count"}))


@contextmanager
def schema(names, binary=frozenset(), plain=frozenset()):
    """Temporarily evolve rules over a different feature set (e.g. the daily Forecaster)."""
    global _SCHEMA
    old = _SCHEMA
    _SCHEMA = Schema(list(names), frozenset(binary), frozenset(plain))
    try:
        yield _SCHEMA
    finally:
        _SCHEMA = old


def feature_names() -> list:
    return _SCHEMA.names


@dataclass(frozen=True)
class Cond:
    feature: int
    greater: bool          # True: feature > threshold, False: feature < threshold
    q: float               # threshold as a quantile of the feature (0..1)
    sigma: float = 0.08    # this leaf's own mutation step size


@dataclass(frozen=True)
class And:
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Or:
    left: "Node"
    right: "Node"


@dataclass(frozen=True)
class Not:
    child: "Node"


Node = Union[Cond, And, Or, Not]


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def random_cond(rng: random.Random) -> Cond:
    f = rng.randrange(len(_SCHEMA.names))
    if _SCHEMA.names[f] in _SCHEMA.binary:
        return Cond(f, rng.random() < 0.5, 0.5, 0.05)
    # Extreme quantiles make selective rules; the middle makes noisy ones.
    q = rng.choice((rng.uniform(0.02, 0.3), rng.uniform(0.7, 0.98), rng.uniform(0.3, 0.7)))
    return Cond(f, rng.random() < 0.5, round(q, 4), 0.08)


def random_tree(rng: random.Random, max_depth: int, p_leaf: float = 0.35) -> Node:
    if max_depth <= 1 or rng.random() < p_leaf:
        return random_cond(rng)
    kind = rng.random()
    if kind < 0.1:
        return Not(random_tree(rng, max_depth - 1, p_leaf + 0.2))
    op = And if kind < 0.7 else Or
    return op(random_tree(rng, max_depth - 1, p_leaf + 0.2), random_tree(rng, max_depth - 1, p_leaf + 0.2))


def cond(name: str, greater: bool, q: float) -> Cond:
    """Convenience for hand-written seed rules."""
    return Cond(_SCHEMA.names.index(name), greater, q)


# --------------------------------------------------------------------------- #
# Inspection
# --------------------------------------------------------------------------- #
def size(node: Node) -> int:
    if isinstance(node, Cond):
        return 1
    if isinstance(node, Not):
        return 1 + size(node.child)
    return 1 + size(node.left) + size(node.right)


def depth(node: Node) -> int:
    if isinstance(node, Cond):
        return 1
    if isinstance(node, Not):
        return 1 + depth(node.child)
    return 1 + max(depth(node.left), depth(node.right))


def leaves(node: Node) -> list[Cond]:
    if isinstance(node, Cond):
        return [node]
    if isinstance(node, Not):
        return leaves(node.child)
    return leaves(node.left) + leaves(node.right)


def key(node: Node | None) -> str:
    """Canonical text used to detect duplicate / converged genomes."""
    if node is None:
        return "-"
    if isinstance(node, Cond):
        return f"{node.feature}{'>' if node.greater else '<'}{node.q:.2f}"
    if isinstance(node, Not):
        return f"!({key(node.child)})"
    a, b = sorted((key(node.left), key(node.right)))
    return f"({a}{'&' if isinstance(node, And) else '|'}{b})"


def describe(node: Node | None, fs: FeatureSet | None = None) -> str:
    """Human-readable rule, e.g. 'rsi_14 < 28.4 AND vwap_dev < -0.21%'."""
    if node is None:
        return "(never)"
    if isinstance(node, Cond):
        name = _SCHEMA.names[node.feature]
        op = ">" if node.greater else "<"
        if name in _SCHEMA.binary:
            return f"{name}" if node.greater else f"not {name}"
        if fs is not None and fs.quantiles is not None:
            value = fs.threshold(node.feature, node.q)
            shown = f"{value:.3%}" if abs(value) < 0.5 and name not in _SCHEMA.plain \
                and not name.startswith(("rsi", "rs_rank")) else f"{value:.3g}"
            return f"{name} {op} {shown}"
        return f"{name} {op} q{node.q:.2f}"
    if isinstance(node, Not):
        return f"NOT ({describe(node.child, fs)})"
    word = "AND" if isinstance(node, And) else "OR"

    def wrap(n: Node) -> str:
        text = describe(n, fs)
        return f"({text})" if isinstance(n, (And, Or)) and type(n) is not type(node) else text

    return f"{wrap(node.left)} {word} {wrap(node.right)}"


# --------------------------------------------------------------------------- #
# Evaluation (vectorised, with a per-generation cache)
# --------------------------------------------------------------------------- #
class TreeEvaluator:
    """Evaluates rule trees on a FeatureSet window, caching leaf results."""

    def __init__(self, fs: FeatureSet, t0: int = 0, t1: int | None = None, max_cache: int = 256) -> None:
        self.fs = fs
        self.t0, self.t1 = t0, t1 if t1 is not None else fs.values.shape[2]
        self._cache: dict[tuple, np.ndarray] = {}
        self.max_cache = max_cache

    def leaf(self, c: Cond) -> np.ndarray:
        k = (c.feature, c.greater, round(c.q, 4))
        hit = self._cache.get(k)
        if hit is None:
            values = self.fs.values[c.feature, :, self.t0:self.t1]
            thr = self.fs.threshold(c.feature, c.q)
            with np.errstate(invalid="ignore"):
                hit = values > thr if c.greater else values < thr   # NaN -> False
            if len(self._cache) >= self.max_cache:
                self._cache.clear()
            self._cache[k] = hit
        return hit

    def __call__(self, node: Node | None) -> np.ndarray:
        if node is None:
            return np.zeros((self.fs.values.shape[1], self.t1 - self.t0), dtype=bool)
        if isinstance(node, Cond):
            return self.leaf(node)
        if isinstance(node, Not):
            return ~self(node.child)
        a, b = self(node.left), self(node.right)
        return (a & b) if isinstance(node, And) else (a | b)


def simplify(node: Node) -> Node:
    """Remove redundancy so a rule reads like a reason: 'A OR A' -> 'A', 'NOT NOT A' -> 'A'."""
    if isinstance(node, Cond):
        return node
    if isinstance(node, Not):
        child = simplify(node.child)
        return child.child if isinstance(child, Not) else Not(child)
    left, right = simplify(node.left), simplify(node.right)
    if key(left) == key(right):
        return left
    op = type(node)
    # (A op B) op A  ->  A op B   (the repeated part adds nothing)
    for inner, other in ((left, right), (right, left)):
        if isinstance(inner, (And, Or)) and key(other) in (key(inner.left), key(inner.right)):
            # same op: (A op B) op A -> A op B.  Mixed (absorption): (A AND B) OR A -> A, (A OR B) AND A -> A
            return inner if isinstance(inner, op) else other
    return op(left, right)


# --------------------------------------------------------------------------- #
# Variation
# --------------------------------------------------------------------------- #
def _mutate_cond(c: Cond, rng: random.Random) -> Cond:
    sigma = min(max(c.sigma * math.exp(_TAU * rng.gauss(0, 1)), 0.005), 0.3)
    roll = rng.random()
    if roll < 0.08:
        return random_cond(rng)                                  # brand-new sense
    if roll < 0.14:
        return replace(c, greater=not c.greater, sigma=sigma)    # flip the comparison
    if _SCHEMA.names[c.feature] in _SCHEMA.binary:
        return replace(c, greater=not c.greater if rng.random() < 0.3 else c.greater)
    q = min(max(c.q + rng.gauss(0, sigma), 0.005), 0.995)
    return replace(c, q=round(q, 4), sigma=sigma)


def mutate(node: Node, rng: random.Random, rate: float, max_depth: int) -> Node:
    """Return a mutated copy. `rate` scales how much changes (adaptive mutation)."""
    def walk(n: Node, d: int) -> Node:
        if rng.random() < 0.04 * rate / 0.15 and d < max_depth:
            return random_tree(rng, max_depth - d + 1)          # grow a fresh branch
        if isinstance(n, Cond):
            return _mutate_cond(n, rng) if rng.random() < min(0.9, rate * 3) else n
        if isinstance(n, Not):
            if rng.random() < 0.05:
                return n.child                                  # prune the NOT
            return Not(walk(n.child, d + 1))
        if rng.random() < 0.05:
            return rng.choice((n.left, n.right))                # prune a branch
        op = type(n)
        if rng.random() < 0.06:
            op = Or if op is And else And                       # AND <-> OR
        return op(walk(n.left, d + 1), walk(n.right, d + 1))

    out = walk(node, 1)
    return out if depth(out) <= max_depth else _trim(out, max_depth)


def _trim(node: Node, max_depth: int) -> Node:
    if max_depth <= 1 or isinstance(node, Cond):
        return node if isinstance(node, Cond) else leaves(node)[0]
    if isinstance(node, Not):
        return Not(_trim(node.child, max_depth - 1))
    return type(node)(_trim(node.left, max_depth - 1), _trim(node.right, max_depth - 1))


def _subtrees(node: Node, path: tuple = ()) -> list[tuple[tuple, Node]]:
    out = [(path, node)]
    if isinstance(node, Not):
        out += _subtrees(node.child, path + (0,))
    elif isinstance(node, (And, Or)):
        out += _subtrees(node.left, path + (0,)) + _subtrees(node.right, path + (1,))
    return out


def _replace_at(node: Node, path: tuple, new: Node) -> Node:
    if not path:
        return new
    if isinstance(node, Not):
        return Not(_replace_at(node.child, path[1:], new))
    if path[0] == 0:
        return type(node)(_replace_at(node.left, path[1:], new), node.right)
    return type(node)(node.left, _replace_at(node.right, path[1:], new))


def crossover(a: Node, b: Node, rng: random.Random, max_depth: int) -> Node:
    """Graft a random branch of parent B onto a random spot in parent A."""
    path, _ = rng.choice(_subtrees(a))
    _, donor = rng.choice(_subtrees(b))
    child = _replace_at(a, path, donor)
    return child if depth(child) <= max_depth else _trim(child, max_depth)


def wobble(node: Node, rng: random.Random, pct: float) -> Node:
    """Nudge every threshold a little (plateau test) - same structure, slightly different numbers."""
    if isinstance(node, Cond):
        if _SCHEMA.names[node.feature] in _SCHEMA.binary:
            return node
        return replace(node, q=min(max(node.q + rng.gauss(0, pct * 0.5), 0.005), 0.995))
    if isinstance(node, Not):
        return Not(wobble(node.child, rng, pct))
    return type(node)(wobble(node.left, rng, pct), wobble(node.right, rng, pct))


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def to_json(node: Node | None):
    if node is None:
        return None
    if isinstance(node, Cond):
        return {"f": _SCHEMA.names[node.feature], "gt": node.greater, "q": node.q, "s": node.sigma}
    if isinstance(node, Not):
        return {"not": to_json(node.child)}
    return {"and" if isinstance(node, And) else "or": [to_json(node.left), to_json(node.right)]}


def from_json(obj) -> Node | None:
    if obj is None:
        return None
    if "f" in obj:
        return Cond(_SCHEMA.names.index(obj["f"]), bool(obj["gt"]), float(obj["q"]), float(obj.get("s", 0.08)))
    if "not" in obj:
        return Not(from_json(obj["not"]))
    op, parts = (And, obj["and"]) if "and" in obj else (Or, obj["or"])
    return op(from_json(parts[0]), from_json(parts[1]))
