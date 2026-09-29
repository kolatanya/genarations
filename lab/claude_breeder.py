"""
CLAUDE BREEDER - an AI designs some of the Prophecy League's children.

Random mutation is blind: it nudges thresholds and swaps senses by dice roll.
The Claude breeder instead reads the evidence a human researcher would look at:

  * the survivor's rule and how every prophet in the last round scored
  * the survivor's accuracy over the past year, when it says YES vs NO
  * for every sense: how often the answer was YES when that sense was very
    low / very high, and how accurate the survivor was in those situations
  * which senses are MARKET-WIDE (same value for every stock) - a rule made
    only of those says the same thing for all 500 stocks

...and designs targeted children ("the survivor is wrong most when X is high,
so try adding NOT X > ..."). Claude's children compete in the same league as
blind random mutants, so the scoreboard shows whether the AI actually breeds
better prophets or not.

No peeking: the evidence window ends at the day being judged, and only uses
answers that were already known that evening.

Needs ANTHROPIC_API_KEY in .env. Any failure (no key, network, bad reply) falls
back to ordinary random mutation - the league never stops because of it.
"""

from __future__ import annotations

import json
import logging
import re
import warnings

import numpy as np

import config
from lab import gp
from lab.forecast import DAILY_FEATURES, TARGETS, WARMUP_DAYS, describe_tree

log = logging.getLogger(__name__)

EVIDENCE_DAYS = 250           # one trading year of already-known answers
_DESCRIPTIONS = dict(DAILY_FEATURES)

SYSTEM = """You are the breeder in a survival-of-the-fittest prediction league.
Every trading day, each prophet (a small rule tree) makes one call per stock for TOMORROW.
The prophet with the most correct calls over a 5-day round survives; you design some of its children.

A rule tree is JSON:
  leaf:     {"f": "<sense name>", "gt": true|false, "q": <0..1>}
            means: sense > (gt=true) or < (gt=false) the value at quantile q of that sense's history.
            For binary senses use q 0.5 ("gt": true = the flag is on, false = off).
  branches: {"and": [A, B]}  {"or": [A, B]}  {"not": A}
The rule says the YES answer when the tree is true, otherwise NO.
Maximum depth is %(depth)d levels (a leaf is depth 1).

Design principles:
- Each child should test ONE clear idea grounded in the evidence (fix a weakness, sharpen a threshold,
  add a filter that separates right from wrong calls). Small, careful edits usually beat rewrites.
- Rules made only of MARKET-WIDE senses give the same call for every stock; stock-specific senses
  let a prophet say YES for some stocks and NO for others, which is how it can beat the naive guess.
- The crown goes to the most POINTS over the last 60 days: 2 points per percentage point of accuracy
  above the naive guess, plus 1 point per 10%% of calls that are YES (max 5 at 50%%). Accuracy counts
  double, but a rule that makes the YES call often (when it has a reason to) is rewarded.
- A rule that gives the same answer on more than 97%% of calls has no reason and cannot win.
- Make the children different from each other and from the survivor.

Reply with ONLY a JSON object: {"children": [{"why": "<one short sentence>", "rule": <tree>}, ...]}"""


class ClaudeBreeder:
    """Designs children with Claude. Set `.fs`, `.ans` and `.t` (the judging day) before calling design()."""

    def __init__(self, model: str | None = None, children: int | None = None, client=None) -> None:
        self.model = model or config.CLAUDE_MODEL
        self.children = config.CLAUDE_CHILDREN if children is None else children
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY, timeout=120, max_retries=2)
        self.client = client
        self.fs = None
        self.ans = None
        self.t: int | None = None
        self.calls = 0
        self.failures = 0
        self.last_error: str | None = None

    # ------------------------------------------------------------------ #
    def design(self, target: str, survivor: gp.Node, last_round: list[dict], history: list[dict]) -> list[tuple[gp.Node, str]]:
        """Return up to `children` (tree, reason) pairs. Empty list on any failure (caller falls back)."""
        if self.children <= 0 or self.fs is None or self.t is None:
            return []
        try:
            prompt = self.prompt(target, survivor, last_round, history)
            self.calls += 1
            msg = self.client.messages.create(
                model=self.model, max_tokens=4000,
                system=SYSTEM % {"depth": config.LAB_MAX_TREE_DEPTH},
                messages=[{"role": "user", "content": prompt}])
            text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", "") == "text")
            kids = parse_children(text, survivor, self.children)
            if not kids:
                raise ValueError("reply contained no usable rules")
            return kids
        except Exception as exc:          # noqa: BLE001 - the league must never stop because of the breeder
            self.failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("Claude breeder failed, falling back to random mutation: %s", self.last_error)
            return []

    # ------------------------------------------------------------------ #
    def prompt(self, target: str, survivor: gp.Node, last_round: list[dict], history: list[dict]) -> str:
        from lab.prophecy import _labels

        fs, t = self.fs, self.t
        yes, no, question = TARGETS[target]
        label, valid = _labels(self.ans, target)
        a = max(WARMUP_DAYS, t - EVIDENCE_DAYS)
        lab, ok = label[:, a:t], valid[:, a:t]          # column i is answered by day i+1's close, all known by t
        pred = gp.TreeEvaluator(fs, a, t)(survivor).astype(bool)
        right = pred == lab
        base = float(lab[ok].mean()) if ok.any() else 0.5
        naive = max(base, 1 - base)

        lines = [f"LEAGUE: {question}  YES = {yes}, NO = {no}.",
                 f"Evidence: the last {t - a} trading days ({ok.sum():,} stock-days), answers already known.",
                 f"Base rate: {yes} {base:.1%} of the time, so the naive guess scores {naive:.1%}.", "",
                 f"SURVIVOR RULE (says {yes} when true):",
                 f"  readable: {describe_tree(gp.to_json(survivor), fs)}",
                 f"  json: {json.dumps(gp.to_json(survivor))}"]
        if ok.any():
            says = pred & ok
            lines.append(f"  past year: {right[ok].mean():.1%} correct; says {yes} on {says.sum() / ok.sum():.1%} of "
                         f"stock-days; right {right[says].mean() if says.any() else float('nan'):.1%} when saying {yes}, "
                         f"{right[~pred & ok].mean() if (~pred & ok).any() else float('nan'):.1%} when saying {no}")
        if last_round:
            lines += ["", "LAST ROUND (5 days, most correct first):"]
            for r in last_round:
                lines.append(f"  {r['accuracy']:.2%}  {r['role']:<9} {r['rule']}")
        if history:
            lines += ["", "RECENT ROUND WINNERS (oldest first):"]
            for h in history[-6:]:
                lines.append(f"  round {h['round']}: {h['survivor_role']} won with {h['survivor_accuracy']:.2%}")

        lines += ["", f"SENSES. q-values: the sense's value at quantiles .1/.5/.9. {yes}% = how often the answer was "
                  f"{yes} when the sense was in its bottom / top 20%. surv% = survivor accuracy there.",
                  f"{'sense':<16}{'scope':<8}{'q.10':>10}{'q.50':>10}{'q.90':>10}   {yes}% lo/hi   surv% lo/hi   meaning"]
        for f, name in enumerate(fs.names):
            v = fs.values[f, :, a:t]
            lo_edge, hi_edge = fs.threshold(f, 0.2), fs.threshold(f, 0.8)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                across = np.nanmedian(np.nanstd(v, axis=0))           # spread between stocks on the same day
                over_time = np.nanstd(v) + 1e-12
            scope = "market" if across / over_time < 1e-3 else "stock"
            with np.errstate(invalid="ignore"):
                lo, hi = (v <= lo_edge) & ok, (v >= hi_edge) & ok
            if name in gp._SCHEMA.binary:
                lo, hi = (v <= 0.5) & ok, (v > 0.5) & ok
            cells = [f"{name:<16}{scope:<8}"] + [f" {fs.threshold(f, q):>9.4g}" for q in (0.1, 0.5, 0.9)]
            cells.append(f"   {_pct(lab, lo)}/{_pct(lab, hi)}   {_pct(right, lo)}/{_pct(right, hi)}   "
                         f"{_DESCRIPTIONS.get(name, '')}")
            lines.append("".join(cells))
        lines += ["", f"Design {self.children} children for the survivor. Use only sense names from the table."]
        return "\n".join(lines)


def _pct(arr: np.ndarray, mask: np.ndarray) -> str:
    n = int(mask.sum())
    return f"{arr[mask].mean():5.1%}" if n >= 200 else "   - "


# --------------------------------------------------------------------------- #
# Parsing Claude's reply into safe rule trees
# --------------------------------------------------------------------------- #
def parse_children(text: str, survivor: gp.Node, limit: int) -> list[tuple[gp.Node, str]]:
    """Extract valid, distinct rule trees from a reply. Invalid ones are silently dropped."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    items = data.get("children", []) if isinstance(data, dict) else []
    seen, out = {gp.key(survivor)}, []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            tree = _node(item.get("rule"))
        except (ValueError, TypeError, KeyError):
            continue
        if gp.depth(tree) > config.LAB_MAX_TREE_DEPTH:
            tree = gp._trim(tree, config.LAB_MAX_TREE_DEPTH)
        if gp.key(tree) in seen:
            continue
        seen.add(gp.key(tree))
        out.append((tree, str(item.get("why", "")).strip()[:160]))
        if len(out) >= limit:
            break
    return out


def _node(obj) -> gp.Node:
    """Strict JSON -> tree conversion (unknown senses or odd shapes raise)."""
    if not isinstance(obj, dict):
        raise ValueError("node must be an object")
    names = gp.feature_names()
    if "f" in obj:
        name = obj["f"]
        if name not in names:
            raise ValueError(f"unknown sense {name!r}")
        if name in gp._SCHEMA.binary:
            return gp.Cond(names.index(name), bool(obj.get("gt", True)), 0.5, 0.05)
        q = float(obj["q"])
        if not np.isfinite(q):
            raise ValueError("bad quantile")
        return gp.Cond(names.index(name), bool(obj.get("gt", True)), round(min(max(q, 0.005), 0.995), 4))
    if "not" in obj:
        return gp.Not(_node(obj["not"]))
    for word, op in (("and", gp.And), ("or", gp.Or)):
        if word in obj:
            parts = [_node(p) for p in obj[word]]
            if len(parts) < 2:
                return parts[0]
            tree = parts[0]
            for p in parts[1:]:
                tree = op(tree, p)
            return tree
    raise ValueError("unrecognised node")


def make_breeder(enabled: bool = True) -> tuple[ClaudeBreeder | None, str]:
    """Build the breeder if possible. Returns (breeder or None, status message)."""
    if not enabled or not config.CLAUDE_BREEDER:
        return None, "Claude breeder OFF - all children are random mutants."
    if not config.ANTHROPIC_API_KEY:
        return None, ("Claude breeder OFF - add ANTHROPIC_API_KEY=... to your .env file to let Claude design "
                      "children. Using random mutation.")
    try:
        breeder = ClaudeBreeder()
    except ImportError:
        return None, "Claude breeder OFF - run: .venv\\Scripts\\pip install anthropic"
    return breeder, (f"Claude breeder ON ({breeder.model}): {breeder.children} of each survivor's "
                     f"{config.CLAUDE_CHILDREN + 2} children are designed by Claude; the rest stay random as a "
                     "control group.")
