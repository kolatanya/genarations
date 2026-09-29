"""Rich output for the Evolution Lab (v2)."""

from __future__ import annotations

import math

import numpy as np
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

from rich import box
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import config
from utils.display import Display, fmt_num, fmt_pct, pnl_style, sparkline

if TYPE_CHECKING:
    from lab.data import LabData
    from lab.evolution import HallEntry, IslandResult, LabSummary
    from lab.features import FeatureSet

STATUS_STYLE = {
    "DEFENDED": ("🛡  DEFENDED", "cyan"),
    "USURPED": ("👑 NEW ALPHA", "bold bright_green"),
    "BLOCKED_WOBBLE": ("🧪 BLOCKED: fragile", "yellow"),
    "BLOCKED_VALIDATION": ("🔒 BLOCKED: failed unseen data", "yellow"),
}


def _pct(x: float, digits: int = 2) -> Text:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return Text("-")
    return Text(fmt_pct(x * 100, digits=digits), style=pnl_style(x))


class LabDisplay(Display):
    """Display with Lab screens. quiet=True prints nothing (used by experiments)."""

    def __init__(self, *args, quiet: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.quiet = quiet

    def _print(self, *items, **kw) -> None:
        if not self.quiet:
            self.console.print(*items, **kw)

    # ------------------------------------------------------------------ #
    def lab_intro(self, data: "LabData", windows: dict[str, str], generations: int) -> None:
        if self.quiet:
            return
        lines = [
            f"Universe: [bold]{data.n_symbols} stocks[/] ({', '.join(data.symbols)})",
            f"Data: [bold]{data.source}[/] · {len(data.unique_days)} trading days · {data.n_bars:,} five-minute bars per stock",
            f"Senses: 33 features incl. SPY/QQQ mood, VIX, VWAP, multi-timeframe trend, earnings/Fed days"
            + (", news sentiment" if data.news is not None else ""),
            f"[bold]Training[/]   {windows['train']}   (split into {config.LAB_TRAIN_FOLDS} eras, "
            f"{config.LAB_FOLDS_PER_GEN} random eras per generation)",
            f"[bold]Validation[/] {windows['validation']}   (a challenger must win here too)",
            f"[bold]FINAL TEST[/] {windows['test']}   (untouched until the very end)",
            f"Arena: {config.LAB_ISLANDS} islands × {config.LAB_BOTS_PER_ISLAND} bots · {generations} generations",
        ]
        self.console.print(Panel("\n".join(lines), title="🧫 EVOLUTION LAB", border_style="magenta"))
        for note in data.notes:
            self.warn(note)

    def lab_generation_header(self, gen: int, last: int, folds: Sequence[int], n_folds: int, cost_mult: float) -> None:
        self._print()
        self._print(Rule(f"[bold cyan]🧬 GENERATION {gen} / {last}[/]", style="cyan"))
        self._print(Text(f"Eras this generation: {', '.join(str(f + 1) for f in folds)} of {n_folds} · "
                         f"trading costs ×{cost_mult:.2f} (noise injection)", style="bright_black"), justify="center")

    @contextmanager
    def lab_progress(self, total: int) -> Iterator[Callable[[str], None]]:
        if self.quiet:
            yield lambda label: None
            return
        progress = Progress(SpinnerColumn(style="magenta"), TextColumn("[bold]{task.description}"),
                            BarColumn(bar_width=30, complete_style="magenta"), TextColumn("{task.completed}/{task.total}"),
                            TimeElapsedColumn(), console=self.console, transient=True)
        with progress:
            task = progress.add_task("Bots trading...", total=total)
            yield lambda label: progress.update(task, advance=1, description=f"{label} traded")

    def lab_generation(self, gen: int, results: list["IslandResult"], events: list[str],
                       hall: list["HallEntry"], trials: int) -> None:
        if self.quiet:
            return
        table = Table(title=f"⚔  ISLAND BOARD · Generation {gen}", box=box.ROUNDED, header_style="bold")
        for col, j in (("Island", "center"), ("Alpha", "left"), ("Origin", "left"), ("Result", "left"),
                       ("Fitness", "right"), ("Sharpe", "right"), ("Avg/era", "right"), ("Win %", "right"),
                       ("Profit F.", "right"), ("Every", "right"), ("Eras +", "right"), ("Markets +", "right"),
                       ("Mutation", "right")):
            table.add_column(col, justify=j)  # type: ignore[arg-type]
        for r in results:
            s = r.winner_score
            label, style = STATUS_STYLE[r.status]
            table.add_row(
                str(r.island + 1), f"Bot #{r.winner_id}", r.winner_role.title(), Text(label, style=style),
                fmt_num(s.fitness, signed=True), fmt_num(s.sharpe), _pct(s.avg_return), f"{s.win_rate:.0%}",
                f"{s.profit_factor:.2f}", Text(s.trade_pace, style="white" if s.on_pace else "yellow"),
                f"{s.eras_profitable}/{s.n_eras}", f"{s.regimes_profitable}/3",
                f"{r.rate_before:.0%}→{r.rate_after:.0%}",
            )
        self.console.print(table)

        for r in results:
            if r.status == "BLOCKED_WOBBLE" and r.wobble:
                self.console.print(Text(
                    f"🧪 Island {r.island + 1}: Bot #{r.challenger_id} out-scored the Alpha, but nudging its genes "
                    f"±{config.LAB_WOBBLE_PCT:.0%} collapsed it (plateau {r.wobble[0]:+.2f} vs {r.wobble[1]:+.2f}). "
                    f"Fragile - crown denied.", style="yellow"))
            elif r.status == "BLOCKED_VALIDATION" and r.validation:
                self.console.print(Text(
                    f"🔒 Island {r.island + 1}: Bot #{r.challenger_id} won in training but lost on unseen validation "
                    f"data (Sharpe {r.validation[0]:.2f} vs {r.validation[1]:.2f}). Memoriser - crown denied.",
                    style="yellow"))
        for r in results:
            purged = ", ".join(f"#{t.bot_id}" for t in r.eliminated)
            best = max(r.eliminated, key=lambda t: t.score.fitness) if r.eliminated else None
            detail = f" (best of them: {best.role.title()} #{best.bot_id}, Sharpe {best.score.sharpe:.2f})" if best else ""
            self.console.print(Text.assemble(("[ELIMINATED]", "bold white on red"), " ",
                                             (f"Island {r.island + 1}: {purged} purged{detail}", "red")))
            self.pause(0.3)
        if results and all(r.purge_verified for r in results):
            n = sum(len(r.eliminated) for r in results)
            self.console.print(Text(f"🧹 Memory purge verified: {n}/{n} eliminated bots garbage-collected", style="bright_black"))
        for r in results:
            if r.status == "USURPED":
                self.console.print(Text(
                    f"[SURVIVOR] Bot #{r.winner_id} ({r.winner_role.title()}) crowned Alpha of Island {r.island + 1}! "
                    f"Sharpe {r.winner_score.sharpe:.2f} · win {r.winner_score.win_rate:.0%} · "
                    f"1 trade every {r.winner_score.trade_pace}", style="bold bright_green"))
        for e in events:
            self.console.print(Text(e, style="bold magenta" if "METEOR" in e else "cyan"))
        if hall:
            top = hall[0]
            self.console.print(Text(f"Hall of Fame leader: Bot #{top.bot_id} · validation Sharpe {top.val_sharpe:.2f} · "
                                    f"strategies tested so far: {trials:,}", style="bright_black"))
        self.pause(1.0)

    # ------------------------------------------------------------------ #
    def lab_final(self, summary: "LabSummary", fs: "FeatureSet") -> None:
        if self.quiet:
            return
        c = self.console
        c.print()
        c.print(Rule("[bold magenta]🏆 EVOLUTION LAB COMPLETE 🏆[/]", style="magenta"))
        if summary.interrupted:
            self.warn("Run interrupted - results up to the last completed generation.")
        if summary.island_fitness:
            best = [max(g) for g in summary.island_fitness]
            c.print(Text(f"Best Alpha fitness per generation: {sparkline(best)}  ({best[0]:+.2f} → {best[-1]:+.2f})",
                         style="cyan"))

        hall = Table(title="🏛  Hall of Fame (ranked on unseen validation data)", box=box.SIMPLE_HEAVY)
        for col in ("#", "Bot", "Island", "Gen", "Val Sharpe", "Val return", "Style", "Rules"):
            hall.add_column(col, justify="right" if col not in ("Bot", "Style") else "left")
        for i, h in enumerate(summary.hall, 1):
            hall.add_row(str(i), f"Bot #{h.bot_id}", str(h.island + 1), str(h.generation), fmt_num(h.val_sharpe),
                         _pct(h.val_return), h.genome.style, str(h.genome.complexity))
        c.print(hall)

        if summary.champion:
            rules = summary.champion.genome.describe(fs)
            body = "\n".join(f"[bold]{k}:[/] {v}" for k, v in rules.items())
            c.print(Panel(body, title=f"🧠 What Champion Bot #{summary.champion.bot_id} learned", border_style="green"))
        if summary.ensemble and len(summary.ensemble.members) > 1:
            c.print(Panel("\n".join(summary.ensemble.describe_weights()),
                          title="🗳  Ensemble vote weights by market type (up / sideways / down days)",
                          border_style="cyan"))

        show = Table(title=f"🔬 FINAL TEST on untouched data ({summary.windows.get('test', '-')})", box=box.ROUNDED)
        for col in ("Strategy", "Return", "Sharpe", "Max DD", "Trades", "Win %", "Profit F.", "Luck range (5%–95%)",
                    "P(profit)"):
            show.add_column(col, justify="left" if col == "Strategy" else "right")
        for s in summary.showdown:
            mc = s.monte_carlo
            show.add_row(s.name, _pct(s.total_return), fmt_num(s.sharpe), f"{s.max_dd:.1%}", str(s.trades),
                         "-" if math.isnan(s.win_rate) else f"{s.win_rate:.0%}",
                         "-" if math.isnan(s.profit_factor) else f"{s.profit_factor:.2f}",
                         f"{mc.p5:+.1%} … {mc.p95:+.1%}" if mc else "-",
                         f"{mc.prob_profit:.0%}" if mc else "-")
        c.print(show)
        c.print(Text("Luck range = 2,000 reshuffled versions of the same daily results (Monte Carlo). "
                     "P(profit) = share of those that ended up.", style="italic bright_black"))

        dsr = summary.deflated_sharpe
        if not math.isnan(dsr):
            verdict = ("convincing" if dsr > 0.95 else "promising" if dsr > 0.8 else
                       "could easily be luck" if dsr > 0.5 else "most likely luck")
            c.print(Panel(
                f"After testing [bold]{summary.trials:,}[/] strategies, the probability the champion's training "
                f"edge is real (Deflated Sharpe Ratio): [bold]{dsr:.0%}[/] → {verdict}.\n"
                "Testing thousands of bots guarantees some look brilliant by chance; this corrects for that.",
                title="🎲 Luck check", border_style="yellow"))
        c.print(Text(f"Pareto archive: {len(summary.pareto)} bots that are best at some trade-off of return, "
                     f"drawdown and consistency (saved in the checkpoint).", style="bright_black"))
        paths = [f"Lab checkpoint: [bold]{summary.checkpoint}[/]"]
        if summary.chart:
            paths.append(f"Lab chart:      [bold]{summary.chart}[/]")
        c.print(Panel("\n".join(paths), border_style="magenta", title="Artifacts"))

    # ------------------------------------------------------------------ #
    def monkey_report(self, real: dict, monkeys: list[dict]) -> None:
        t = Table(title="🐒 LUCK DETECTOR: real market vs monkey markets (shuffled days, no real patterns)",
                  box=box.ROUNDED)
        for col in ("Market", "Final-test return", "Final-test Sharpe", "Val Sharpe (best)"):
            t.add_column(col, justify="left" if col == "Market" else "right")
        t.add_row("REAL", _pct(real["return"]), fmt_num(real["sharpe"]), fmt_num(real["val"]), style="bold")
        for i, m in enumerate(monkeys, 1):
            t.add_row(f"Monkey {i}", _pct(m["return"]), fmt_num(m["sharpe"]), fmt_num(m["val"]))
        self.console.print(t)
        beaten = sum(real["sharpe"] > m["sharpe"] for m in monkeys)
        if beaten == len(monkeys) and real["sharpe"] > 0:
            msg, style = (f"The real-market bots beat all {len(monkeys)} monkey runs. "
                          "That's evidence of a real pattern (more monkey runs = more certainty)."), "bold green"
        elif beaten == 0:
            msg, style = ("The monkeys did as well or better. The real results are indistinguishable "
                          "from luck."), "bold red"
        else:
            msg, style = (f"Real bots beat {beaten} of {len(monkeys)} monkey runs. Inconclusive: "
                          "the edge, if any, is weak."), "bold yellow"
        self.console.print(Panel(msg, border_style=style.split()[-1]))

    def walk_forward_report(self, steps: list[dict], total: dict, bench: dict) -> None:
        t = Table(title="🚶 WALK-FORWARD: evolve on the past, trade the next month, repeat", box=box.ROUNDED)
        for col in ("Step", "Evolved on", "Traded", "Bot", "Return", "Trades", "Buy & hold"):
            t.add_column(col, justify="left" if col in ("Evolved on", "Traded", "Bot") else "right")
        for i, s in enumerate(steps, 1):
            t.add_row(str(i), s["train"], s["trade"], s["bot"], _pct(s["return"]), str(s["trades"]), _pct(s["bench"]))
        self.console.print(t)
        self.console.print(Panel(
            f"Walk-forward total: [bold]{fmt_pct(total['return'] * 100, digits=2)}[/] · Sharpe {total['sharpe']:.2f} · "
            f"max DD {total['max_dd']:.1%}\nBuy & hold over the same months: "
            f"{fmt_pct(bench['return'] * 100, digits=2)} · Sharpe {bench['sharpe']:.2f} · max DD {bench['max_dd']:.1%}\n"
            "This is the most realistic backtest: every month is traded by a bot that had never seen it.",
            border_style="cyan"))

    # ------------------------------------------------------------------ #
    # Lab live
    # ------------------------------------------------------------------ #
    def lab_live_idle(self, cycle: int, now) -> None:
        if cycle == 1 or cycle % 15 == 0:
            self.console.print(Text(f"{now:%Y-%m-%d %H:%M} ET · US market closed - waiting (checks every minute, "
                                    f"Ctrl+C to stop)", style="bright_black"))

    def lab_live_cycle(self, cycle: int, bar_time: str, regime: str, equity: float, day_pnl: float, rows,
                       benched: bool, loss_day: bool, shadow_line: str | None, dry_run: bool) -> None:
        mode = "DRY RUN" if dry_run else "PAPER ORDERS"
        t = Table(title=f"🗳  Ensemble · cycle {cycle} · last bar {bar_time} ET · market: {regime} · {mode}",
                  box=box.ROUNDED, header_style="bold")
        for col, j in (("Symbol", "left"), ("Price", "right"), ("Position", "right"), ("Action", "left"),
                       ("Reason", "left"), ("Order", "left")):
            t.add_column(col, justify=j)  # type: ignore[arg-type]
        styles = {"BUY": "bold bright_green", "SHORT": "bold red", "CLOSE": "bold magenta", "HOLD": "bright_black"}
        interesting = [r for r in rows if r[3] != "HOLD" or r[2]]
        quiet = len(rows) - len(interesting)
        for sym, price, pos, action, reason, order in interesting:
            t.add_row(sym, f"${price:,.2f}", f"{pos:+g}" if pos else "flat", Text(action, style=styles.get(action, "white")),
                      reason, order)
        if not interesting:
            t.add_row("-", "", "", Text("HOLD", style="bright_black"), "no votes on any stock this bar", "-")
        self.console.print(t)
        status = f"Equity ${equity:,.2f} · today {day_pnl:+.2%} · {quiet} stock(s) with no vote"
        if benched:
            status += " · ⚠ BENCHED by drift alarm (run with --unbench to resume)"
        if loss_day:
            status += " · ⛔ daily loss limit hit - done for today"
        self.console.print(Text(status, style="yellow" if (benched or loss_day) else "cyan"))
        if shadow_line:
            self.console.print(Text(shadow_line, style="magenta"))

    # ------------------------------------------------------------------ #
    # Forecaster Arena
    # ------------------------------------------------------------------ #
    def forecast_intro(self, engine, n_symbols: int, generations: int, source: str) -> None:
        from lab.forecast import TARGETS

        yes, no, question = TARGETS[engine.target]
        lines = [
            f"Question: [bold]{question}[/]  ({yes} / {no}, one call per stock per day - no skipping hard days)",
            f"Data: {n_symbols} large US stocks · daily bars · {source}",
            f"[bold]Training[/]   {engine.dates(engine.train)}   (6 eras, 4 random per generation)",
            f"[bold]Validation[/] {engine.dates(engine.val)}   (a challenger must beat the Alpha here too)",
            f"[bold]FINAL TEST[/] {engine.dates(engine.test)}   (untouched until the end)",
            f"Naive baseline in training: always guessing the most common answer is right "
            f"[bold]{engine._base(engine.train):.1%}[/] of the time. Only accuracy ABOVE that counts.",
            f"Arena: {engine.n_islands} islands × {engine.bots} forecasters · {generations} generations",
        ]
        self.console.print(Panel("\n".join(lines), title=f"🔮 FORECASTER ARENA · {engine.target.upper()}",
                                 border_style="magenta"))

    def forecast_generation(self, engine, gen: int, last: int, rows: list, events: list, purged: int) -> None:
        if self.quiet:
            return
        self.console.print()
        self.console.print(Rule(f"[bold cyan]🔮 GENERATION {gen} / {last}[/]", style="cyan"))
        t = Table(box=box.ROUNDED, header_style="bold")
        for col, j in (("Island", "center"), ("Alpha", "left"), ("Origin", "left"), ("Result", "left"),
                       ("Train accuracy", "right"), ("Edge", "right"), ("Val accuracy", "right"),
                       ("Val edge", "right"), ("Says YES", "right"), ("Mutation", "right")):
            t.add_column(col, justify=j)  # type: ignore[arg-type]
        for r in rows:
            s, v = r["score"], r["val"]
            label, style = STATUS_STYLE[r["status"]]
            t.add_row(str(r["island"] + 1), f"Bot #{r['bot']}", r["role"].title(), Text(label, style=style),
                      f"{s.accuracy:.2%}", Text(f"{s.edge * 100:+.2f}", style=pnl_style(s.edge)),
                      f"{v.accuracy:.2%}", Text(f"{v.edge * 100:+.2f}", style=pnl_style(v.edge)),
                      f"{s.says_true:.0%}", f"{r['rate']:.0%}")
        self.console.print(t)
        for r in rows:
            if r["status"] == "BLOCKED_VALIDATION":
                self.console.print(Text(f"🔒 Island {r['island'] + 1}: Bot #{r['challenger']} predicted better in "
                                        "training but worse on unseen years. Memoriser - crown denied.", style="yellow"))
            ids = ", ".join(f"#{b}" for b, _, _ in r["losers"])
            self.console.print(Text.assemble(("[ELIMINATED]", "bold white on red"), " ",
                                             (f"Island {r['island'] + 1}: {ids} purged", "red")))
            self.pause(0.3)
        self.console.print(Text(f"🧹 Memory purge verified: {purged}/{purged} eliminated forecasters garbage-collected",
                                style="bright_black"))
        for r in rows:
            if r["status"] == "USURPED":
                self.console.print(Text(f"[SURVIVOR] Bot #{r['bot']} ({r['role'].title()}) crowned Alpha of Island "
                                        f"{r['island'] + 1}! Validation accuracy {r['val'].accuracy:.2%}",
                                        style="bold bright_green"))
        for e in events:
            self.console.print(Text(e, style="bold magenta" if "METEOR" in e else "cyan"))
        self.pause(0.8)

    def forecast_final(self, s, fs) -> None:
        from lab.forecast import TARGETS, describe_tree

        yes, no, question = TARGETS[s.target]
        c = self.console
        c.print()
        c.print(Rule(f"[bold magenta]🔮 FORECASTER RESULTS · {s.target.upper()} 🔮[/]", style="magenta"))
        c.print(Text(f"Best training accuracy by generation:   {sparkline(s.train_acc)}  "
                     f"({s.train_acc[0]:.2%} → {s.train_acc[-1]:.2%}, naive {s.baseline_train:.2%})", style="cyan"))
        vals = [v for v in s.val_acc if not math.isnan(v)]
        if vals:
            c.print(Text(f"Best validation accuracy by generation: {sparkline(vals)}  "
                         f"({vals[0]:.2%} → {vals[-1]:.2%}, naive {s.baseline_val:.2%})", style="cyan"))
        if s.champion:
            c.print(Panel(f"[bold]Predicts {yes} when:[/] {describe_tree(s.champion['tree'], fs)}\n"
                          f"[bold]Otherwise:[/] {no}", title=f"🧠 What Champion Bot #{s.champion['bot_id']} learned",
                          border_style="green"))
        t = Table(title=f"🔬 FINAL TEST on untouched data ({s.windows['test']}) · {question}", box=box.ROUNDED)
        for col in ("Forecaster", "Accuracy", "Naive baseline", "Edge (pts)", "Luck p-value", f"Says {yes}", "What it means"):
            t.add_column(col, justify="left" if col in ("Forecaster", "What it means") else "right")
        for r in s.results:
            p = "-" if r.p_value is None else f"{r.p_value:.3f}"
            t.add_row(r.name, f"{r.accuracy:.2%}", f"{r.baseline:.2%}",
                      Text(f"{r.edge * 100:+.2f}", style=pnl_style(r.edge)), p, f"{r.says_true:.0%}", r.extra)
        c.print(t)
        c.print(Text("Edge = accuracy minus the naive baseline (always guessing the more common answer). "
                     "Luck p-value = share of 500 shuffled-answer tests that scored as well: below 0.05 = unlikely luck.",
                     style="italic bright_black"))
        champ = s.results[0] if s.results and s.results[0].name.startswith("Champion") else None
        if champ:
            if champ.edge > 0 and champ.p_value is not None and champ.p_value < 0.05:
                verdict = (f"[bold green]Real predictive power.[/] The champion beat the naive baseline by "
                           f"{champ.edge * 100:.2f} points on years it never saw, and shuffled answers almost never do that.")
            elif champ.edge > 0:
                verdict = ("[bold yellow]Slightly better than naive, but not clearly beyond luck.[/] "
                           "More data or a longer test would be needed to be sure.")
            else:
                verdict = "[bold red]No real predictive power.[/] On unseen years it did no better than the naive guess."
            c.print(Panel(verdict + f"\n{s.trials:,} forecasters were tested to find this champion.",
                          title="🎲 Verdict", border_style="yellow"))
        paths = [f"Checkpoint: [bold]{s.checkpoint}[/]"] + ([f"Chart:      [bold]{s.chart}[/]"] if s.chart else [])
        c.print(Panel("\n".join(paths), border_style="magenta", title="Artifacts"))

    # ------------------------------------------------------------------ #
    # Prophecy League
    # ------------------------------------------------------------------ #
    def prophecy_replay_intro(self, league, first, last, days: int, fresh: bool = False) -> None:
        from lab.forecast import TARGETS
        from lab.prophecy import ACCURACY_WEIGHT, ACTIVITY_WEIGHT, JUDGE_DAYS, LEAGUE_SIZE, ROUND_DAYS, crown_margin_points

        yes, no, question = TARGETS[league.target]
        founder = ("a plain beginner rule - everything else must be learned by survival" if fresh
                   else "the Forecaster Arena champion (if you've run option 8)")
        self.console.print(Panel(
            f"Question: [bold]{question}[/] ({yes}/{no}) for every stock, every day\n"
            f"[bold]{days // ROUND_DAYS} generations[/], {first} → {last}. Each generation is judged on "
            f"{ROUND_DAYS} trading days it has NEVER seen.\n"
            "Bots keep no market data - only their rule (the strategy they inherited from their Alpha).\n"
            f"Crown = most POINTS over the last {JUDGE_DAYS} known days: {ACCURACY_WEIGHT:g} per % of calls right "
            f"above the naive guess + {ACTIVITY_WEIGHT:g} per 10% of calls that are the bold {yes} call (max 5).\n"
            f"A challenger must beat the Alpha by {crown_margin_points():.1f} points (one lucky week can't steal the "
            f"crown). The survivor has {LEAGUE_SIZE - 1} children; the rest are purged.\n"
            f"Founding Alpha: Bot #{league.alpha.bot_id}: {founder}",
            title=f"📜 PROPHECY LEAGUE · EVOLUTION THROUGH HISTORY · {league.target.upper()}", border_style="magenta"))

    def prophecy_round(self, league, rec: dict, naive_days: list) -> None:
        if self.quiet:
            return
        naive = float(np.mean(naive_days)) if naive_days else float("nan")
        judged = rec.get("judge_days", 5)
        det = rec.get("details") or {}
        if rec.get("points") and det:
            def cell(b: int) -> str:
                d = det.get(str(b))
                if d is None:
                    return f"#{b} ?"
                tag = "" if d["reasoned"] else " NO-REASON"
                return f"#{b} {d['points']:+.1f}pts ({d['acc']:.1%} right, bold {d['bold']:.0%}){tag}"

            board = "  ".join(cell(b) for b, _, _, _, _ in rec["leaderboard"])
            label = f"Points over the last {judged} days"
        else:
            board = "  ".join(f"#{b} {acc:.1%}" if acc >= 0 else f"#{b} no-reason"
                              for b, _, acc, _, _ in rec["leaderboard"])
            label = "Calls correct" if judged <= 5 else f"Correct over the last {judged} days"
        self.console.print(Rule(f"[cyan]Generation {rec['round']} · {rec['dates']}[/]", style="cyan"))
        self.console.print(Text(f"{label}: {board}" + (f"   (naive guess this week {naive:.1%})" if naive == naive else ""),
                                style="white"))
        losers = [f"#{b}" for b, _, _, _, _ in rec["leaderboard"][1:]]
        self.console.print(Text.assemble(("[ELIMINATED]", "bold white on red"), " ",
                                         (f"{', '.join(losers)} purged", "red"),
                                         (f"  · memory purge verified ({rec['purged']}/{rec['purged']})", "bright_black")))
        n_claude = rec.get("claude_children", 0)
        born = f"5 children born ({n_claude} designed by Claude)" if n_claude else "5 children born"
        pts = bool(rec.get("points") and det)

        def fmt(x: float) -> str:
            if pts:
                return f"{x:+.2f} pts" if x > -500 else "no reason - always the same answer"
            return f"{x:.1%}" if x >= 0 else "no reason - always the same answer"

        if rec["dethroned"]:
            verb = (f"beat the old Alpha Bot #{rec['alpha_before']} on the SAME {judged} days: "
                    f"{fmt(rec['survivor_accuracy'])} vs {fmt(rec['alpha_accuracy'])}")
        else:
            verb = f"defended the crown with {fmt(rec['survivor_accuracy'])}"
            close = [(b, acc) for b, _, acc, _, _ in rec["leaderboard"][1:] if acc > rec["survivor_accuracy"]]
            if close:
                b, acc = max(close, key=lambda x: x[1])
                margin = rec.get("margin", 0.003)
                verb += (f" (Bot #{b} scored {fmt(acc)}, but a challenger must beat the Alpha by "
                         f"{fmt(margin) if pts else f'{margin * 100:.1f} pts'} - it needed "
                         f"{fmt(rec['survivor_accuracy'] + margin)})")
        self.console.print(Text(f"[SURVIVOR] Bot #{rec['survivor']} ({_role(rec['survivor_role'])}) {verb} · "
                                f"{born} · mutation {league.rate:.0%}", style="bold bright_green"))
        survivor = next((p for p in league.prophets if p.bot_id == rec["survivor"]), None)
        if survivor is not None and survivor.note and rec["survivor_role"] == "CLAUDE":
            self.console.print(Text(f"   🧠 Claude's idea that won: {survivor.note}", style="magenta"))
        self.pause(0.5)

    def prophecy_replay_summary(self, s, fs) -> None:
        from lab.forecast import TARGETS, describe_tree

        yes, no, question = TARGETS[s.target]
        lg = s.league
        c = self.console
        c.print()
        c.print(Rule(f"[bold magenta]📜 PROPHECY LEAGUE RESULT · {s.target.upper()}[/]", style="magenta"))
        wins = sum(1 for h in lg.history if h["alpha_accuracy"] > 0)
        dethroned = sum(1 for h in lg.history if h["dethroned"])
        t = Table(box=box.ROUNDED, title=f"The honest score: the REIGNING Alpha's calls ({len(s.days)} days, "
                                         f"{lg.alpha_total:,} calls)")
        for col in ("Who", "Calls correct", "vs naive (pts)", "90% range of the difference"):
            t.add_column(col, justify="left" if col == "Who" else "right")
        diff = (s.alpha_accuracy - s.naive_accuracy) * 100
        lo, hi = s.ci
        t.add_row("Reigning Alpha (chosen BEFORE each round)", f"{s.alpha_accuracy:.2%}",
                  Text(f"{diff:+.2f}", style=pnl_style(diff)), f"{lo:+.2f} … {hi:+.2f}")
        t.add_row(f"Naive guess (the more common answer last year)", f"{s.naive_accuracy:.2%}", "0.00", "-")
        t.add_row("Coin flip", "50.00%", f"{(0.5 - s.naive_accuracy) * 100:+.2f}", "-")
        c.print(t)
        c.print(Text(f"{len(lg.history)} generations · the crown changed hands {dethroned} times · "
                     f"{len(lg.history) * 5:,} prophets purged", style="bright_black"))
        eras = s.eras()
        if eras:
            e = Table(box=box.SIMPLE_HEAVY, title="📈 Did they get better? The reigning Alpha on unseen weeks, era by era")
            for col in ("Generations", "Dates", "Alpha correct", "Naive guess", "Edge (pts)"):
                e.add_column(col, justify="left" if col in ("Generations", "Dates") else "right")
            for r in eras:
                e.add_row(f"{r['first_gen']}–{r['last_gen']}", f"{r['from']} → {r['to']}", f"{r['alpha']:.2%}",
                          f"{r['naive']:.2%}", Text(f"{r['edge']:+.2f}", style=pnl_style(r["edge"])))
            c.print(e)
            first, last_ = eras[0]["edge"], eras[-1]["edge"]
            trend = ("the edge grew from the first era to the last" if last_ > first + 0.5 else
                     "the edge shrank from the first era to the last" if last_ < first - 0.5 else
                     "the edge stayed about the same from the first era to the last")
            c.print(Text(f"Edge = points better than the naive guess. Over this run {trend} "
                         f"({first:+.2f} → {last_:+.2f}). Markets change too, so eras aren't perfectly comparable.",
                         style="bright_black"))
        self.breeder_scoreboard(lg)
        if lo > 0:
            verdict = "[bold green]Real predictive power:[/] the reigning prophet beat the naive guess, and even the pessimistic end of the range is above zero."
        elif hi < 0:
            verdict = "[bold red]Worse than naive:[/] on days it hadn't seen, the reigning prophet lost to the simple guess."
        else:
            verdict = "[bold yellow]Indistinguishable from the naive guess:[/] the range includes zero, so this could be luck either way."
        c.print(Panel(verdict, title="🎲 Verdict", border_style="yellow"))
        final = lg.alpha
        c.print(Panel(f"[bold]Predicts {yes} when:[/] {describe_tree(final.tree, fs)}\n[bold]Otherwise:[/] {no}\n"
                      f"Lifetime accuracy {final.life_accuracy:.2%} over {final.life_total:,} calls",
                      title=f"👑 Current Alpha: Bot #{final.bot_id}", border_style="green"))
        if s.explain:
            self.prophecy_playbook(s.target, s.explain)
        if s.chart:
            c.print(Text(f"Chart: {s.chart}", style="bright_black"))
        _ = wins

    def prophecy_live_founded(self, today: str) -> None:
        self.console.print(Panel(f"A new Prophecy League was founded with data through {today}. "
                                 "Run this once every evening after 9:15pm UK: it scores yesterday's calls "
                                 "and makes tomorrow's.", border_style="magenta", title="📜 PROPHECY LEAGUE · LIVE"))

    def prophecy_live(self, today: str, symbols, leagues, reveals, rounds, calls_today, fs, judge_scores=None) -> None:
        from lab.forecast import TARGETS
        from lab.prophecy import JUDGE_DAYS

        c = self.console
        if reveals:
            t = Table(title="📜 PROPHECIES REVEALED", box=box.ROUNDED)
            for col in ("League", "Called on", "Result day", "Alpha correct", "Best prophet", "Naive guess"):
                t.add_column(col, justify="left" if col == "League" else "right")
            for r in reveals:
                if not r["accs"]:
                    continue
                best = max(r["accs"].items(), key=lambda kv: kv[1])
                t.add_row(r["target"].title(), r["day"], r["result_day"],
                          f"{r['accs'].get(r['alpha'], float('nan')):.1%}", f"#{best[0]} {best[1]:.1%}",
                          f"{r['naive']:.1%}")
            c.print(t)
        else:
            c.print(Text("No prophecies to reveal yet (the day after a call has to close first).", style="bright_black"))
        for target, rec in rounds:
            self.prophecy_round(leagues[target], rec, [])
        for lg in leagues.values():
            self.breeder_scoreboard(lg)
        for target, lg in leagues.items():
            yes, no, question = TARGETS[target]
            t = Table(title=f"{target.upper()} league · round {lg.round_no + 1} · day {lg.round_days}/5 · {question}",
                      box=box.SIMPLE_HEAVY)
            judge = (judge_scores or {}).get(target, {})
            for col in ("Prophet", "Role", f"Points, last {JUDGE_DAYS} days (decides the crown)", "Right",
                        f"Says {yes}", "This week (live)", f"Tomorrow: says {yes} for"):
                t.add_column(col, justify="left" if col in ("Prophet", "Role") else "right")
            calls = calls_today.get(target, {})
            for p in sorted(lg.prophets, key=lambda p: judge.get(p.bot_id, {}).get("points", -1e9), reverse=True):
                n_yes = sum(calls.get(str(p.bot_id), []))
                d = judge.get(p.bot_id)
                t.add_row(f"Bot #{p.bot_id}", _role(p.role),
                          "-" if d is None else (f"{d['points']:+.1f}" if d["reasoned"] else "no reason"),
                          "-" if d is None else f"{d['acc']:.1%}", "-" if d is None else f"{d['bold']:.0%}",
                          f"{p.accuracy:.1%}" if p.total else "-", f"{n_yes}/{len(symbols)} stocks")
            c.print(t)
            ideas = [p for p in lg.prophets if p.note]
            if ideas:
                c.print(Panel("\n".join(f"[bold]Bot #{p.bot_id}[/]: {p.note}" for p in ideas),
                              title=f"🧠 Claude's children in the {target} league - the idea behind each",
                              border_style="magenta"))
            if lg.alpha_total:
                c.print(Text(f"Reigning Alpha's live record: {lg.alpha_correct / lg.alpha_total:.2%} correct vs naive "
                             f"{lg.naive_correct / max(lg.naive_total, 1):.2%} over {lg.alpha_total:,} calls",
                             style="cyan"))
        if calls_today:
            alpha_calls = calls_today.get("direction", {}).get(str(leagues["direction"].alpha.bot_id), [])
            ups = [s for s, v in zip(symbols, alpha_calls) if v]
            c.print(Panel(f"Direction Alpha says UP tomorrow for {len(ups)} of {len(symbols)} stocks"
                          + (f": {', '.join(ups[:15])}{' …' if len(ups) > 15 else ''}" if ups else ""),
                          title=f"🔮 Tomorrow's prophecies (made with data through {today})", border_style="magenta"))

    def prophecy_playbook(self, target: str, ex: dict) -> None:
        """Why the Alpha calls what it calls: its rule, each condition's track record, and example calls."""
        from lab.forecast import TARGETS

        yes, no, _ = TARGETS[target]
        pb = ex["playbook"]
        c = self.console
        lines = [f"[bold]Says {yes} when:[/] {ex['rule']}   [bright_black](otherwise {no})[/]"]
        if pb.get("n"):
            lines.append(f"Track record, last {pb['days']} days ({pb['n']:,} calls): says {yes} on {pb['says_yes']:.0%} "
                         f"of calls · right [bold]{_pct(pb['right_yes'])}[/] when it says {yes}, "
                         f"{_pct(pb['right_no'])} when it says {no} · overall {pb['accuracy']:.1%} "
                         f"(the answer was {yes} {pb['base']:.1%} of the time)")
        c.print(Panel("\n".join(lines), title=f"🔎 Bot #{ex['bot']}'s playbook · {target} league", border_style="cyan"))
        if pb.get("leaves"):
            t = Table(box=box.SIMPLE, title="Its reasons, one by one")
            for col in ("Condition", "Fires on", f"{yes} when it fires", f"{yes} normally", "Cases"):
                t.add_column(col, justify="left" if col == "Condition" else "right")
            for lf in pb["leaves"]:
                rate = lf["yes_when_fires"]
                style = "green" if rate == rate and rate > pb["base"] + 0.02 else (
                    "red" if rate == rate and rate < pb["base"] - 0.02 else "white")
                t.add_row(lf["text"], f"{lf['fires']:.1%}", Text("-" if rate != rate else f"{rate:.1%}", style=style),
                          f"{pb['base']:.1%}", f"{lf['cases']:,}")
            c.print(t)
        for sym, hit, why in ex.get("samples", []):
            c.print(Text.assemble((f"  {sym:<6} → ", "white"), (f"{yes if hit else no:<5}", "bold green" if hit else "bright_black"),
                                  ("  because " + " AND ".join(why), "bright_black")))

    def breeder_scoreboard(self, lg) -> None:
        """Who wins rounds: Claude's designs or blind random offspring? (per child slot, so it's fair)"""
        entered = [h for h in lg.history if h.get("claude_children")]
        if not entered:
            return
        # A round's contestants were born in the PREVIOUS round, so pair each result with the lineup it faced.
        wins = {"CLAUDE": 0, "MUTANT": 0, "CROSSOVER": 0, "ALPHA": 0}
        slots = {"CLAUDE": 0, "MUTANT": 0, "CROSSOVER": 0}
        for prev, cur in zip(lg.history, lg.history[1:]):
            n_c = prev.get("claude_children") or 0
            if not n_c:
                continue
            slots["CLAUDE"] += n_c
            slots["MUTANT"] += 4 - n_c
            slots["CROSSOVER"] += 1
            wins[cur["survivor_role"]] = wins.get(cur["survivor_role"], 0) + 1
        judged = sum(wins.values())
        if not judged:
            self.console.print(Text(f"🧠 {lg.target.title()} league: Claude has designed children in {len(entered)} "
                                    "round(s); the first verdict comes when that round ends.", style="magenta"))
            return
        parts = [f"Claude-designed {wins['CLAUDE']}/{slots['CLAUDE']} children won",
                 f"random mutants {wins['MUTANT']}/{slots['MUTANT']}",
                 f"crossovers {wins['CROSSOVER']}/{slots['CROSSOVER']}",
                 f"Alpha defended {wins['ALPHA']}/{judged}"]
        self.console.print(Text(f"🧠 Breeder scoreboard ({lg.target}, {judged} rounds): " + " · ".join(parts),
                                style="magenta"))

    def claude_breeder_status(self, breeder) -> None:
        if breeder is None or not breeder.calls:
            return
        ok = breeder.calls - breeder.failures
        msg = f"Claude designed children {ok}/{breeder.calls} times this run"
        if breeder.failures:
            msg += f" (fell back to random mutation {breeder.failures}x: {breeder.last_error})"
        self.console.print(Text(msg, style="bright_black" if not breeder.failures else "yellow"))


def _pct(x: float) -> str:
    return f"{x:.1%}" if x == x else "-"


def _role(role: str) -> str:
    return "Claude-bred" if role == "CLAUDE" else role.title()
