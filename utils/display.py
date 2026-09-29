"""
Rich terminal rendering - built to look dramatic on a YouTube screen recording.

Nothing in here holds a reference to a TradingBot: it only receives immutable
reports and genomes, so the display can never keep a purged bot alive.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

import config
from bot.genome import GENE_SPECS, TradingGenome

if TYPE_CHECKING:  # imported for type hints only (avoids circular imports)
    from broker.alpaca_broker import AccountSummary
    from broker.base import MarketData, Windows
    from engine.evaluator import FitnessReport, SelectionResult
    from engine.evolution import EvolutionSummary
    from engine.live_trader import LiveDecision

_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _no_neg_zero(value: float, digits: int) -> float:
    """round(-0.001, 2) prints as '-0.00'; normalise it to 0."""
    return round(value, digits) or 0.0


def fmt_num(value: float, digits: int = 2, signed: bool = False) -> str:
    value = _no_neg_zero(value, digits)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def fmt_pct(value: float, signed: bool = True, digits: int = 1) -> str:
    return fmt_num(value, digits, signed) + "%"


def fmt_gene(gene: str, value: float | int) -> str:
    if gene.endswith("_pct"):
        return f"{value * 100:.1f}%"
    return str(int(value)) if GENE_SPECS[gene].is_int else f"{value:.3f}"


def pnl_style(value: float) -> str:
    return "bright_green" if value > 0 else ("red" if value < 0 else "white")


def sparkline(values: Sequence[float]) -> str:
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return _SPARK_BLOCKS[3] * len(values)
    scale = (len(_SPARK_BLOCKS) - 1) / (hi - lo)
    return "".join(_SPARK_BLOCKS[int(round((v - lo) * scale))] for v in values)


class Display:
    """All terminal output for both modes."""

    def __init__(self, console: Console | None = None, delay: float = config.DISPLAY_DELAY) -> None:
        self.console = console or Console(highlight=False)
        self.delay = max(delay, 0.0)

    # ------------------------------------------------------------------ #
    # Generic messages
    # ------------------------------------------------------------------ #
    def pause(self, multiplier: float = 1.0) -> None:
        if self.delay > 0:
            time.sleep(self.delay * multiplier)

    def info(self, message: str) -> None:
        self.console.print(Text(message, style="cyan"))

    def success(self, message: str) -> None:
        self.console.print(Text(f"✔ {message}", style="bold green"))

    def warn(self, message: str) -> None:
        self.console.print(Text(f"⚠ {message}", style="bold yellow"))

    def error(self, message: str) -> None:
        self.console.print(Panel(Text(message, style="bold white"), title="ERROR", border_style="red"))

    # ------------------------------------------------------------------ #
    # Intro
    # ------------------------------------------------------------------ #
    def banner(self) -> None:
        title = Text("🧬  E V O L U T I O N   A R E N A  🧬", style="bold magenta", justify="center")
        subtitle = Text("Genetic Algorithm Trading Bots", style="bold white", justify="center")
        rules = Text(
            f"1 Alpha · {config.clone_count()} Mutated Clones · Only ONE survives each generation",
            style="italic bright_black",
            justify="center",
        )
        self.console.print(Panel(Group(title, subtitle, rules), box=box.DOUBLE, border_style="magenta", padding=(1, 4)))

    def data_summary(self, market: "MarketData", windows: "Windows") -> None:
        table = Table(box=box.SIMPLE, title="📈 Market Data Loaded", title_style="bold cyan", show_edge=False)
        table.add_column("Symbol", style="bold")
        table.add_column("Bars", justify="right")
        table.add_column("From")
        table.add_column("To")
        table.add_column("Last Close", justify="right")
        for symbol, df in market.bars.items():
            table.add_row(symbol, str(len(df)), str(df.index[0].date()), str(df.index[-1].date()), f"${df['close'].iloc[-1]:,.2f}")
        self.console.print(table)

        lines = [
            f"Source: [bold]{market.source}[/]",
            f"Evolution (in-sample) window: [bold]{windows.train_start.date()} → {windows.train_end.date()}[/]",
        ]
        if windows.has_test:
            lines.append(f"Hold-out (out-of-sample) window: [bold]{windows.test_start.date()} → {windows.test_end.date()}[/]  (bots never see this during evolution)")
        self.console.print("\n".join(lines))
        if market.source == "synthetic":
            self.console.print(Panel("SYNTHETIC PRICES - random-walk data, not real market history.",
                                     border_style="yellow", style="bold yellow"))
        for note in market.notes:
            if "SYNTHETIC" not in note:
                self.warn(note)

    # ------------------------------------------------------------------ #
    # Generation lifecycle
    # ------------------------------------------------------------------ #
    def generation_header(self, generation: int, last_generation: int, folds: Sequence[tuple], n_symbols: int) -> None:
        self.console.print()
        self.console.print(Rule(f"[bold cyan]🧬 GENERATION {generation} / {last_generation}[/]", style="cyan"))
        fmt = "%d %b" if config.IS_INTRADAY else "%b %Y"
        periods = "  |  ".join(f"{a:{fmt}}–{b:{fmt}}" for a, b in folds)
        label = "test period" if len(folds) == 1 else "test periods"
        pace = (f" · target 1 trade / {config.TARGET_MINUTES_PER_TRADE:.0f} min"
                if config.TARGET_MINUTES_PER_TRADE else "")
        self.console.print(Text(
            f"{len(folds)} {label} ({periods}) · {config.BAR_INTERVAL} bars · {n_symbols} symbols · "
            f"${config.INITIAL_BALANCE / 1000:,.0f}k each · min {config.MIN_TRADES} trades{pace}",
            style="bright_black",
        ), justify="center")

    @contextmanager
    def arena_progress(self, total: int) -> Iterator[Callable[[str], None]]:
        """Progress bar while the bots trade. Yields `advance(label)`."""
        progress = Progress(
            SpinnerColumn(style="magenta"),
            TextColumn("[bold]{task.description}"),
            BarColumn(bar_width=30, complete_style="magenta"),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True,
        )
        with progress:
            task = progress.add_task("Bots trading...", total=total)

            def advance(label: str) -> None:
                progress.update(task, advance=1, description=f"{label} finished trading")

            yield advance

    def leaderboard(self, result: "SelectionResult") -> None:
        table = Table(
            title=f"⚔  ARENA RESULTS · Generation {result.generation}",
            title_style="bold white",
            box=box.ROUNDED,
            header_style="bold",
        )
        for name, justify in (("#", "right"), ("Bot", "left"), ("Role", "left"), ("DNA", "left"),
                              ("Avg PnL", "right"), ("Sharpe", "right"), ("Worst DD", "right"),
                              ("Trades", "right"), ("Every", "right"), ("Win %", "right"), ("Profit F.", "right"),
                              ("Periods +", "right"), ("Fitness", "right")):
            table.add_column(name, justify=justify)  # type: ignore[arg-type]
        for r in result.leaderboard:
            row_style = "bold bright_green" if r.rank == 1 else ("bright_black" if r.disqualified else "red")
            table.add_row(
                str(r.rank),
                ("👑 " if r.rank == 1 else "") + r.name,
                "DQ" if r.disqualified else r.role.value,
                r.genome.fingerprint(),
                Text(fmt_pct(r.net_profit_pct, digits=2), style=pnl_style(r.net_profit_pct) if r.rank == 1 else row_style),
                f"{fmt_num(r.sharpe)}",
                f"{r.max_drawdown:.1%}",
                str(r.total_trades),
                Text(r.trade_pace, style=row_style if r.on_pace else "yellow"),
                f"{r.win_rate:.0%}",
                f"{r.profit_factor:.2f}",
                f"{r.folds_profitable}/{r.n_folds}",
                f"{fmt_num(r.fitness, signed=True)}",
                style=row_style,
            )
        self.console.print(table)
        self.console.print(Text(
            "Every = avg time between trades (yellow = off pace) · Profit F. = $ won ÷ $ lost · "
            "Periods + = profitable periods · DQ = too few trades",
            style="bright_black",
        ), justify="center")
        self.pause(2)

    def extinction_log(self, result: "SelectionResult") -> None:
        """Red elimination lines - worst bot first, building suspense toward the survivor."""
        self.console.print(Rule("[bold red]☠  EXTINCTION EVENT  ☠[/]", style="red"))
        per_line = min(1.0, 5.0 / max(len(result.eliminated), 1))  # keep the whole purge ~5 pauses long
        for r in reversed(result.eliminated):
            detail = (f"   DISQUALIFIED: only {r.total_trades} trades" if r.disqualified else
                      f"   OFF PACE: 1 trade every {r.trade_pace} · fitness {fmt_num(r.fitness, signed=True)}"
                      if not r.on_pace else
                      f"   fitness {fmt_num(r.fitness, signed=True)} · 1 trade every {r.trade_pace} · "
                      f"win {r.win_rate:.0%} · profit factor {r.profit_factor:.2f}")
            line = Text.assemble(
                ("[ELIMINATED]", "bold white on red"),
                " ",
                (f"{r.name} purged (PnL: {r.net_profit_pct:+.1f}%)", "bold red"),
                (detail, "red"),
            )
            self.console.print(line)
            self.pause(per_line)

    def purge_verification(self, result: "SelectionResult") -> None:
        if result.purge_verified:
            self.console.print(Text(
                f"🧹 Memory purge verified: {result.purged_count}/{len(result.eliminated)} eliminated bots "
                f"garbage-collected (0 lingering references)",
                style="bright_black",
            ))
        else:
            self.console.print(Text(
                f"⚠ Purge incomplete: {result.lingering_refs} eliminated bot(s) still referenced in memory",
                style="bold yellow",
            ))

    def victory_banner(self, result: "SelectionResult", clones_to_spawn: int) -> None:
        r = result.victor_report
        if clones_to_spawn > 0:
            headline = f"[SURVIVOR] {r.name} crowned Alpha! Replicating {clones_to_spawn} mutated clones..."
        else:
            headline = f"[SURVIVOR] {r.name} crowned Alpha! Final champion of this run."

        if result.incumbent_retained:
            story = f"Defended the crown · reign length {r.generations_survived + 1} generation(s)"
        else:
            story = f"A mutant clone dethroned its own parent, Bot #{r.parent_id}!"

        stats = (
            f"Avg PnL {fmt_pct(r.net_profit_pct, digits=2)}  │  1 trade every {r.trade_pace}  │  Win {r.win_rate:.0%}  │  "
            f"Profit factor {r.profit_factor:.2f}  │  "
            f"Profitable in {r.folds_profitable}/{r.n_folds} periods  │  Worst DD {r.max_drawdown:.1%}  │  "
            f"Fitness {fmt_num(r.fitness, signed=True)}"
        )
        if r.disqualified:
            stats += "  │  ⚠ every bot was under the trade minimum"
        elif not r.on_pace:
            stats += "  │  ⚠ no bot hit the target pace yet - closest pace wins"
        body = Group(
            Align.center(Text(headline, style="bold bright_green")),
            Align.center(Text(story, style="green")),
            Align.center(Text(stats, style="white")),
        )
        self.console.print(Panel(body, title="👑 SOLE SURVIVOR 👑", border_style="bright_green", box=box.DOUBLE))
        self.pause(1.5)

    def mutation_rate_update(self, old: float, new: float, reason: str) -> None:
        arrow = "▲" if new > old else ("▼" if new < old else "=")
        self.console.print(Text(f"🎛  Mutation rate {old:.0%} → {new:.0%} {arrow}  ({reason})", style="magenta"))

    def mutation_table(
        self,
        generation: int,
        alpha_id: int,
        alpha: TradingGenome,
        clones: Sequence[tuple[int, TradingGenome]],
        mutation_rate: float,
        max_columns: int = 5,
    ) -> None:
        """Gene-by-gene comparison of the Alpha and its freshly mutated clones (first few shown)."""
        shown = list(clones)[:max_columns]
        more = f" (showing {len(shown)})" if len(clones) > len(shown) else ""
        table = Table(
            title=f"🧪 MUTATION LAB · Generation {generation} · {len(clones)} clones of Bot #{alpha_id}{more} "
                  f"· mutation rate {mutation_rate:.0%}",
            title_style="bold magenta",
            box=box.SIMPLE_HEAVY,
            header_style="bold",
        )
        table.add_column("Gene", style="bold")
        table.add_column(f"👑 #{alpha_id}", justify="right", style="bold bright_green")
        for clone_id, _ in shown:
            table.add_column(f"#{clone_id}", justify="right")
        table.add_column("Avg drift (all)", justify="right", style="bright_black")

        for gene in TradingGenome.GENE_NAMES:
            base = getattr(alpha, gene)
            cells: list[Text | str] = [gene, fmt_gene(gene, base)]
            drifts = [abs(getattr(g, gene) - base) / abs(base) * 100 if base else 0.0 for _, g in clones]
            for _, genome in shown:
                value = getattr(genome, gene)
                delta = value - base
                if delta == 0:
                    cells.append(Text(f"{fmt_gene(gene, value)} ·", style="bright_black"))
                else:
                    arrow, style = ("▲", "green") if delta > 0 else ("▼", "red")
                    cells.append(Text(f"{fmt_gene(gene, value)} {arrow}", style=style))
            cells.append(f"{sum(drifts) / len(drifts):.1f}%" if drifts else "-")
            table.add_row(*cells)
        self.console.print(table)
        self.pause()

    def checkpoint_saved(self, path_label: str) -> None:
        self.console.print(Text(f"💾 Alpha genome saved → {path_label}", style="bright_black"))

    # ------------------------------------------------------------------ #
    # End of run
    # ------------------------------------------------------------------ #
    def final_summary(self, summary: "EvolutionSummary") -> None:
        self.console.print()
        self.console.print(Rule("[bold magenta]🏆 EVOLUTION COMPLETE 🏆[/]", style="magenta"))
        if summary.interrupted:
            self.warn("Run interrupted by user - showing results up to the last completed generation.")
        if not summary.history:
            self.warn("No generation completed, nothing to summarise.")
            return

        # Crown history (reigns)
        reigns = Table(title="👑 Crown History", title_style="bold", box=box.SIMPLE_HEAVY)
        reigns.add_column("Alpha")
        reigns.add_column("Reign", justify="right")
        reigns.add_column("Gens", justify="right")
        reigns.add_column("Best Fitness", justify="right")
        reigns.add_column("Best PnL", justify="right")
        for reign in summary.reigns():
            reigns.add_row(
                f"Bot #{reign['bot_id']}",
                f"Gen {reign['start']}–{reign['end']}",
                str(reign['end'] - reign['start'] + 1),
                f"{fmt_num(reign['best_fitness'], signed=True)}",
                Text(fmt_pct(reign['best_pnl'], digits=2), style=pnl_style(reign['best_pnl'])),
            )
        self.console.print(reigns)
        fitness = [h.victor.fitness for h in summary.history]
        self.console.print(Text(f"Survivor fitness per generation: {sparkline(fitness)}  "
                                f"({fmt_num(fitness[0], signed=True)} → {fmt_num(fitness[-1], signed=True)})", style="cyan"))

        # Genetic drift from the Generation-1 ancestor
        drift = Table(title=f"🧬 Genetic Drift: {summary.origin_label} → Champion", title_style="bold", box=box.SIMPLE_HEAVY)
        drift.add_column("Gene", style="bold")
        drift.add_column("Origin", justify="right")
        drift.add_column("Champion", justify="right", style="bold bright_green")
        drift.add_column("Drift", justify="right")
        for gene in TradingGenome.GENE_NAMES:
            a, b = getattr(summary.origin_genome, gene), getattr(summary.champion_genome, gene)
            change = (b - a) / a * 100 if a else 0.0
            drift.add_row(gene, fmt_gene(gene, a), fmt_gene(gene, b),
                          Text(f"{change:+.0f}%", style="bright_black" if abs(change) < 0.5 else ("green" if change > 0 else "red")))
        self.console.print(drift)

        # The honest test
        if summary.showdown:
            showdown = Table(title="🔬 Showdown: In-Sample vs Out-of-Sample", title_style="bold", box=box.ROUNDED)
            showdown.add_column("Strategy", style="bold")
            showdown.add_column("In-sample PnL", justify="right")
            if summary.has_test_window:
                for col in ("Out-of-sample PnL", "OOS Win %", "OOS Sharpe", "OOS Max DD", "OOS Fitness"):
                    showdown.add_column(col, justify="right")
            for name, row in summary.showdown.items():
                cells: list[Text | str] = [name, Text(fmt_pct(row["in_sample"].net_profit_pct, digits=2),
                                                      style=pnl_style(row["in_sample"].net_profit_pct))]
                oos = row.get("out_of_sample")
                if summary.has_test_window and oos is not None:
                    cells += [
                        Text(fmt_pct(oos.net_profit_pct, digits=2), style=pnl_style(oos.net_profit_pct)),
                        f"{oos.win_rate:.0%}" if oos.win_rate is not None else "-",
                        f"{fmt_num(oos.sharpe)}",
                        f"{oos.max_drawdown:.1%}",
                        f"{fmt_num(oos.fitness, signed=True)}",
                    ]
                showdown.add_row(*cells)
            self.console.print(showdown)
            if summary.has_test_window:
                self.console.print(Text(
                    "In-sample numbers are flattering by construction (the bots evolved on that data). "
                    "The out-of-sample columns are the honest result.",
                    style="italic bright_black",
                ))

        paths = [f"Champion genome: [bold]{summary.checkpoint_label}[/]"]
        if summary.chart_label:
            paths.append(f"Evolution chart: [bold]{summary.chart_label}[/]")
        paths.append(f"Generation log:  [bold]{summary.history_label}[/]")
        self.console.print(Panel("\n".join(paths), border_style="magenta", title="Artifacts"))

    # ------------------------------------------------------------------ #
    # Mode 2 - live paper trading
    # ------------------------------------------------------------------ #
    def genome_panel(self, genome: TradingGenome, title: str) -> None:
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Gene", style="bold")
        table.add_column("Value", justify="right", style="bright_green")
        for gene in TradingGenome.GENE_NAMES:
            table.add_row(gene, fmt_gene(gene, getattr(genome, gene)))
        self.console.print(Panel(table, title=title, border_style="green", expand=False))

    def live_header(self, account: "AccountSummary | None", dry_run: bool, data_source: str, symbols: Sequence[str]) -> None:
        mode = "[bold yellow]DRY RUN - no orders will be sent[/]" if dry_run else "[bold green]PAPER ORDERS ENABLED[/]"
        lines = [f"Mode: {mode}", f"Signals from: [bold]{data_source}[/] daily bars", f"Symbols: [bold]{', '.join(symbols)}[/]"]
        if account is not None:
            lines.append(
                f"Paper account: equity [bold]${account.equity:,.2f}[/] · cash ${account.cash:,.2f} · "
                f"buying power ${account.buying_power:,.2f} · status {account.status}"
            )
        self.console.print(Panel("\n".join(lines), title="📡 LIVE PAPER EXECUTION", border_style="cyan"))

    def live_cycle(self, cycle: int, timestamp: str, decisions: Sequence["LiveDecision"]) -> None:
        table = Table(title=f"Cycle {cycle} · {timestamp}", box=box.ROUNDED, header_style="bold")
        for col, justify in (("Symbol", "left"), ("Price", "right"), ("Position", "right"), ("Signal bar", "left"),
                             ("Action", "left"), ("Reason", "left"), ("Order", "left")):
            table.add_column(col, justify=justify)  # type: ignore[arg-type]
        action_styles = {"BUY": "bold bright_green", "SELL": "bold red", "CLOSE_POSITION": "bold magenta", "HOLD": "bright_black"}
        for d in decisions:
            table.add_row(
                d.symbol,
                f"${d.price:,.2f}" if d.price else "-",
                f"{d.position_qty:g}" if d.position_qty else "flat",
                d.signal_date or "-",
                Text(d.action.value, style=action_styles.get(d.action.value, "white")),
                d.reason,
                d.order_status,
            )
        self.console.print(table)
