"""
Matplotlib evolution report (a PNG you can drop straight into a video edit).

Two stacked panels, each with ONE y-axis:
  1. Survivor fitness per generation (crown changes marked)
  2. Equity curves on the out-of-sample window: evolved champion vs the
     Generation-1 textbook bot vs equal-weight buy & hold
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Validated categorical slots 1-3 (safe for colour-blind viewers as a set of three).
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def render_evolution_report(
    generations: list[int],
    fitness: list[float],
    crown_changes: list[int],
    curves: dict[str, pd.Series],
    curves_title: str,
    output_path: Path,
) -> Path | None:
    """Save the two-panel PNG. Returns the path, or None if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless - never pops a window mid-recording
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    except ImportError:
        log.warning("matplotlib not installed - skipping evolution chart")
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "text.color": TEXT_PRIMARY,
    })
    fig, (ax_fit, ax_eq) = plt.subplots(2, 1, figsize=(11, 8), facecolor=SURFACE,
                                        gridspec_kw={"height_ratios": [1, 1.3]})
    for ax in (ax_fit, ax_eq):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # --- Panel 1: survivor fitness -------------------------------------------
    ax_fit.plot(generations, fitness, color=SERIES_COLORS[0], linewidth=2,
                marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
    changed = [(g, f) for g, f in zip(generations, fitness) if g in set(crown_changes)]
    if changed:
        gx, fy = zip(*changed)
        ax_fit.scatter(gx, fy, s=90, facecolor=SURFACE, edgecolor=SERIES_COLORS[0],
                       linewidth=2, zorder=3, label="New Alpha crowned")
        ax_fit.legend(frameon=False, loc="best", labelcolor=TEXT_SECONDARY)
    ax_fit.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax_fit.set_title("Survivor fitness by generation", loc="left", fontsize=13,
                     fontweight="bold", color=TEXT_PRIMARY)
    ax_fit.set_xlabel("Generation")
    ax_fit.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_fit.set_ylabel("Fitness")
    if generations:
        ax_fit.annotate(f"{fitness[-1]:.2f}", (generations[-1], fitness[-1]),
                        textcoords="offset points", xytext=(8, 0), va="center",
                        color=TEXT_PRIMARY, fontweight="bold")

    # --- Panel 2: equity curves ---------------------------------------------
    end_labels: list[tuple[float, str]] = []
    for (label, curve), color in zip(curves.items(), SERIES_COLORS):
        if curve.empty:
            continue
        ax_eq.plot(curve.index, curve.values, color=color, linewidth=2, label=label)
        short_label = label.split(" (")[0]  # full name lives in the legend
        end_labels.append((float(curve.iloc[-1]), f"{short_label}  ${curve.iloc[-1]:,.0f}"))
    ax_eq.set_title(curves_title, loc="left", fontsize=13, fontweight="bold", color=TEXT_PRIMARY)
    ax_eq.set_ylabel("Equity ($)")
    ax_eq.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax_eq.legend(frameon=False, loc="best", labelcolor=TEXT_SECONDARY)
    ax_eq.margins(x=0.02)
    _place_end_labels(ax_eq, end_labels, fig)

    fig.tight_layout()
    fig.subplots_adjust(right=0.80)  # room for end-of-line labels
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    except OSError as exc:
        log.warning("Could not save chart %s: %s", output_path, exc)
        return None
    finally:
        plt.close(fig)
    return output_path


def _place_end_labels(ax, labels: list[tuple[float, str]], fig) -> None:
    """Direct labels at the right edge, nudged apart so close finishes don't overlap."""
    from matplotlib.transforms import blended_transform_factory, offset_copy

    if not labels:
        return
    lo, hi = ax.get_ylim()
    min_gap = (hi - lo) * 0.045
    ordered = sorted(labels)
    placed: list[float] = []
    for y, _ in ordered:
        placed.append(max(y, placed[-1] + min_gap) if placed else y)
    overflow = placed[-1] - hi
    if overflow > 0:  # pushed off the top -> shift the whole stack down
        placed = [p - overflow for p in placed]

    # x in axes coordinates (just past the right edge), y in data coordinates
    base = blended_transform_factory(ax.transAxes, ax.transData)
    transform = offset_copy(base, fig=fig, x=6, units="points")
    for (_, text), y in zip(ordered, placed):
        ax.text(1.0, y, text, transform=transform, va="center", ha="left",
                color=TEXT_PRIMARY, fontsize=9, clip_on=False)


# Categorical slots 1-4 in fixed order (validated on the adjacent pairlist for line charts).
LAB_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def render_lab_report(summary, data, output_path: Path) -> Path | None:
    """Lab PNG: (1) each island's Alpha fitness per generation, (2) the final-test equity race."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    except ImportError:
        log.warning("matplotlib not installed - skipping lab chart")
        return None
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=SURFACE,
                                   gridspec_kw={"height_ratios": [1, 1.3]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=TEXT_SECONDARY)

    fit = np.array(summary.island_fitness) if summary.island_fitness else np.zeros((0, 0))
    if fit.size:
        for i in range(fit.shape[1]):
            ax1.plot(summary.generations, fit[:, i], color=LAB_COLORS[i % 4], linewidth=2, label=f"Island {i + 1}")
        ax1.legend(frameon=False, loc="best", labelcolor=TEXT_SECONDARY, ncol=min(fit.shape[1], 4))
    ax1.axhline(0, color=TEXT_SECONDARY, linewidth=0.8)
    ax1.set_title("Each island's Alpha fitness by generation", loc="left", fontsize=13, fontweight="bold",
                  color=TEXT_PRIMARY)
    ax1.set_xlabel("Generation", color=TEXT_SECONDARY)
    ax1.set_ylabel("Fitness", color=TEXT_SECONDARY)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    labels = []
    for s, color in zip(summary.showdown, LAB_COLORS):
        eq = s.equity_daily
        ax2.plot(np.arange(len(eq)), eq, color=color, linewidth=2, label=s.name)
        labels.append((float(eq[-1]), f"{s.name.split(' (')[0]}  {s.total_return:+.1%}"))
    ax2.set_title(f"Final test on untouched data ({summary.windows.get('test', '')})", loc="left", fontsize=13,
                  fontweight="bold", color=TEXT_PRIMARY)
    ax2.set_xlabel("Trading days into the final test", color=TEXT_SECONDARY)
    ax2.set_ylabel("Equity ($)", color=TEXT_SECONDARY)
    ax2.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
    ax2.legend(frameon=False, loc="best", labelcolor=TEXT_SECONDARY)
    ax2.margins(x=0.02)
    fig.tight_layout()
    fig.subplots_adjust(right=0.80)
    _place_end_labels(ax2, labels, fig)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    except OSError as exc:
        log.warning("Could not save chart %s: %s", output_path, exc)
        return None
    finally:
        plt.close(fig)
    return output_path


def render_forecast_report(summary, output_path: Path) -> Path | None:
    """(1) accuracy by generation vs the naive line, (2) the final-test payoff."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, PercentFormatter, StrMethodFormatter
    except ImportError:
        return None
    import numpy as np

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), facecolor=SURFACE, gridspec_kw={"height_ratios": [1, 1.1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=TEXT_SECONDARY)

    g = summary.generations
    ax1.plot(g, summary.train_acc, color=LAB_COLORS[0], linewidth=2, label="Best Alpha, training years")
    ax1.plot(g, np.array(summary.val_acc, dtype=float), color=LAB_COLORS[1], linewidth=2,
             label="Hall-of-Fame leader, unseen validation years")
    ax1.axhline(summary.baseline_train, color=LAB_COLORS[0], linewidth=1, linestyle="--")
    ax1.axhline(summary.baseline_val, color=LAB_COLORS[1], linewidth=1, linestyle="--")
    ax1.set_title("Prediction accuracy by generation (dashed lines = naive guess)", loc="left", fontsize=13,
                  fontweight="bold", color=TEXT_PRIMARY)
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Generation", color=TEXT_SECONDARY)
    ax1.legend(frameon=False, loc="best", labelcolor=TEXT_SECONDARY)

    if summary.target == "direction" and summary.equity:
        labels = []
        shown = [(n, e) for n, e in summary.equity.items() if not n.startswith(("Coin", "Naive"))]
        for (name, eq), color in zip(shown, LAB_COLORS):
            ax2.plot(np.arange(len(eq)), eq, color=color, linewidth=2, label=name)
            labels.append((float(eq[-1]), f"{name.split(' (')[0]}  {eq[-1] / eq[0] - 1:+.1%}"))
        ax2.set_title(f"If you traded the UP calls on the final test ({summary.windows['test']})", loc="left",
                      fontsize=13, fontweight="bold", color=TEXT_PRIMARY)
        ax2.yaxis.set_major_formatter(StrMethodFormatter("${x:,.0f}"))
        ax2.set_xlabel("Trading days into the final test", color=TEXT_SECONDARY)
        ax2.legend(frameon=False, loc="best", labelcolor=TEXT_SECONDARY)
        fig.tight_layout()
        fig.subplots_adjust(right=0.80)
        _place_end_labels(ax2, labels, fig)
    else:
        rows = [r for r in summary.results if not r.name.startswith("Coin")]
        names = [r.name.split(" (")[0].replace("Naive: ", "") for r in rows]
        acc = [r.accuracy for r in rows]
        bars = ax2.bar(names, acc, color=[LAB_COLORS[i % 4] for i in range(len(rows))], width=0.6)
        base = rows[-1].baseline if rows else 0.5
        ax2.axhline(base, color=TEXT_SECONDARY, linestyle="--", linewidth=1)
        for b, a in zip(bars, acc):
            ax2.annotate(f"{a:.1%}", (b.get_x() + b.get_width() / 2, a), textcoords="offset points", xytext=(0, 4),
                         ha="center", color=TEXT_PRIMARY, fontsize=10)
        if acc:
            ax2.set_ylim(min(acc + [base]) - 0.03, max(acc) + 0.03)
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        title = "Final-test accuracy (dashed = naive guess)"
        if summary.move_ratio and all(np.isfinite(summary.move_ratio)):
            big, calm = summary.move_ratio
            title += f"\nChampion's BIG calls moved {big:.2%} on average vs {calm:.2%} on its CALM calls"
        ax2.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY)
        fig.tight_layout()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    except OSError:
        return None
    finally:
        plt.close(fig)
    return output_path


def render_prophecy_report(summary, output_path: Path) -> Path | None:
    """Running accuracy of the reigning Alpha vs the naive guess, day by day through the replay."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import PercentFormatter
    except ImportError:
        return None
    import numpy as np

    a = np.array(summary.alpha_daily, dtype=float)
    n = np.array(summary.naive_daily, dtype=float)
    if not len(a):
        return None
    run_a = np.nancumsum(a) / np.arange(1, len(a) + 1)
    run_n = np.nancumsum(n) / np.arange(1, len(n) + 1)
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY)
    x = np.arange(len(a))
    ax.plot(x, run_a, color=LAB_COLORS[0], linewidth=2, label="Reigning Alpha (running accuracy)")
    ax.plot(x, run_n, color=LAB_COLORS[1], linewidth=2, label="Naive guess (running accuracy)")
    ax.axhline(0.5, color=TEXT_SECONDARY, linewidth=1, linestyle="--")
    crowns = [i for i, h in enumerate(summary.league.history) if h["dethroned"]]
    for i in crowns:
        pos = min((i + 1) * 5 - 1, len(x) - 1)
        ax.axvline(pos, color=GRID, linewidth=1)
    ax.set_title(f"Prophecy League replay · {summary.target} · {summary.days[0]} → {summary.days[-1]} "
                 f"(thin lines = a new Alpha was crowned)", loc="left", fontsize=12, fontweight="bold",
                 color=TEXT_PRIMARY)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("Trading days into the replay", color=TEXT_SECONDARY)
    ax.legend(frameon=False, loc="best", labelcolor=TEXT_SECONDARY)
    lo = min(run_a[5:].min() if len(a) > 5 else run_a.min(), run_n.min(), 0.5) - 0.01
    hi = max(run_a[5:].max() if len(a) > 5 else run_a.max(), run_n.max(), 0.5) + 0.01
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    except OSError:
        return None
    finally:
        plt.close(fig)
    return output_path
