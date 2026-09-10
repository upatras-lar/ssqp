#!/usr/bin/env python3
"""Plot per-solver runtime, iteration count, objective, and constraint
violation for one benchmark problem.

Reads the JSON files written by ``run_benchmark.py`` (one per solver, under
``<data-dir>/<solver>/<problem>.json``) and bars each solver's reported
runtime, iteration count, final objective, and constraint violation side
by side, using the common "benchmark" summary every solver wrapper writes.
Constraint violation is the sum, over stages, of the worst single bound
violation at that stage (position/velocity/torque/friction, and, for the
direct-transcription solvers, the dynamics-transition defect) -- it is
what makes the comparison fair: a solver that is fast or takes few
iterations but returns an infeasible trajectory is not actually better.
The dashed line on that panel is TOLERANCE (see benchmark_config.py): the
same nominal convergence tolerance every solver is run with.

Example
-------
    python plot_benchmarks.py go2_standing
    python plot_benchmarks.py pendubot --show
"""

import argparse
import json
from pathlib import Path

import numpy as np

from benchmark_config import TOLERANCE

# A fixed identity -> color mapping, assigned once and never reassigned
# based on which solvers happen to have a result for a given problem, since
# otherwise a missing solver would silently repaint every solver after it
# from one plot to the next, using the first six slots (in order) of a
# categorical palette whose *adjacent* pairs (the relevant comparison
# within one bar-chart panel) validate colorblind-safe; since these bars
# are also individually labeled on the x-axis, identity never rests on
# color alone even across the two slots (yellow, magenta) that don't
# clear the stricter all-pairs floor
SOLVER_COLORS = {
    "ssqp-admm": "#2a78d6",  # blue
    "ssqp-hpipm": "#eb6834",  # orange
    "fatrop": "#1baf7a",  # aqua
    "ipopt": "#eda100",  # yellow
    "aligator": "#e87ba4",  # magenta
    "mim_solvers": "#008300",  # green
}
DEFAULT_COLOR = "#898781"  # muted, fallback for a solver not in the map above

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
STATUS_CRITICAL = "#d03b3b"


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "problem",
        choices=[
            "pendubot",
            "go2_standing",
            "go2_jump",
            "go2_frog",
            "panda_circle",
        ],
        help="The benchmarking problem to plot",
    )
    parser.add_argument(
        "-d", "--data-dir", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Output image path (PNG by default)."
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_results(data_dir, problem):
    results = {}
    for path in sorted(Path(data_dir).glob(f"*/{problem}.json")):
        try:
            with path.open() as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
            continue
        benchmark = result.get("benchmark")
        if not benchmark:
            print(f"Skipping {path}: no \"benchmark\" summary")
            continue
        solver = benchmark.get("solver") or path.parent.name
        results[solver] = benchmark
    if not results:
        raise SystemExit(f"No results for {problem!r} found below {data_dir}")
    return results


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", which="major", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.title.set_color(INK_PRIMARY)


def plot_results(problem, results, output, show):
    # Select a non-interactive backend unless the user explicitly requests a
    # window, which keeps the script usable in Docker/CI without a display
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    solvers = sorted(results)
    colors = [SOLVER_COLORS.get(s, DEFAULT_COLOR) for s in solvers]
    runtime_ms = [results[s].get("runtime_ms") for s in solvers]
    iterations = [results[s].get("iterations") for s in solvers]
    objective = [results[s].get("objective") for s in solvers]
    violation = [results[s].get("constraint_violation") for s in solvers]
    success = [results[s].get("success") for s in solvers]
    x = np.arange(len(solvers))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    fig.set_facecolor(SURFACE)
    fig.suptitle(
        f"{problem}: solver comparison", color=INK_PRIMARY, fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5, 0.955,
        f"All solvers run to the same nominal tolerance ({TOLERANCE:g})",
        ha="center", color=INK_SECONDARY, fontsize=10,
    )

    for ax, values, title in (
        (axes[0, 0], runtime_ms, "Runtime [ms]"),
        (axes[0, 1], iterations, "Iterations"),
        (axes[1, 0], objective, "Final objective"),
        (axes[1, 1], violation, "Constraint violation"),
    ):
        plotted = np.array(
            [v if v is not None else np.nan for v in values], dtype=float
        )
        _style_axes(ax)
        ax.bar(
            x, plotted, width=0.6, color=colors, edgecolor="none", zorder=3
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x, solvers, rotation=30, ha="right")
        # A log scale keeps the fastest/slowest solvers both legible when
        # they're an order of magnitude or more apart; a linear scale
        # otherwise flattens the smaller bars to invisibility
        positive = plotted[np.isfinite(plotted) & (plotted > 0)]
        if positive.size >= 2 and positive.max() / positive.min() >= 10:
            ax.set_yscale("log")
        for i, ok in enumerate(success):
            if ok is False:
                y = plotted[i] if np.isfinite(plotted[i]) else 0.0
                ax.text(
                    i, y, "failed", ha="center", va="bottom", fontsize=8,
                    color=STATUS_CRITICAL, fontweight="bold", zorder=4,
                )

    # The one piece of cross-cutting information a bar chart can't carry on
    # its own axis: the shared target every solver's own convergence check
    # is bound to, so a bar that clears it can be read as "feasible", not
    # just "small compared to its neighbors"
    ax_violation = axes[1, 1]
    ax_violation.axhline(
        TOLERANCE, color=INK_SECONDARY, linestyle="--", linewidth=1.2, zorder=2
    )
    ax_violation.text(
        len(solvers) - 0.4, TOLERANCE, f" tolerance ({TOLERANCE:g})",
        color=INK_SECONDARY, fontsize=8, va="bottom", ha="right",
    )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SOLVER_COLORS.get(s, DEFAULT_COLOR))
        for s in solvers
    ]
    fig.legend(
        handles, solvers, loc="lower center", ncol=len(solvers),
        bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9,
        labelcolor=INK_SECONDARY,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved plot to {output}")

    if show:
        plt.show()


if __name__ == "__main__":
    args = _parse_args()
    results = load_results(args.data_dir, args.problem)
    output = args.output or (args.data_dir / "plots" / f"{args.problem}.png")
    plot_results(args.problem, results, output, args.show)
