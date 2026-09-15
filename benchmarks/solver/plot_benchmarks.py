#!/usr/bin/env python3
"""Plot per-solver runtime, iteration count, objective, and constraint
violation for one benchmark problem

Reads the JSON files written by ``run_benchmark.py`` (one per solver, under
``<data-dir>/<solver>/<problem>_tol_<tolerance>.json``) and bars each solver's
reported runtime, iteration count, final objective, and constraint violation
side by side, using the common "benchmark" summary every solver wrapper writes
Constraint violation is the sum, over stages, of the worst single bound
violation at that stage (position/velocity/torque/friction, and the
dynamics-transition defect, folded in here for SSQP's separately-reported
"stage_gap" so every solver's plotted value is on the same footing) --
it is what makes the comparison fair: a solver that is fast or takes few
iterations but returns an infeasible trajectory is not actually better

Example
-------
    python plot_benchmarks.py go2_standing
    python plot_benchmarks.py pendubot --show
    python plot_benchmarks.py pendubot --tolerance 1e-3
    python plot_benchmarks.py --all-problems
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common.benchmark_config import TOLERANCE, tolerance_label

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
        nargs="?",
        choices=[
            "pendubot",
            "go2_standing",
            "go2_jump",
            "go2_wheelie",
            "go2_frog",
            "panda_circle",
            "duffing_local",
            "duffing_global",
        ],
        help="The benchmarking problem to plot. Ignored if --all-problems is given.",
    )
    parser.add_argument(
        "-d", "--data-dir", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        default=TOLERANCE,
        help="Convergence tolerance the results to plot were run with.",
    )
    parser.add_argument(
        "--all-problems",
        action="store_true",
        help="Plot every problem, instead of just the given problem.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (PNG by default). Ignored if --all-problems is given.",
    )
    parser.add_argument(
        "--duffing-m",
        type=int,
        default=None,
        help=(
            "Only meaningful for duffing_local/duffing_global: number of "
            "coupled oscillators the results to plot were run with. Only "
            "needed if that run_benchmark.py run passed a non-default "
            "--duffing-m (defaults to the same preset, 16, if omitted)."
        ),
    )
    parser.add_argument(
        "--duffing-horizon",
        type=int,
        default=None,
        help=(
            "Only meaningful for duffing_local/duffing_global: OCP horizon "
            "the results to plot were run with. Only needed if that "
            "run_benchmark.py run passed a non-default --duffing-horizon "
            "(defaults to the same preset, 80, if omitted)."
        ),
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


DUFFING_PROBLEMS = {"duffing_local", "duffing_global"}
# Matches problems.py's make_duffing_local_problem/make_duffing_global_problem
# presets, and run_benchmark.py's own --duffing-m/--duffing-horizon help text
DUFFING_DEFAULT_M = 16
DUFFING_DEFAULT_HORIZON = 80


def _resolve_duffing_size(duffing_m, duffing_horizon):
    m = duffing_m if duffing_m is not None else DUFFING_DEFAULT_M
    horizon = (
        duffing_horizon
        if duffing_horizon is not None
        else DUFFING_DEFAULT_HORIZON
    )
    return m, horizon


def _duffing_suffix(problem, duffing_m, duffing_horizon):
    """Filename suffix matching run_benchmark.py's own scaled-run naming
    (see run_solver's output_name construction): present only when at
    least one of --duffing-m/--duffing-horizon was passed for a Duffing
    problem, and then always carrying both values, resolved against the
    same presets a bare run_benchmark.py invocation would have used"""
    if problem not in DUFFING_PROBLEMS:
        return ""
    if duffing_m is None and duffing_horizon is None:
        return ""
    m, horizon = _resolve_duffing_size(duffing_m, duffing_horizon)
    return f"_m{m}_N{horizon}"


def _duffing_label(problem, duffing_m, duffing_horizon):
    """Short "m=.., horizon=.." string for the plot's own subtitle, shown
    whenever the problem is a Duffing one (even at the default size), so
    the image itself carries that context and not just its filename"""
    if problem not in DUFFING_PROBLEMS:
        return None
    m, horizon = _resolve_duffing_size(duffing_m, duffing_horizon)
    return f"m={m}, horizon={horizon}"


def load_results(data_dir, problem, tolerance, suffix=""):
    results = {}
    pattern = f"*/{problem}{suffix}_tol_{tolerance_label(tolerance)}.json"
    for path in sorted(Path(data_dir).glob(pattern)):
        try:
            with path.open() as f:
                result = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Skipping {path}: {exc}")
            continue
        benchmark = result.get("benchmark")
        if not benchmark:
            print(f'Skipping {path}: no "benchmark" summary')
            continue
        # SSQP reports the dynamics-transition defect as a separate
        # "stage_gap" field, since its multi-shooting formulation can
        # leave it nonzero at any given iterate, while fatrop/ipopt fold
        # that same defect straight into "constraint_violation" (see
        # CasadiProblem._evaluate_violation), and aligator/mim_solvers'
        # shooting rollouts make it identically zero either way, folding
        # stage_gap into constraint_violation here puts every solver's
        # plotted value on the same footing
        stage_gap = benchmark.get("stage_gap")
        if stage_gap is not None:
            benchmark["constraint_violation"] = (
                benchmark.get("constraint_violation") or 0.0
            ) + stage_gap
        solver = benchmark.get("solver") or path.parent.name
        results[solver] = benchmark
    if not results:
        raise SystemExit(
            f"No results for {problem!r} at tolerance {tolerance!r} found "
            f"below {data_dir}"
        )
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


def _use_plain_log_labels(axis):
    """Replaces a log-scaled axis's default "6 x 10^1"-style major tick
    labels with the plain number instead, via the older plain-digit
    LogFormatter, and drops minor tick labels entirely -- _style_axes's
    tick_params() only recolors major ticks grey, so a minor tick label
    would otherwise show up in matplotlib's default black instead"""
    import matplotlib.ticker as mticker

    axis.set_major_formatter(mticker.LogFormatter())
    axis.set_minor_formatter(mticker.NullFormatter())


def plot_results(problem, results, tolerance, output, show, duffing_label=None):
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
    # Reserves a fixed top strip for the title + subtitle text below,
    # rather than letting constrained_layout size it off the suptitle
    # alone -- it doesn't know about the separate fig.text() subtitle, so
    # a longer subtitle (e.g. with a Duffing m/horizon label appended)
    # would otherwise run into the panel titles right below it. rect is
    # (left, bottom, width, height), not (left, bottom, right, top) --
    # bottom=0 here so width/height and right/top happen to coincide
    fig.get_layout_engine().set(rect=(0, 0, 1, 0.91))
    fig.suptitle(
        f"{problem}: solver comparison",
        color=INK_PRIMARY,
        fontsize=15,
        fontweight="bold",
    )
    subtitle = f"All solvers run to the same nominal tolerance ({tolerance:g})"
    if duffing_label:
        subtitle += f" · {duffing_label}"
    fig.text(
        0.5,
        0.955,
        subtitle,
        ha="center",
        color=INK_SECONDARY,
        fontsize=10,
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
        ax.bar(x, plotted, width=0.6, color=colors, edgecolor="none", zorder=3)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks(x, solvers, rotation=30, ha="right")
        # A log scale keeps the fastest/slowest solvers both legible when
        # they're an order of magnitude or more apart; a linear scale
        # otherwise flattens the smaller bars to invisibility
        positive = plotted[np.isfinite(plotted) & (plotted > 0)]
        if positive.size >= 2 and positive.max() / positive.min() >= 10:
            ax.set_yscale("log")
            _use_plain_log_labels(ax.yaxis)
        for i, ok in enumerate(success):
            if ok is False:
                y = plotted[i] if np.isfinite(plotted[i]) else 0.0
                ax.text(
                    i,
                    y,
                    "failed",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=STATUS_CRITICAL,
                    fontweight="bold",
                    zorder=4,
                )

    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=SOLVER_COLORS.get(s, DEFAULT_COLOR)
        )
        for s in solvers
    ]
    fig.legend(
        handles,
        solvers,
        loc="lower center",
        ncol=len(solvers),
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved plot to {output}")

    if show:
        plt.show()


ALL_PROBLEMS = [
    "pendubot",
    "go2_standing",
    "go2_jump",
    "go2_wheelie",
    "go2_frog",
    "panda_circle",
    "duffing_local",
    "duffing_global",
]


def _default_output(data_dir, problem, tolerance, suffix=""):
    tol_label = tolerance_label(tolerance)
    return data_dir / "plots" / f"{problem}{suffix}_tol_{tol_label}.png"


if __name__ == "__main__":
    args = _parse_args()
    if not args.all_problems and args.problem is None:
        raise SystemExit("Either a problem or --all-problems must be given.")
    if (
        not args.all_problems
        and args.problem not in DUFFING_PROBLEMS
        and (args.duffing_m is not None or args.duffing_horizon is not None)
    ):
        raise SystemExit(
            "--duffing-m/--duffing-horizon only apply to duffing_local/"
            f"duffing_global, not --problem {args.problem}"
        )
    problems_to_plot = ALL_PROBLEMS if args.all_problems else [args.problem]

    for problem in problems_to_plot:
        suffix = _duffing_suffix(problem, args.duffing_m, args.duffing_horizon)
        duffing_label = _duffing_label(
            problem, args.duffing_m, args.duffing_horizon
        )
        try:
            results = load_results(
                args.data_dir, problem, args.tolerance, suffix
            )
        except SystemExit as exc:
            if not args.all_problems:
                raise
            print(exc)
            continue
        if args.output and not args.all_problems:
            output = args.output
        else:
            output = _default_output(
                args.data_dir, problem, args.tolerance, suffix
            )
        plot_results(
            problem, results, args.tolerance, output, args.show, duffing_label
        )
