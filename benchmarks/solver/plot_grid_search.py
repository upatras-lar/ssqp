#!/usr/bin/env python3
"""Plot how runtime and iterations scale with problem size for a Duffing
oscillator grid search (see run_benchmark.py's --duffing-m/--duffing-horizon).

Reads the same per-solver JSON files run_benchmark.py writes for every
(m, horizon) grid point (reusing plot_benchmarks.py's load_results
unchanged) and, for each solver, lines runtime (left axis, solid) and
iterations (right axis, dashed) via a shared twin y-axis, against one
swept variable, in one subplot row per value of the other. Produces a
single image with two such columns side by side: m on the x-axis
(faceted by horizon), and horizon on the x-axis (faceted by m). A grid
point with no result for a given solver (not run, or run at a different
tolerance) just leaves a gap in that solver's line rather than failing
the whole plot, since a partially finished sweep is the common case,
not an error; a point where the solver ran but didn't converge is
marked with a red X on top of its usual marker.

Example
-------
    python plot_grid_search.py duffing_local \\
        --duffing-m 8 16 32 64 --duffing-horizon 20 40 80
    python plot_grid_search.py duffing_local \\
        --duffing-m 8 16 32 64 --duffing-horizon 20 40 80 \\
        --tolerance 1e-2 1e-3
"""

import argparse
from pathlib import Path

import numpy as np

from common.benchmark_config import TOLERANCE, tolerance_label
from plot_benchmarks import (
    DEFAULT_COLOR,
    DUFFING_PROBLEMS,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    SOLVER_COLORS,
    STATUS_CRITICAL,
    SURFACE,
    _style_axes,
    _use_plain_log_labels,
    load_results,
    runtime_unit,
)

METRICS = (
    ("runtime_ms", "Runtime [ms]"),
    ("iterations", "Iterations"),
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "problem",
        choices=sorted(DUFFING_PROBLEMS),
        help="The Duffing problem the grid search was run on.",
    )
    parser.add_argument(
        "--duffing-m",
        type=int,
        nargs="+",
        required=True,
        help="The m values swept (matches run_benchmark.py's own flag).",
    )
    parser.add_argument(
        "--duffing-horizon",
        type=int,
        nargs="+",
        required=True,
        help=(
            "The horizon values swept (matches run_benchmark.py's own "
            "flag)."
        ),
    )
    parser.add_argument(
        "-d", "--data-dir", type=Path, default=Path(__file__).parent / "data"
    )
    parser.add_argument(
        "-t",
        "--tolerance",
        type=float,
        nargs="+",
        default=[TOLERANCE],
        help=(
            "Convergence tolerance the grid search was run with. Provide "
            "multiple values to plot each one."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output image path (default: <data-dir>/plots/...).",
    )
    parser.add_argument(
        "-s",
        "--solvers",
        nargs="+",
        choices=list(SOLVER_COLORS),
        help="Limit the plot to these solvers, instead of every solver with results.",
    )
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def load_grid_results(
    data_dir, problem, tolerance, grid_m, grid_horizon, solvers=None
):
    """{horizon: {m: {solver: benchmark_dict}}}, dropping grid points with
    nothing on disk rather than aborting the whole sweep"""
    grid = {}
    for horizon in grid_horizon:
        grid[horizon] = {}
        for m in grid_m:
            suffix = f"_m{m}_N{horizon}"
            try:
                grid[horizon][m] = load_results(
                    data_dir, problem, tolerance, suffix, solvers
                )
            except SystemExit:
                continue
    return grid


def _grid_solvers(grid):
    return sorted(
        {
            solver
            for m_results in grid.values()
            for benchmarks in m_results.values()
            for solver in benchmarks
        }
    )


def _axis_log_scale(values):
    positive = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    return len(positive) >= 2 and max(positive) / min(positive) >= 10


METRIC_STYLE = {
    "runtime_ms": dict(linestyle="-", marker="o", markersize=6),
    "iterations": dict(linestyle="--", marker="s", markersize=5),
}

# Iteration counts are small integers, so it's common for several solvers
# to land on the exact same value at every grid point -- their dashed
# lines would then sit perfectly on top of each other, hiding all but the
# last-drawn solver. Nudging each solver's iterations series sideways by
# a small fraction of the gap between neighboring x ticks keeps identical
# lines visually separated without moving them off their true x tick, or
# meaningfully changing where they sit on the x axis. Sized as a total
# spread (not per-solver), so it doesn't balloon with the solver count
ITERATIONS_DODGE_SPREAD_FRAC = 0.3


def _dodge_offsets(x_values, solvers, log_x):
    """{solver: x-offset} for the iterations series, spacing solvers
    evenly across a small window centered on each true x value -- additive
    on a linear x axis, multiplicative (so the visual spacing stays even)
    on a log one. None when there's nothing to space against (a single
    x value, or a single solver)"""
    if len(solvers) < 2 or len(x_values) < 2:
        return {s: (1.0 if log_x else 0.0) for s in solvers}

    if log_x:
        min_gap = min(
            x_values[i + 1] / x_values[i] for i in range(len(x_values) - 1)
        )
        spread = min_gap**ITERATIONS_DODGE_SPREAD_FRAC
    else:
        min_gap = min(
            x_values[i + 1] - x_values[i] for i in range(len(x_values) - 1)
        )
        spread = min_gap * ITERATIONS_DODGE_SPREAD_FRAC

    positions = np.linspace(-0.5, 0.5, len(solvers))
    if log_x:
        return {s: spread**p for s, p in zip(solvers, positions)}
    return {s: p * spread for s, p in zip(solvers, positions)}


def _render_block(
    subfig,
    x_values,
    facet_values,
    x_label,
    facet_label,
    lookup,
    solvers,
    colors,
    log_scale,
    section_title,
    runtime_divisor,
    runtime_label,
):
    """Draws one column of len(facet_values) panels (m or horizon on the
    x-axis, the other swept variable as facet rows), each combining
    runtime (left axis, solid) and iterations (right axis, dashed) via a
    twin y-axis, into the given subfigure"""
    subfig.set_facecolor(SURFACE)
    subfig.suptitle(
        section_title, fontsize=12, fontweight="bold", color=INK_PRIMARY
    )
    axes = subfig.subplots(len(facet_values), 1, squeeze=False)[:, 0]

    log_x = len(x_values) >= 2 and x_values[-1] / max(x_values[0], 1) >= 10

    for row, facet in enumerate(facet_values):
        ax = axes[row]
        _style_axes(ax)
        ax2 = ax.twinx()
        ax2.set_facecolor(SURFACE)
        ax2.grid(False)
        for side in ("top", "left", "bottom"):
            ax2.spines[side].set_visible(False)
        ax2.spines["right"].set_color(INK_SECONDARY)
        ax2.tick_params(colors=INK_MUTED, labelsize=9)

        benchmarks_by_x = {x: lookup(facet, x) for x in x_values}
        metric_axes = {"runtime_ms": ax, "iterations": ax2}
        iter_dodge = _dodge_offsets(x_values, solvers, log_x)

        for solver in solvers:
            for key, _ in METRICS:
                metric_ax = metric_axes[key]
                ys = np.full(len(x_values), np.nan)
                failed_idx = []
                for i, x in enumerate(x_values):
                    benchmark = benchmarks_by_x[x].get(solver)
                    if benchmark is None:
                        continue
                    value = benchmark.get(key)
                    if value is not None:
                        ys[i] = (
                            value / runtime_divisor
                            if key == "runtime_ms"
                            else value
                        )
                    if benchmark.get("success") is False:
                        failed_idx.append(i)
                if np.all(np.isnan(ys)):
                    continue
                if key == "iterations":
                    op = (lambda a, b: a * b) if log_x else (lambda a, b: a + b)
                    xs = [op(x, iter_dodge[solver]) for x in x_values]
                else:
                    xs = x_values
                metric_ax.plot(
                    xs,
                    ys,
                    color=colors[solver],
                    linewidth=2,
                    zorder=3,
                    **METRIC_STYLE[key],
                )
                if failed_idx:
                    metric_ax.scatter(
                        [xs[i] for i in failed_idx],
                        [ys[i] for i in failed_idx],
                        marker="x",
                        color=STATUS_CRITICAL,
                        s=70,
                        linewidths=2,
                        zorder=4,
                    )

        if log_x:
            ax.set_xscale("log")
            _use_plain_log_labels(ax.xaxis)
        if log_scale["runtime_ms"]:
            ax.set_yscale("log")
            _use_plain_log_labels(ax.yaxis)
        if log_scale["iterations"]:
            ax2.set_yscale("log")
            _use_plain_log_labels(ax2.yaxis)
        ax.set_xticks(x_values)
        ax.set_xticklabels([str(x) for x in x_values])

        ax.set_title(
            f"{facet_label} = {facet}", fontsize=11, fontweight="bold", pad=10
        )
        ax.set_xlabel(x_label, fontsize=10, color=INK_SECONDARY)
        ax.set_ylabel(runtime_label, fontsize=10, color=INK_SECONDARY)
        ax2.set_ylabel("Iterations", fontsize=10, color=INK_SECONDARY)


def plot_grid(problem, grid, grid_m, grid_horizon, tolerance, output, show):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid_m = sorted(grid_m)
    grid_horizon = sorted(grid_horizon)

    solvers = _grid_solvers(grid)
    if not solvers:
        raise SystemExit(
            f"No results for {problem!r} at tolerance {tolerance!r} found "
            f"across m={grid_m}, horizon={grid_horizon}"
        )
    colors = {s: SOLVER_COLORS.get(s, DEFAULT_COLOR) for s in solvers}

    # One log-scale decision per metric, shared by both blocks below --
    # they plot the exact same underlying values, just against a
    # different swept variable, so the same metric should read on the
    # same kind of scale in both
    log_scale = {}
    for key, _ in METRICS:
        values = (
            b.get(key)
            for m_results in grid.values()
            for benchmarks in m_results.values()
            for b in benchmarks.values()
        )
        log_scale[key] = _axis_log_scale(values)

    runtime_values = (
        b.get("runtime_ms")
        for m_results in grid.values()
        for benchmarks in m_results.values()
        for b in benchmarks.values()
    )
    runtime_divisor, runtime_label = runtime_unit(runtime_values)

    nrows = max(len(grid_horizon), len(grid_m))
    block_w = 6.3  # one twin-axis panel needs more width than height
    row_h = 3.8
    fig = plt.figure(
        figsize=(2 * block_w + 1.2, row_h * nrows + 1.9),
        constrained_layout=True,
    )
    fig.set_facecolor(SURFACE)
    # Reserves a fixed top/bottom strip, sized in inches rather than a
    # flat figure-fraction, for the title/subtitle/metric-legend above and
    # the solver legend below -- constrained_layout doesn't know about
    # the separate fig.text()/fig.legend() calls, and rect is (left,
    # bottom, width, height), not (left, bottom, right, top)
    fig_h = fig.get_size_inches()[1]
    top_in, bottom_in = 1.35, 0.55
    fig.get_layout_engine().set(
        rect=(0, bottom_in / fig_h, 1, 1 - (top_in + bottom_in) / fig_h)
    )
    fig.suptitle(
        f"{problem}: grid search",
        color=INK_PRIMARY,
        fontsize=15,
        fontweight="bold",
        y=1 - 0.32 / fig_h,
    )
    fig.text(
        0.5,
        1 - 0.72 / fig_h,
        f"All solvers run to the same nominal tolerance ({tolerance:g})",
        ha="center",
        color=INK_SECONDARY,
        fontsize=10,
    )
    metric_handles = [
        plt.Line2D(
            [0], [0], color=INK_SECONDARY, linewidth=2, marker="o", markersize=5
        ),
        plt.Line2D(
            [0],
            [0],
            color=INK_SECONDARY,
            linewidth=2,
            linestyle="--",
            marker="s",
            markersize=5,
        ),
    ]
    fig.legend(
        metric_handles,
        [f"{runtime_label} (left axis)", "Iterations (right axis)"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 1.0 / fig_h),
        ncol=2,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )

    subfig_m, subfig_horizon = fig.subfigures(1, 2, wspace=0.05)
    _render_block(
        subfig_m,
        grid_m,
        grid_horizon,
        "m",
        "horizon",
        lambda facet, x: grid.get(facet, {}).get(x, {}),
        solvers,
        colors,
        log_scale,
        "Scaling with m",
        runtime_divisor,
        runtime_label,
    )
    _render_block(
        subfig_horizon,
        grid_horizon,
        grid_m,
        "horizon",
        "m",
        lambda facet, x: grid.get(x, {}).get(facet, {}),
        solvers,
        colors,
        log_scale,
        "Scaling with horizon",
        runtime_divisor,
        runtime_label,
    )

    handles = [
        plt.Line2D([0], [0], color=colors[s], linewidth=2, marker="o")
        for s in solvers
    ]
    fig.legend(
        handles,
        solvers,
        loc="lower center",
        ncol=len(solvers),
        bbox_to_anchor=(0.5, 0.15 / fig_h),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    print(f"Saved plot to {output}")

    if show:
        plt.show()


def _default_output(data_dir, problem, tolerance):
    tol_label = tolerance_label(tolerance)
    return data_dir / "plots" / f"{problem}_grid_tol_{tol_label}.png"


if __name__ == "__main__":
    args = _parse_args()

    single_run = len(args.tolerance) == 1

    for tolerance in args.tolerance:
        grid = load_grid_results(
            args.data_dir,
            args.problem,
            tolerance,
            args.duffing_m,
            args.duffing_horizon,
            args.solvers,
        )
        output = (
            args.output
            if args.output and single_run
            else _default_output(args.data_dir, args.problem, tolerance)
        )
        try:
            plot_grid(
                args.problem,
                grid,
                args.duffing_m,
                args.duffing_horizon,
                tolerance,
                output,
                args.show,
            )
        except SystemExit as exc:
            if single_run:
                raise
            print(exc)
