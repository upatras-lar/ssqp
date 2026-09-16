import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import argparse

# Categorical palette (colorblind-safe, fixed order - never cycled per series
# count) and chart chrome, shared across all panels
PALETTE = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

CHART_SURFACE = "#fcfcfb"
PAGE_PLANE = "#f9f9f7"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def pretty_label(stem, prefix):
    return stem.replace(prefix, "").replace("_", " ").title()


def axis_log_scale(values):
    """Whether an axis plotting these values should use a log scale --
    true once the largest and smallest positive values are far enough
    apart (>=10x) that a linear scale would flatten the smaller ones to
    invisibility"""
    positive = [v for v in values if v is not None and np.isfinite(v) and v > 0]
    return len(positive) >= 2 and max(positive) / min(positive) >= 10


def style_axis(ax, xlabel, ylabel, title):
    ax.set_facecolor(CHART_SURFACE)
    ax.set_title(
        title, fontsize=11, fontweight="bold", color=INK_PRIMARY, pad=10
    )
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SECONDARY)
    ax.tick_params(colors=INK_MUTED, labelsize=8.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script for plotting certain metrics from result files"
    )

    parser.add_argument(
        "-d",
        "--data-dir",
        default="data/results",
        help="Directory for result data",
    )

    parser.add_argument(
        "model",
        help="The problem model (result data starting with `[model]_` will be used for plotting)",
    )

    parser.add_argument(
        "-q",
        "--qp-solver",
        choices=["admm", "hpipm"],
        default="admm",
        help="The QP backend whose result files should be plotted",
    )

    return parser.parse_args()


def collect_problem_solved_statistics(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    # failed problems never count as solved, but still count towards the
    # total, so they pull the percentage down rather than being ignored
    solved_problems = [p for p in problems if not p.get("failed")]

    max_sqp_iter = max(
        len(problem["convergence"]["iteration_data"])
        for problem in solved_problems
    )
    iteration_ps_perc = np.zeros(max_sqp_iter)

    for problem in solved_problems:
        # subtract 1 so iteration index starts at 0
        idx = problem["convergence"]["iterations"]
        if idx < max_sqp_iter:
            iteration_ps_perc[idx] += 1

    # cumulative percentage
    cumulative = np.cumsum(iteration_ps_perc)
    percentage = cumulative / len(problems) * 100.0

    return percentage


def collect_cost_statistics(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    problems = [p for p in problems if not p.get("failed")]

    max_sqp_iter = max(
        len(problem["convergence"]["iteration_data"]) for problem in problems
    )

    cost_matrix = []

    for problem in problems:
        iteration_data = problem["convergence"]["iteration_data"]

        costs = []

        for iter in iteration_data:
            # some later iterations may not have per-stage data recorded;
            # treat those as unchanged from the last known value
            stage_costs = iter.get("stage_costs")
            if stage_costs is not None:
                costs.append(max(stage_costs))
            elif costs:
                costs.append(costs[-1])

        while len(costs) < max_sqp_iter:
            costs.append(costs[-1])

        cost_matrix.append(costs)

    cost_matrix = np.array(cost_matrix)

    median = np.nanpercentile(cost_matrix, 50, axis=0)
    p25 = np.nanpercentile(cost_matrix, 25, axis=0)
    p75 = np.nanpercentile(cost_matrix, 70, axis=0)

    return median, p25, p75


def collect_violation_statistics(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    problems = [p for p in problems if not p.get("failed")]

    max_sqp_iter = max(
        len(problem["convergence"]["iteration_data"]) for problem in problems
    )

    violations_matrix = []

    for problem in problems:
        iteration_data = problem["convergence"]["iteration_data"]

        violations = []

        for iter in iteration_data:
            # some later iterations may not have per-stage data recorded;
            # treat those as unchanged from the last known value
            stage_gap_violations = iter.get("stage_gap_violations")
            stage_constraint_violations = iter.get(
                "stage_constraint_violations"
            )
            if (
                stage_gap_violations is not None
                and stage_constraint_violations is not None
            ):
                violations.append(
                    max(
                        max(stage_gap_violations),
                        max(stage_constraint_violations),
                    )
                )
            elif violations:
                violations.append(violations[-1])

        while len(violations) < max_sqp_iter:
            violations.append(violations[-1])

        violations_matrix.append(violations)

    violations_matrix = np.array(violations_matrix)

    median = np.nanpercentile(violations_matrix, 50, axis=0)
    p25 = np.nanpercentile(violations_matrix, 25, axis=0)
    p75 = np.nanpercentile(violations_matrix, 70, axis=0)

    return median, p25, p75


import json
import numpy as np


def collect_time_to_satisfaction(filepath, include_cost=False):
    """
    Computes median, 25th, and 75th percentiles of walltime
    to reach constraint satisfaction given solver tolerance.
    """
    with open(filepath) as f:
        problems = json.load(f)

    times_to_satisfy = []

    for problem in problems:
        tolerance = problem["solver"]["tolerance"]
        iteration_data = problem["convergence"]["iteration_data"]
        runtime_ms_cumulative = 0.0

        for iter_data in iteration_data:
            # max of constraint violations at this iteration
            stage_gap_norm = max(iter_data["stage_gap_violations"])
            stage_constraint_violation_norm = max(
                iter_data["stage_constraint_violations"]
            )
            total = max(stage_gap_norm, stage_constraint_violation_norm)

            if include_cost:
                stage_cost_norm = max(iter_data["stage_costs"])
                total = max(total, stage_cost_norm)

            # accumulate runtime
            runtime_ms_cumulative += iter_data["qp_runtime_us"] / 1000.0

            # check if satisfied
            if total <= tolerance:
                times_to_satisfy.append(
                    runtime_ms_cumulative / 1000.0
                )  # seconds
                break
        else:
            # if never satisfied, take total runtime
            times_to_satisfy.append(runtime_ms_cumulative / 1000.0)

    times_to_satisfy = np.array(times_to_satisfy)

    median = np.nanpercentile(times_to_satisfy, 50)
    p25 = np.nanpercentile(times_to_satisfy, 25)
    p75 = np.nanpercentile(times_to_satisfy, 75)

    return median, p25, p75


args = parse_args()
data_dir = Path(args.data_dir)

result_prefix = f"{args.model}_{args.qp_solver}_"

files = sorted([file for file in data_dir.glob(f"{result_prefix}*.json")])

if not files:
    raise SystemExit(
        f"No result files matching '{result_prefix}*.json' found in "
        f"{data_dir}. Did you run run_benchmark.py with -q {args.qp_solver}?"
    )

plt.rcParams["font.family"] = "sans-serif"

all_data = {}
for filepath in files:
    label = pretty_label(filepath.stem, result_prefix)
    all_data[label] = collect_problem_solved_statistics(filepath)

fig, axes = plt.subplots(3, 1, figsize=(5, 10), facecolor=PAGE_PLANE)
fig.suptitle(
    f"{args.model.replace('_', ' ').title()} — {args.qp_solver.upper()}",
    fontsize=13,
    fontweight="bold",
    color=INK_PRIMARY,
)

# =========================
# First Plot (Problems Solved /w SQP iterations)
# =========================
ax1 = axes[0]

for i, label in enumerate(all_data):
    y = all_data[label]
    x = np.arange(1, len(y) + 1)
    ax1.plot(
        x,
        y,
        label=label,
        color=PALETTE[i % len(PALETTE)],
        linewidth=1.75,
    )

style_axis(ax1, "SQP Iteration", "Converged Problems [%]", "Problems Solved")

# =========================
# Second Plot (Costs)
# =========================
ax2 = axes[1]

cost_values = []
for i, filepath in enumerate(files):
    label = pretty_label(filepath.stem, result_prefix)

    median, p25, p75 = collect_cost_statistics(filepath)
    cost_values.extend(median)
    cost_values.extend(p25)
    cost_values.extend(p75)

    x = np.arange(1, len(median) + 1)
    color = PALETTE[i % len(PALETTE)]

    # Median line
    ax2.plot(x, median, label=label, color=color, linewidth=1.75)

    # Interquartile band (25–75%)
    ax2.fill_between(x, p25, p75, alpha=0.15, color=color, linewidth=0)

if axis_log_scale(cost_values):
    ax2.set_yscale("log")
style_axis(ax2, "SQP Iteration", "Cost", "Cost")

# =========================
# Third Plot (Violations)
# =========================
ax3 = axes[2]

violation_values = []
for i, filepath in enumerate(files):
    label = pretty_label(filepath.stem, result_prefix)

    median, p25, p75 = collect_violation_statistics(filepath)
    violation_values.extend(median)
    violation_values.extend(p25)
    violation_values.extend(p75)

    x = np.arange(1, len(median) + 1)
    color = PALETTE[i % len(PALETTE)]

    # Median line
    ax3.plot(x, median, label=label, color=color, linewidth=1.75)

    # Interquartile band (25–75%)
    ax3.fill_between(x, p25, p75, alpha=0.15, color=color, linewidth=0)

if axis_log_scale(violation_values):
    ax3.set_yscale("log")
style_axis(
    ax3, "SQP Iteration", "Constraint Violations", "Constraint Violation"
)

plt.tight_layout()
plt.savefig(f"data/plots/plot_{args.model}_{args.qp_solver}.pdf")
plt.show()
