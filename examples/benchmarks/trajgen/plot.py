import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import argparse


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
        "-m",
        "--model",
        required=True,
        help="The problem model (result data starting with `[model]_` will be used for plotting)",
    )

    return parser.parse_args()


def collect_problem_solved_statistics(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    max_sqp_iter = max(
        len(problem["convergence"]["iteration_data"]) for problem in problems
    )
    iteration_ps_perc = np.zeros(max_sqp_iter)

    for problem in problems:
        # subtract 1 so iteration index starts at 0
        idx = problem["convergence"]["iterations"]
        if idx < max_sqp_iter:
            iteration_ps_perc[idx] += 1

    # cumulative percentage
    cumulative = np.cumsum(iteration_ps_perc)
    percentage = cumulative / len(problems) * 100.0

    return percentage


def collect_problem_solved_walltime_statistics(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    max_sqp_iter = max(
        len(problem["convergence"]["iteration_data"]) for problem in problems
    )

    sorted_walltime_problems = sorted(
        problems, key=lambda problem: problem["convergence"]["runtime_ms"]
    )

    walltimes = [
        problem["convergence"]["runtime_ms"] / 1000
        for problem in sorted_walltime_problems
    ]
    walltimes.insert(0, 0.0)

    problems_solved = 0
    percentages = [0.0]

    for problem in problems:
        if problem["convergence"]["iterations"] < max_sqp_iter:
            problems_solved += 1
            percentages.append(problems_solved / len(problems) * 100)
            continue

        percentages.append(percentages[-1])

    return percentages, walltimes


def collect_violation_statistics(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    max_sqp_iter = max(
        len(problem["convergence"]["iteration_data"]) for problem in problems
    )

    violations_matrix = []

    for problem in problems:
        iteration_data = problem["convergence"]["iteration_data"]

        violations = []

        for iter in iteration_data:
            stage_gap_norm = max(iter["stage_gap_violations"])
            stage_constraint_violation_norm = max(
                iter["stage_constraint_violations"]
            )
            violations.append(
                max(stage_gap_norm, stage_constraint_violation_norm)
            )

        while len(violations) < max_sqp_iter:
            violations.append(violations[-1])

        violations_matrix.append(violations)

    violations_matrix = np.array(violations_matrix)

    median = np.percentile(violations_matrix, 50, axis=0)
    p25 = np.percentile(violations_matrix, 25, axis=0)
    p75 = np.percentile(violations_matrix, 70, axis=0)

    return median, p25, p75


args = parse_args()
data_dir = Path(args.data_dir)

files = sorted([file for file in data_dir.glob(f"{args.model}_admm_*.json")])

all_data = {}
for filepath in files:
    label = filepath.stem.replace(f"{args.model}_admm_", "").capitalize()
    all_data[label] = collect_problem_solved_statistics(filepath)

file_labels = list(all_data.keys())
colors = plt.cm.tab10.colors

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f"{args.model}")

# =========================
# First Plot (Problems Solved /w SQP iterations)
# =========================
ax1 = axes[0]

for i, label in enumerate(file_labels):
    y = all_data[label]
    x = np.arange(1, len(y) + 1)
    ax1.plot(
        x,
        y,
        label=label,
        color=colors[i % len(colors)],
    )

ax1.set_xlabel("SQP Iteration")
ax1.set_ylabel("Converged Problems [%]")
ax1.grid(True)
ax1.legend()

# =========================
# Second Plot (Problems Solved /w Wall time)
# =========================
ax2 = axes[1]

for i, filepath in enumerate(files):
    label = filepath.stem.replace(f"{args.model}_admm_", "").capitalize()

    percentages, walltimes = collect_problem_solved_walltime_statistics(
        filepath
    )
    color = colors[i % len(colors)]

    ax2.plot(
        walltimes,
        percentages,
        label=label,
        color=color,
    )

ax2.set_xlabel("Walltime (s)")
ax2.set_ylabel("Converged Problems [%]")
ax2.grid(True)
ax2.legend()

# =========================
# Third Plot (Violations)
# =========================
ax3 = axes[2]

for i, filepath in enumerate(files):
    label = filepath.stem.replace(f"{args.model}_admm_", "").capitalize()

    median, p25, p75 = collect_violation_statistics(filepath)

    x = np.arange(1, len(median) + 1)
    color = colors[i % len(colors)]

    # Median line
    ax3.plot(
        x,
        median,
        label=f"{label} (median)",
        color=color,
    )

    # Interquartile band (25–75%)
    ax3.fill_between(
        x,
        p25,
        p75,
        alpha=0.2,
        color=color,
    )

ax3.set_xlabel("SQP Iteration")
ax3.set_ylabel("Constraint Violations")
ax3.set_yscale("log")
ax3.grid(True)
ax3.legend()

plt.tight_layout()
plt.savefig(f"data/plots/plot_{args.model}.pdf")
plt.show()
