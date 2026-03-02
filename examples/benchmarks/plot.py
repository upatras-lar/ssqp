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

    parser.add_argument(
        "-H",
        "--horizon",
        required=True,
        help="The problem horizon (result data filenames will be searched for `_H[horizon]_`)",
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


def collect_qp_iteration_statistics(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    # Determine global max SQP iterations
    max_sqp_iter = max(problem["solver"]["max_iter"] for problem in problems)

    # Matrix: (num_problems, max_sqp_iter)
    qp_matrix = []

    for problem in problems:
        iteration_data = problem["convergence"]["iteration_data"]

        qp_values = []

        # Collect actual QP iterations
        for iter_data in iteration_data:
            qp_values.append(iter_data["qp_iterations"])

        # Forward-fill after convergence
        if len(qp_values) > 0:
            last_value = qp_values[-1]
            while len(qp_values) < max_sqp_iter:
                qp_values.append(last_value)
        else:
            qp_values = [np.nan] * max_sqp_iter

        qp_matrix.append(qp_values)

    qp_matrix = np.array(qp_matrix)

    # Now compute percentiles across ALL problems
    median = np.percentile(qp_matrix, 50, axis=0)
    p25 = np.percentile(qp_matrix, 25, axis=0)
    p75 = np.percentile(qp_matrix, 75, axis=0)

    return median, p25, p75


args = parse_args()
data_dir = Path(args.data_dir)

files = sorted([file for file in data_dir.glob(f"{args.model}_admm_*.json")])

all_data = {}
for filepath in files:
    label = filepath.stem.replace(f"{args.model}_", "").replace(f"_H{args.horizon}", "")
    all_data[label] = collect_problem_solved_statistics(filepath)

file_labels = list(all_data.keys())
colors = plt.cm.tab10.colors

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"{args.model} - Horizon {args.horizon}")

# =========================
# First Plot (Problems Solved)
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

ax1.set_xlabel("Iterations")
ax1.set_ylabel("Converged Problems [%]")
ax1.grid(True)
ax1.legend()


# =========================
# Second Plot (QP Statistics)
# =========================
ax2 = axes[1]

for i, filepath in enumerate(files):
    label = filepath.stem.replace(f"{args.model}_", "").replace(f"_H{args.horizon}", "")

    median, p25, p75 = collect_qp_iteration_statistics(filepath)

    x = np.arange(1, len(median) + 1)
    color = colors[i % len(colors)]

    # Median line
    ax2.plot(
        x,
        median,
        label=f"{label} (median)",
        color=color,
    )

    # Interquartile band (25–75%)
    ax2.fill_between(
        x,
        p25,
        p75,
        alpha=0.2,
        color=color,
    )

ax2.set_xlabel("QP Iteration")
ax2.set_ylabel("QP Solver Iterations")
ax2.grid(True)
ax2.legend()


plt.tight_layout()
plt.savefig(f"data/plots/plot_{args.model}_H{args.horizon}.pdf")
plt.show()
