import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_convergence_point(problem, tolerance):
    """
    Find the first iteration where all breaking criteria are below tolerance.
    Returns (iteration_number, cumulative_time_us, converged).
    If not converged, returns max iterations and total time.
    """
    iter_data = problem["convergence"]["iteration_data"]
    cumulative_time_us = 0

    for i, it in enumerate(iter_data):
        cumulative_time_us += it["iter_duration_us"]

        if (
            it["norm_dx"] < tolerance
            and it["norm_du"] < tolerance
            and it["trajectory_stage_gap"] < tolerance
            and it["trajectory_constraint_violation"] < tolerance
        ):
            return i + 1, cumulative_time_us, True  # 1-indexed iteration count

    # Did not converge - return max iters and total time
    return len(iter_data), cumulative_time_us, False


def compute_stats(values):
    """Compute median, 25th percentile, and 75th percentile."""
    if len(values) == 0:
        return None
    arr = np.array(values)
    return {
        "median": np.median(arr),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
    }


def collect_data(filepath, tolerances):
    """Collect convergence data for all tolerances."""
    with open(filepath) as f:
        problems = json.load(f)

    results = {}
    for tol in tolerances:
        iters_list = []
        times_list = []
        converged_count = 0

        for problem in problems:
            iters, time_us, converged = find_convergence_point(problem, tol)
            iters_list.append(iters)
            times_list.append(time_us / 1000.0)  # Convert to ms
            if converged:
                converged_count += 1

        results[tol] = {
            "iters": compute_stats(iters_list),
            "time": compute_stats(times_list),
            "converged": converged_count,
            "total": len(problems),
            "converged_pct": 100.0 * converged_count / len(problems),
        }

    return results


def main():
    data_dir = Path("data/results")
    files = sorted(
        [
            file
            for file in data_dir.glob("pendubot_*.json")
            if "200" in file.stem
        ]
    )

    # Tolerance values to test (logarithmically spaced from 1e-1 to 1e-6)
    tolerances = np.logspace(-1, -6, 100)

    # Collect data for all files
    all_data = {}
    for filepath in files:
        # Create shorter label from filename
        label = filepath.stem.replace("pendubot_", "").replace("_H200", "")
        all_data[label] = collect_data(filepath, tolerances)

    # Create figure with 2 rows, 1 column
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_iter = axes[0]
    ax_time = axes[1]

    file_labels = list(all_data.keys())
    colors = plt.cm.tab10.colors  # Use tab10 colormap for distinct colors

    x = np.arange(len(tolerances))  # x positions for tolerances

    for i, label in enumerate(file_labels):
        color = colors[i % len(colors)]

        # Extract data for this file across all tolerances
        medians_iter = []
        p25_iter = []
        p75_iter = []
        medians_time = []
        p25_time = []
        p75_time = []

        for tol in tolerances:
            data = all_data[label][tol]
            if data["iters"] is not None:
                medians_iter.append(data["iters"]["median"])
                p25_iter.append(data["iters"]["p25"])
                p75_iter.append(data["iters"]["p75"])
            else:
                medians_iter.append(0)
                p25_iter.append(0)
                p75_iter.append(0)

            if data["time"] is not None:
                medians_time.append(data["time"]["median"])
                p25_time.append(data["time"]["p25"])
                p75_time.append(data["time"]["p75"])
            else:
                medians_time.append(0)
                p25_time.append(0)
                p75_time.append(0)

        # Plot iterations: line for median, shaded region for P25-P75
        ax_iter.plot(x, medians_iter, color=color, linewidth=2, label=label)
        ax_iter.fill_between(x, p25_iter, p75_iter, color=color, alpha=0.3)

        # Plot time: line for median, shaded region for P25-P75
        ax_time.plot(x, medians_time, color=color, linewidth=2, label=label)
        ax_time.fill_between(x, p25_time, p75_time, color=color, alpha=0.3)

    # Configure iterations plot
    ax_iter.set_ylabel("SQP Iterations")
    ax_iter.set_yscale("log")
    ax_iter.legend(loc="upper left")
    ax_iter.grid(True, alpha=0.3)

    # Configure time plot
    ax_time.set_ylabel("Time (ms)")
    ax_time.set_yscale("log")
    ax_time.set_xlabel("Tolerance")
    # Only show major tick labels at powers of 10
    major_tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    major_indices = [np.argmin(np.abs(tolerances - t)) for t in major_tols]
    ax_time.set_xticks(major_indices)
    ax_time.set_xticklabels([f"{t:.0e}" for t in major_tols])
    ax_time.legend(loc="upper left")
    ax_time.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("data/plots/convergence_lines.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
