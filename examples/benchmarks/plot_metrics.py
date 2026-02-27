import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def compute_stats(values):
    """Compute median, 25th percentile, and 75th percentile."""
    arr = np.array(values)
    return {
        "median": np.median(arr),
        "p25": np.percentile(arr, 25),
        "p75": np.percentile(arr, 75),
    }


def process_file(filepath):
    with open(filepath) as f:
        problems = json.load(f)

    # Collect metrics from all problems
    sqp_iterations = [p["convergence"]["iterations"] for p in problems]
    runtime_ms = [p["convergence"]["runtime_ms"] for p in problems]

    # Median QP iterations per problem (median across all SQP iterations within each problem)
    median_qp_iters = [
        np.median(
            [it["qp_iterations"] for it in p["convergence"]["iteration_data"]]
        )
        for p in problems
    ]
    # Median QP time (us) per problem
    median_qp_time_us = [
        np.median(
            [it["qp_runtime_us"] for it in p["convergence"]["iteration_data"]]
        )
        for p in problems
    ]

    # Final solution quality
    final_cost = [p["convergence"]["trajectory_cost"] for p in problems]
    final_constraint_viol = [
        p["convergence"]["trajectory_constraint_violation"] for p in problems
    ]
    final_stage_gap = [
        p["convergence"]["trajectory_stage_gap"] for p in problems
    ]

    return {
        "sqp_iterations": compute_stats(sqp_iterations),
        "runtime_ms": compute_stats(runtime_ms),
        "qp_iterations": compute_stats(median_qp_iters),
        "qp_time_us": compute_stats(median_qp_time_us),
        "final_cost": compute_stats(final_cost),
        "constraint_viol": compute_stats(final_constraint_viol),
        "stage_gap": compute_stats(final_stage_gap),
    }


def main():
    data_dir = Path("data/results")
    metrics_file = data_dir / "metrics_cache.json"

    # === COMPUTE AND SAVE METRICS (uncomment to regenerate) ===
    files = sorted(data_dir.glob("panda_*.json"))
    all_data = {}
    for filepath in files:
        label = filepath.stem.replace("panda_", "").replace("_H25", "")
        all_data[label] = process_file(filepath)
    with open(metrics_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    # === END COMPUTE ===

    # Load metrics from cache
    with open(metrics_file) as f:
        all_data = json.load(f)

    file_labels = list(all_data.keys())
    colors = plt.cm.tab10.colors

    # Define metric groups with their y-axis properties
    metric_groups = [
        {
            "metrics": ["sqp_iterations", "qp_iterations"],
            "labels": ["SQP Iters", "Avg QP Iters"],
            "ylabel": "Iterations",
            "log": True,
        },
        {
            "metrics": ["runtime_ms", "qp_time_us"],
            "labels": ["Runtime (ms)", "Avg QP Time (μs)"],
            "ylabel": "Time",
            "log": True,
        },
        {
            "metrics": ["final_cost", "constraint_viol", "stage_gap"],
            "labels": ["Cost", "Constr Viol", "Stage Gap"],
            "ylabel": "Cost / Violations",
            "log": True,
        },
    ]

    # Build flat list of all metrics and their group indices
    all_metrics = []
    all_labels = []
    metric_to_group = []
    for gi, group in enumerate(metric_groups):
        for m, l in zip(group["metrics"], group["labels"]):
            all_metrics.append(m)
            all_labels.append(l)
            metric_to_group.append(gi)

    n_metrics = len(all_metrics)
    n_files = len(file_labels)

    # ============ TUNABLE PARAMETERS ============
    metric_width = 0.2  # Spacing between metrics within a group
    group_gap = 0.3  # Distance between groups
    bar_width_frac = 0.025  # Bar width as fraction of total x-axis range
    yaxis_offset = (
        -0.1
    )  # Y-axis position relative to group start (negative = left of group)
    # ============================================

    # Build x positions with gaps between groups
    # group_gap is the total distance between last metric of one group and first of next
    # (when group_gap == metric_width, all metrics are evenly spaced)
    x = []
    group_starts = []  # Track where each group starts
    pos = 0
    for gi, group in enumerate(metric_groups):
        group_starts.append(pos)
        for mi, _ in enumerate(group["metrics"]):
            x.append(pos)
            # Use metric_width within group, group_gap to next group
            if mi < len(group["metrics"]) - 1:
                pos += metric_width
            elif gi < len(metric_groups) - 1:
                pos += group_gap
    x = np.array(x)
    group_starts = np.array(group_starts)

    fig, ax1 = plt.subplots(figsize=(14, 3.5))

    # Calculate axis boundary positions
    x_min = x[0] - 0.5 * metric_width
    x_max = x[-1] + 0.5 * metric_width
    x_range = x_max - x_min

    # Bar width relative to total x range
    bar_width = bar_width_frac * x_range

    # Position y-axes at yaxis_offset from start of each group
    yaxis_positions = [
        group_starts[i] + yaxis_offset for i in range(len(metric_groups))
    ]

    # Create additional y-axes
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()

    # Convert to axes fraction and position spines
    ax1_pos_frac = (yaxis_positions[0] - x_min) / x_range
    ax2_pos_frac = (yaxis_positions[1] - x_min) / x_range
    ax3_pos_frac = (yaxis_positions[2] - x_min) / x_range

    # Position first y-axis (ax1)
    ax1.spines["left"].set_position(("axes", ax1_pos_frac))

    ax2.spines["left"].set_position(("axes", ax2_pos_frac))
    ax2.spines["left"].set_visible(True)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.set_ticks_position("left")

    ax3.spines["left"].set_position(("axes", ax3_pos_frac))
    ax3.spines["left"].set_visible(True)
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.yaxis.set_label_position("left")
    ax3.yaxis.set_ticks_position("left")

    # Remove top and right spines from main axis
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    axes_list = [ax1, ax2, ax3]

    # Plot bars for each file
    for i, file_label in enumerate(file_labels):
        for gi, group in enumerate(metric_groups):
            ax = axes_list[gi]
            group_metrics = group["metrics"]

            medians = []
            err_low = []
            err_high = []
            positions = []

            for mi, metric in enumerate(all_metrics):
                if metric in group_metrics:
                    stats = all_data[file_label][metric]
                    medians.append(stats["median"])
                    err_low.append(stats["median"] - stats["p25"])
                    err_high.append(stats["p75"] - stats["median"])
                    positions.append(x[mi])  # Use actual x position with gaps

            offset = (i - n_files / 2 + 0.5) * bar_width
            ax.bar(
                np.array(positions) + offset,
                medians,
                bar_width,
                yerr=[err_low, err_high],
                capsize=2,
                color=colors[i % len(colors)],
                label=file_label if gi == 0 else None,
            )

    # Configure axes with grey color
    grey = "grey"

    ax1.set_ylabel(metric_groups[0]["ylabel"])
    ax1.tick_params(axis="y", colors=grey)
    ax1.spines["left"].set_color(grey)
    ax1.yaxis.label.set_color(grey)
    if metric_groups[0]["log"]:
        ax1.set_yscale("log")

    ax2.set_ylabel(metric_groups[1]["ylabel"])
    ax2.tick_params(axis="y", colors=grey)
    ax2.spines["left"].set_color(grey)
    ax2.yaxis.label.set_color(grey)
    if metric_groups[1]["log"]:
        ax2.set_yscale("log")

    ax3.set_ylabel(metric_groups[2]["ylabel"])
    ax3.tick_params(axis="y", colors=grey)
    ax3.spines["left"].set_color(grey)
    ax3.yaxis.label.set_color(grey)
    if metric_groups[2]["log"]:
        ax3.set_yscale("log")

    # Set x-axis
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=45, ha="right")
    ax1.set_xlim(x_min, x_max)

    ax1.legend(loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig("data/plots/metrics_bars.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
