#!/usr/bin/env python3
"""Build the workshop paper's trajectory-and-batch-size figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from compare_multiclass_trajectories import METHODS, read_points, t_interval


ROOT = Path(__file__).resolve().parents[1]
TRAJECTORY_CSV = (
    ROOT
    / "output/experiments/multiclass_eps0p1_lr0p01_zero_init/raw_trajectories.csv"
)
SWEEP_CSV = (
    ROOT
    / "output/experiments/multiclass_batch_sweep_eps0p1_lr0p01/raw_trajectories.csv"
)
OUTPUT = (
    ROOT
    / "output/figures/"
    "mnist_multiclass_trajectory_and_batch_sweep_test_accuracy.png"
)

COLORS = {"geomed": "#0072B2", "radius": "#D55E00"}
LABELS = {"geomed": "GeoMed", "radius": "Radius"}


def interval(points, method: str, iteration: int):
    values = [
        100.0 * point.test_accuracy
        for point in points
        if point.method == method and point.iteration == iteration
    ]
    return t_interval(values)


def main() -> None:
    trajectory = [
        point for point in read_points(TRAJECTORY_CSV) if point.batch_size == 8
    ]
    sweep = read_points(SWEEP_CSV)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.2))
    ax_accuracy, ax_gap, ax_batch = axes

    iterations = sorted({point.iteration for point in trajectory})
    for method in METHODS:
        summaries = [interval(trajectory, method, iteration) for iteration in iterations]
        centers = [summary[0] for summary in summaries]
        lows = [summary[1] for summary in summaries]
        highs = [summary[2] for summary in summaries]
        ax_accuracy.plot(
            iterations,
            centers,
            color=COLORS[method],
            linewidth=2.3,
            marker="o",
            markersize=3.2,
            label=LABELS[method],
        )
        ax_accuracy.fill_between(
            iterations, lows, highs, color=COLORS[method], alpha=0.16
        )

    ax_accuracy.set_title("(a) Accuracy trajectory, $b=8$")
    ax_accuracy.set_xlabel("SGD updates")
    ax_accuracy.set_ylabel("Test accuracy (%)")
    ax_accuracy.set_xlim(0, 10_000)
    ax_accuracy.set_ylim(8, 92)
    ax_accuracy.set_xticks([0, 2_500, 5_000, 7_500, 10_000])
    ax_accuracy.set_xticklabels(["0", "2.5k", "5k", "7.5k", "10k"])
    ax_accuracy.set_yticks([10, 30, 50, 70, 90])
    ax_accuracy.legend(loc="lower right", frameon=True)

    keyed = {
        (point.seed, point.method, point.iteration): point for point in trajectory
    }
    gap_iterations = [iteration for iteration in iterations if iteration >= 500]
    gap_centers, gap_lows, gap_highs = [], [], []
    for iteration in gap_iterations:
        seeds = sorted(
            {
                point.seed
                for point in trajectory
                if point.iteration == iteration
            }
        )
        gaps = [
            100.0
            * (
                keyed[seed, "geomed", iteration].test_accuracy
                - keyed[seed, "radius", iteration].test_accuracy
            )
            for seed in seeds
        ]
        center, low, high = t_interval(gaps)
        gap_centers.append(center)
        gap_lows.append(low)
        gap_highs.append(high)
    ax_gap.plot(
        gap_iterations,
        gap_centers,
        color="#6A3D9A",
        linewidth=2.3,
        marker="o",
        markersize=3.2,
    )
    ax_gap.fill_between(
        gap_iterations, gap_lows, gap_highs, color="#6A3D9A", alpha=0.17
    )
    ax_gap.axhline(0.0, color="#333333", linewidth=1.0)
    ax_gap.set_title("(b) Paired accuracy gap, $b=8$")
    ax_gap.set_xlabel("SGD updates")
    ax_gap.set_ylabel("GeoMed $-$ Radius (points)")
    ax_gap.set_xlim(500, 10_000)
    ax_gap.set_ylim(0, 6.5)
    ax_gap.set_xticks([500, 2_500, 5_000, 7_500, 10_000])
    ax_gap.set_xticklabels([".5k", "2.5k", "5k", "7.5k", "10k"])
    ax_gap.set_yticks([0, 2, 4, 6])

    batch_sizes = sorted({point.batch_size for point in sweep})
    for method in METHODS:
        summaries = [
            interval(
                [point for point in sweep if point.batch_size == batch_size],
                method,
                5_000,
            )
            for batch_size in batch_sizes
        ]
        centers = [summary[0] for summary in summaries]
        lower_errors = [
            summary[0] - summary[1] for summary in summaries
        ]
        upper_errors = [
            summary[2] - summary[0] for summary in summaries
        ]
        ax_batch.errorbar(
            batch_sizes,
            centers,
            yerr=[lower_errors, upper_errors],
            color=COLORS[method],
            linewidth=2.3,
            marker="o",
            markersize=4.2,
            capsize=3.0,
            label=LABELS[method],
        )
    ax_batch.set_title("(c) Accuracy at update 5,000")
    ax_batch.set_xlabel("Minibatch size $b$")
    ax_batch.set_ylabel("Test accuracy (%)")
    ax_batch.set_xticks(batch_sizes)
    ax_batch.set_ylim(85, 90)
    ax_batch.set_yticks([85, 86, 87, 88, 89, 90])

    for axis in axes:
        axis.grid(alpha=0.24, linewidth=0.8)
        axis.tick_params(labelsize=12.5)
        axis.xaxis.label.set_size(14)
        axis.yaxis.label.set_size(14)
        axis.title.set_size(15)
    ax_accuracy.legend(loc="lower right", frameon=True, fontsize=12.5)

    fig.tight_layout(w_pad=1.5)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
