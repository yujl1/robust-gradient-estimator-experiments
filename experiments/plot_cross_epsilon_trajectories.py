#!/usr/bin/env python3
"""Combine matched trajectory runs into a cross-corruption comparison."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t


def t_interval(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    if len(values) < 2 or stdev(values) == 0.0:
        return center, center, center
    half_width = (
        float(student_t.ppf(0.975, len(values) - 1))
        * stdev(values)
        / math.sqrt(len(values))
    )
    return center, center - half_width, center + half_width


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    keyed: dict[tuple[float, int, int, str, int], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                key = (
                    float(row["epsilon"]),
                    int(row["batch_size"]),
                    int(row["seed"]),
                    row["method"],
                    int(row["iteration"]),
                )
                keyed[key] = row
    return list(keyed.values())


def summarize(rows: list[dict[str, str]]) -> list[dict[str, float | int]]:
    by_condition: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_condition[(float(row["epsilon"]), int(row["iteration"]))].append(row)

    summary: list[dict[str, float | int]] = []
    for (epsilon, iteration), group in sorted(by_condition.items()):
        by_seed: dict[int, dict[str, dict[str, str]]] = defaultdict(dict)
        for row in group:
            by_seed[int(row["seed"])][row["method"]] = row
        pairs = [pair for pair in by_seed.values() if {"geomed", "radius"} <= pair.keys()]
        if not pairs:
            continue
        gm_accuracy = [float(pair["geomed"]["test_accuracy"]) for pair in pairs]
        radius_accuracy = [float(pair["radius"]["test_accuracy"]) for pair in pairs]
        gaps = [gm - radius for gm, radius in zip(gm_accuracy, radius_accuracy)]
        gap_mean, gap_low, gap_high = t_interval(gaps)
        gm_balanced = [float(pair["geomed"]["balanced_accuracy"]) for pair in pairs]
        radius_balanced = [float(pair["radius"]["balanced_accuracy"]) for pair in pairs]
        balanced_gaps = [
            gm - radius for gm, radius in zip(gm_balanced, radius_balanced)
        ]
        balanced_gap_mean, balanced_gap_low, balanced_gap_high = t_interval(
            balanced_gaps
        )
        summary.append(
            {
                "epsilon": epsilon,
                "iteration": iteration,
                "paired_seeds": len(pairs),
                "geomed_mean_accuracy": mean(gm_accuracy),
                "radius_mean_accuracy": mean(radius_accuracy),
                "mean_gap": gap_mean,
                "gap_ci95_low": gap_low,
                "gap_ci95_high": gap_high,
                "geomed_mean_balanced_accuracy": mean(gm_balanced),
                "radius_mean_balanced_accuracy": mean(radius_balanced),
                "mean_balanced_accuracy_gap": balanced_gap_mean,
                "balanced_accuracy_gap_ci95_low": balanced_gap_low,
                "balanced_accuracy_gap_ci95_high": balanced_gap_high,
                "geomed_mean_test_loss": mean(
                    float(pair["geomed"]["test_logistic_loss"]) for pair in pairs
                ),
                "radius_mean_test_loss": mean(
                    float(pair["radius"]["test_logistic_loss"]) for pair in pairs
                ),
            }
        )
    return summary


def write_summary(path: Path, summary: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def plot(summary: list[dict[str, float | int]], output_dir: Path) -> None:
    colors = {0.01: "#059669", 0.1: "#2563eb", 0.2: "#dc2626"}
    epsilons = sorted({float(row["epsilon"]) for row in summary})
    fig, axes = plt.subplots(len(epsilons), 2, figsize=(11.2, 3.35 * len(epsilons)))
    if len(epsilons) == 1:
        axes = np.array([axes])

    for row_index, epsilon in enumerate(epsilons):
        condition = sorted(
            (row for row in summary if float(row["epsilon"]) == epsilon),
            key=lambda row: int(row["iteration"]),
        )
        iterations = np.array([int(row["iteration"]) for row in condition])
        gm = np.array([float(row["geomed_mean_accuracy"]) for row in condition])
        radius = np.array([float(row["radius_mean_accuracy"]) for row in condition])
        gap = 100.0 * np.array([float(row["mean_gap"]) for row in condition])
        low = 100.0 * np.array([float(row["gap_ci95_low"]) for row in condition])
        high = 100.0 * np.array([float(row["gap_ci95_high"]) for row in condition])

        accuracy_axis = axes[row_index, 0]
        gap_axis = axes[row_index, 1]
        accuracy_axis.plot(iterations, gm, color="#2563eb", linewidth=2.2,
                           marker="o", markersize=3, label="GeoMed")
        accuracy_axis.plot(iterations, radius, color="#dc2626", linewidth=2.2,
                           marker="o", markersize=3, label="Radius")
        accuracy_axis.axhline(0.8865, color="#111827", linestyle=":",
                              linewidth=1.2, label="Always-not-1 baseline")
        accuracy_axis.set_ylabel("Mean clean test accuracy")
        accuracy_axis.set_title(f"epsilon={epsilon:g}, b=8")
        accuracy_axis.legend(fontsize=8)

        gap_axis.plot(iterations, gap, color=colors[epsilon], linewidth=2.2,
                      marker="o", markersize=3)
        gap_axis.fill_between(iterations, low, high, color=colors[epsilon], alpha=0.16)
        gap_axis.axhline(0.0, color="black", linewidth=1)
        gap_axis.set_ylabel("GeoMed - Radius (percentage points)")
        gap_axis.set_title("Paired accuracy difference (95% t-CI)")

        for axis in (accuracy_axis, gap_axis):
            axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
            axis.set_xlabel("SGD updates")
            axis.grid(alpha=0.22)

    fig.tight_layout()
    fig.savefig(output_dir / "cross_epsilon_trajectories.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(len(epsilons), 2, figsize=(11.2, 3.35 * len(epsilons)))
    if len(epsilons) == 1:
        axes = np.array([axes])
    for row_index, epsilon in enumerate(epsilons):
        condition = sorted(
            (row for row in summary if float(row["epsilon"]) == epsilon),
            key=lambda row: int(row["iteration"]),
        )
        iterations = np.array([int(row["iteration"]) for row in condition])
        gm = np.array(
            [float(row["geomed_mean_balanced_accuracy"]) for row in condition]
        )
        radius = np.array(
            [float(row["radius_mean_balanced_accuracy"]) for row in condition]
        )
        gap = 100.0 * np.array(
            [float(row["mean_balanced_accuracy_gap"]) for row in condition]
        )
        low = 100.0 * np.array(
            [float(row["balanced_accuracy_gap_ci95_low"]) for row in condition]
        )
        high = 100.0 * np.array(
            [float(row["balanced_accuracy_gap_ci95_high"]) for row in condition]
        )

        metric_axis = axes[row_index, 0]
        gap_axis = axes[row_index, 1]
        metric_axis.plot(iterations, gm, color="#2563eb", linewidth=2.2,
                         marker="o", markersize=3, label="GeoMed")
        metric_axis.plot(iterations, radius, color="#dc2626", linewidth=2.2,
                         marker="o", markersize=3, label="Radius")
        metric_axis.axhline(0.5, color="#111827", linestyle=":", linewidth=1.2,
                            label="Chance")
        metric_axis.set_ylabel("Mean balanced accuracy")
        metric_axis.set_title(f"epsilon={epsilon:g}, b=8")
        metric_axis.legend(fontsize=8)

        gap_axis.plot(iterations, gap, color=colors[epsilon], linewidth=2.2,
                      marker="o", markersize=3)
        gap_axis.fill_between(iterations, low, high, color=colors[epsilon], alpha=0.16)
        gap_axis.axhline(0.0, color="black", linewidth=1)
        gap_axis.set_ylabel("GeoMed - Radius (percentage points)")
        gap_axis.set_title("Paired balanced-accuracy difference (95% t-CI)")

        for axis in (metric_axis, gap_axis):
            axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
            axis.set_xlabel("SGD updates")
            axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "cross_epsilon_balanced_trajectories.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for epsilon in epsilons:
        condition = sorted(
            (row for row in summary if float(row["epsilon"]) == epsilon),
            key=lambda row: int(row["iteration"]),
        )
        iterations = np.array([int(row["iteration"]) for row in condition])
        gap = 100.0 * np.array([float(row["mean_gap"]) for row in condition])
        low = 100.0 * np.array([float(row["gap_ci95_low"]) for row in condition])
        high = 100.0 * np.array([float(row["gap_ci95_high"]) for row in condition])
        axis.plot(iterations, gap, color=colors[epsilon], linewidth=2.2,
                  marker="o", markersize=3, label=f"epsilon={epsilon:g}")
        axis.fill_between(iterations, low, high, color=colors[epsilon], alpha=0.10)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
    axis.set_xlabel("SGD updates")
    axis.set_ylabel("GeoMed - Radius accuracy (percentage points)")
    axis.set_title("Accuracy advantage across corruption levels")
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cross_epsilon_gap_comparison.png", dpi=200)
    plt.close(fig)


def plot_corruption_sensitivity(
    rows: list[dict[str, str]], output_dir: Path
) -> None:
    """Make cross-epsilon changes visible instead of compressing them by scale."""
    colors = {0.01: "#059669", 0.1: "#2563eb", 0.2: "#dc2626"}
    epsilons = sorted({float(row["epsilon"]) for row in rows})
    iterations = sorted(
        {int(row["iteration"]) for row in rows if int(row["iteration"]) >= 1000}
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    for row_index, method in enumerate(("geomed", "radius")):
        for epsilon in epsilons:
            accuracy = []
            balanced = []
            for iteration in iterations:
                selected = [
                    row
                    for row in rows
                    if float(row["epsilon"]) == epsilon
                    and row["method"] == method
                    and int(row["iteration"]) == iteration
                ]
                accuracy.append(mean(float(row["test_accuracy"]) for row in selected))
                balanced.append(
                    mean(float(row["balanced_accuracy"]) for row in selected)
                )
            label = f"epsilon={epsilon:g}"
            axes[row_index, 0].plot(
                iterations, accuracy, color=colors[epsilon], linewidth=2.2,
                marker="o", markersize=3, label=label
            )
            axes[row_index, 1].plot(
                iterations, balanced, color=colors[epsilon], linewidth=2.2,
                marker="o", markersize=3, label=label
            )
        axes[row_index, 0].axhline(
            0.8865, color="#111827", linestyle=":", linewidth=1.2,
            label="Always-not-1 baseline"
        )
        axes[row_index, 1].axhline(
            0.5, color="#111827", linestyle=":", linewidth=1.2, label="Chance"
        )
        axes[row_index, 0].set_title(f"{method.title()}: raw accuracy")
        axes[row_index, 1].set_title(f"{method.title()}: balanced accuracy")
        axes[row_index, 0].set_ylabel("Mean clean test accuracy")
        axes[row_index, 1].set_ylabel("Mean balanced accuracy")
        for axis in axes[row_index]:
            axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
            axis.set_xlabel("SGD updates")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "corruption_sensitivity_metrics.png", dpi=200)
    plt.close(fig)

    keyed = {
        (
            float(row["epsilon"]),
            int(row["seed"]),
            row["method"],
            int(row["iteration"]),
        ): row
        for row in rows
    }
    seeds = sorted({int(row["seed"]) for row in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    metrics = (
        ("test_accuracy", "Raw accuracy: epsilon=0.01 minus 0.2 (pp)", 100.0),
        (
            "balanced_accuracy",
            "Balanced accuracy: epsilon=0.01 minus 0.2 (pp)",
            100.0,
        ),
        ("test_logistic_loss", "Test loss: epsilon=0.01 minus 0.2", 1.0),
    )
    method_colors = {"geomed": "#2563eb", "radius": "#dc2626"}
    for axis, (metric, label, scale) in zip(axes, metrics):
        for method in ("geomed", "radius"):
            centers = []
            lows = []
            highs = []
            for iteration in iterations:
                values = [
                    float(keyed[(0.01, seed, method, iteration)][metric])
                    - float(keyed[(0.2, seed, method, iteration)][metric])
                    for seed in seeds
                ]
                center, low, high = t_interval(values)
                centers.append(scale * center)
                lows.append(scale * low)
                highs.append(scale * high)
            axis.plot(
                iterations, centers, color=method_colors[method], linewidth=2.2,
                marker="o", markersize=3, label=method.title()
            )
            axis.fill_between(
                iterations, lows, highs, color=method_colors[method], alpha=0.13
            )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
        axis.set_xlabel("SGD updates")
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "corruption_sensitivity_paired_differences.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-files", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_rows(args.raw_files)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(args.output_dir / "cross_epsilon_summary.csv", summary)
    plot(summary, args.output_dir)
    plot_corruption_sensitivity(rows, args.output_dir)


if __name__ == "__main__":
    main()
