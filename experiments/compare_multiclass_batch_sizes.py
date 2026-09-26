#!/usr/bin/env python3
"""Paired ten-class MNIST GeoMed/Radius sweep over minibatch sizes.

This reuses the audited experiment implementation in
``compare_multiclass_trajectories.py`` and adds resumable execution over
multiple minibatch sizes.  Every condition keeps the per-update observation
budget fixed, so changing ``b`` changes the number of minibatch gradients
rather than the number of sampled training examples.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean

import numpy as np

from compare_multiclass_trajectories import (
    METHODS,
    Point,
    Task,
    plot,
    read_points,
    summarize,
    t_interval,
    train,
    write_points,
)


def plot_accuracy_at_iteration(
    points: list[Point], batch_sizes: list[int], iteration: int, output_dir: Path
) -> list[dict[str, float | int | str]]:
    """Plot method accuracy and paired gaps over b; return plotted statistics."""
    import matplotlib.pyplot as plt

    colors = {"geomed": "#2563eb", "radius": "#dc2626"}
    labels = {"geomed": "GeoMed", "radius": "Radius"}
    records: list[dict[str, float | int | str]] = []

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for method in METHODS:
        centers, lows, highs = [], [], []
        for batch_size in batch_sizes:
            values = [
                p.test_accuracy
                for p in points
                if p.batch_size == batch_size
                and p.method == method
                and p.iteration == iteration
            ]
            center, low, high = t_interval(values)
            centers.append(center)
            lows.append(low)
            highs.append(high)
            records.append(
                {
                    "iteration": iteration,
                    "batch_size": batch_size,
                    "method": method,
                    "seeds": len(values),
                    "mean_test_accuracy": center,
                    "accuracy_ci95_low": low,
                    "accuracy_ci95_high": high,
                }
            )
        axis.plot(
            batch_sizes,
            centers,
            color=colors[method],
            linewidth=2.2,
            marker="o",
            markersize=5,
            label=labels[method],
        )
        axis.fill_between(batch_sizes, lows, highs, color=colors[method], alpha=0.15)

    axis.set_xlabel("Minibatch size b")
    axis.set_ylabel("Clean test accuracy")
    axis.set_title(
        f"Ten-class MNIST at iteration {iteration:,} "
        "(mean and 95% t-CI, n=5)"
    )
    axis.set_xticks(batch_sizes)
    axis.set_ylim(0.85, 0.90)
    axis.grid(alpha=0.22)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"accuracy_at_{iteration}_by_batch_size.png", dpi=300)
    plt.close(fig)

    keyed = {
        (p.batch_size, p.seed, p.method): p
        for p in points
        if p.iteration == iteration
    }
    gap_centers, gap_lows, gap_highs = [], [], []
    for batch_size in batch_sizes:
        seeds = sorted(
            {
                p.seed
                for p in points
                if p.batch_size == batch_size and p.iteration == iteration
            }
        )
        gaps = [
            100.0
            * (
                keyed[batch_size, seed, "geomed"].test_accuracy
                - keyed[batch_size, seed, "radius"].test_accuracy
            )
            for seed in seeds
            if (batch_size, seed, "geomed") in keyed
            and (batch_size, seed, "radius") in keyed
        ]
        center, low, high = t_interval(gaps)
        gap_centers.append(center)
        gap_lows.append(low)
        gap_highs.append(high)
        records.append(
            {
                "iteration": iteration,
                "batch_size": batch_size,
                "method": "geomed_minus_radius_paired_gap_points",
                "seeds": len(gaps),
                "mean_test_accuracy": center,
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
            }
        )

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.plot(
        batch_sizes,
        gap_centers,
        color="#6d28d9",
        linewidth=2.2,
        marker="o",
        markersize=5,
    )
    axis.fill_between(
        batch_sizes, gap_lows, gap_highs, color="#6d28d9", alpha=0.16
    )
    axis.axhline(0.0, color="black", linewidth=1.0)
    axis.set_xlabel("Minibatch size b")
    axis.set_ylabel("GeoMed - Radius accuracy (percentage points)")
    axis.set_title(f"Paired accuracy gap at iteration {iteration:,} (95% t-CI, n=5)")
    axis.set_xticks(batch_sizes)
    axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / f"paired_gap_at_{iteration}_by_batch_size.png", dpi=300)
    plt.close(fig)
    return records


def plot_all_accuracy_trajectories(
    points: list[Point], batch_sizes: list[int], budget: int, output_dir: Path
) -> None:
    """Make a shared-scale small-multiple overview of every b trajectory."""
    import matplotlib.pyplot as plt

    colors = {"geomed": "#2563eb", "radius": "#dc2626"}
    labels = {"geomed": "GeoMed", "radius": "Radius"}
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), sharex=True, sharey=True)
    for axis, batch_size in zip(axes.flat, batch_sizes):
        condition = [p for p in points if p.batch_size == batch_size]
        iterations = sorted({p.iteration for p in condition})
        for method in METHODS:
            centers, lows, highs = [], [], []
            for iteration in iterations:
                values = [
                    p.test_accuracy
                    for p in condition
                    if p.method == method and p.iteration == iteration
                ]
                center, low, high = t_interval(values)
                centers.append(center)
                lows.append(low)
                highs.append(high)
            axis.plot(
                iterations,
                centers,
                color=colors[method],
                linewidth=2.0,
                marker="o",
                markersize=2.7,
                label=labels[method],
            )
            axis.fill_between(
                iterations, lows, highs, color=colors[method], alpha=0.14
            )
        axis.axhline(0.1, color="#111827", linestyle=":", linewidth=1.0)
        axis.set_title(f"b={batch_size} (m={budget // batch_size})")
        axis.set_ylim(0.08, 0.92)
        axis.grid(alpha=0.22)
    for axis in axes[-1, :]:
        axis.set_xlabel("SGD updates")
    for axis in axes[:, 0]:
        axis.set_ylabel("Clean test accuracy")
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=2,
        frameon=True,
    )
    fig.suptitle(
        "Ten-class MNIST trajectories by minibatch size "
        "(mean and 95% t-CI, five paired seeds)",
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    fig.savefig(output_dir / "accuracy_trajectories_by_batch_size.png", dpi=250)
    plt.close(fig)


def write_batch_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist-path", default=str(Path.home() / ".keras" / "datasets" / "mnist.npz")
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[8, 16, 32, 64, 80, 128]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--summary-iteration", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000,
                 3000, 4000, 5000],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/multiclass_batch_sweep_eps0p1_lr0p01"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mnist_path = Path(args.mnist_path)
    if not mnist_path.exists():
        raise FileNotFoundError(mnist_path)
    batch_sizes = sorted(set(args.batch_sizes))
    invalid = [b for b in batch_sizes if args.budget % b]
    if invalid:
        raise ValueError(f"batch sizes must divide budget {args.budget}: {invalid}")
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if checkpoints[-1] > args.steps:
        raise ValueError("the largest checkpoint exceeds --steps")
    if args.summary_iteration not in checkpoints:
        raise ValueError("summary iteration must be one of the checkpoints")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_trajectories.csv"
    points = read_points(raw_path)
    expected_iterations = set(checkpoints)
    completed: set[tuple[int, int, str]] = set()
    for batch_size in batch_sizes:
        for seed in args.seeds:
            for method in METHODS:
                observed = {
                    p.iteration
                    for p in points
                    if p.epsilon == args.epsilon
                    and p.batch_size == batch_size
                    and p.seed == seed
                    and p.method == method
                    and p.learning_rate == args.learning_rate
                    and p.budget == args.budget
                }
                if expected_iterations <= observed:
                    completed.add((batch_size, seed, method))

    tasks = [
        Task(
            epsilon=args.epsilon,
            batch_size=batch_size,
            seed=seed,
            method=method,
            steps=args.steps,
            learning_rate=args.learning_rate,
            budget=args.budget,
            checkpoints=checkpoints,
            mnist_path=str(mnist_path),
        )
        for batch_size in batch_sizes
        for seed in args.seeds
        for method in METHODS
        if (batch_size, seed, method) not in completed
    ]

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(train, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                task_points = future.result()
                points = [
                    p
                    for p in points
                    if not (
                        p.epsilon == task.epsilon
                        and p.batch_size == task.batch_size
                        and p.seed == task.seed
                        and p.method == task.method
                        and p.learning_rate == task.learning_rate
                        and p.budget == task.budget
                    )
                ]
                points.extend(task_points)
                write_points(raw_path, points)
                endpoint = task_points[-1]
                print(
                    f"[{index}/{len(tasks)}] b={task.batch_size} seed={task.seed} "
                    f"{task.method} accuracy={endpoint.test_accuracy:.4f} "
                    f"loss={endpoint.test_cross_entropy:.4f}",
                    flush=True,
                )

    selected = [
        p
        for p in points
        if p.epsilon == args.epsilon
        and p.batch_size in batch_sizes
        and p.seed in args.seeds
        and p.method in METHODS
        and p.iteration in expected_iterations
        and p.learning_rate == args.learning_rate
        and p.budget == args.budget
    ]
    # Keep this sweep's raw table self-contained; the bootstrapped b=8 file can
    # contain later checkpoints from the separate 10,000-update experiment.
    write_points(raw_path, selected)
    summary = summarize(selected, args.output_dir / "trajectory_summary.csv")
    for batch_size in batch_sizes:
        condition_points = [p for p in selected if p.batch_size == batch_size]
        condition_summary = [
            row for row in summary if int(row["batch_size"]) == batch_size
        ]
        condition_dir = args.output_dir / f"b{batch_size:03d}"
        condition_dir.mkdir(parents=True, exist_ok=True)
        plot(condition_points, condition_summary, condition_dir)

    plot_all_accuracy_trajectories(selected, batch_sizes, args.budget, args.output_dir)

    batch_rows = plot_accuracy_at_iteration(
        selected, batch_sizes, args.summary_iteration, args.output_dir
    )
    write_batch_summary(args.output_dir / "batch_size_summary.csv", batch_rows)

    metadata = {
        "question": (
            "How do GeoMed and Radius ten-class MNIST accuracy trajectories vary "
            "with minibatch size, and how do they compare at iteration 5000?"
        ),
        "task": "ten-class MNIST softmax regression without intercept",
        "corruption": (
            "An exact epsilon fraction of training images is selected once and "
            "pixelwise reversed; labels are unchanged; the clean test set is used."
        ),
        "epsilon": args.epsilon,
        "batch_sizes": batch_sizes,
        "num_minibatches_by_batch_size": {
            str(b): args.budget // b for b in batch_sizes
        },
        "seeds": args.seeds,
        "steps": args.steps,
        "summary_iteration": args.summary_iteration,
        "checkpoints": list(checkpoints),
        "learning_rate": args.learning_rate,
        "budget": args.budget,
        "methods": list(METHODS),
        "mnist_source": os.path.abspath(mnist_path),
        "pairing": (
            "Same corruption mask, all-zero initialization, and sampled observations "
            "for both aggregators within every (batch size, seed) pair."
        ),
        "initialization": "All-zero 784-by-10 weight matrix.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
