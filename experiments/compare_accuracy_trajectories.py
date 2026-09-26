#!/usr/bin/env python3
"""Compare GeoMed and Radius training trajectories in the MNIST setup.

The experiment extends the matched reproduction in
``reproduce_small_b_accuracy.py``.  Within every condition and seed, the two
aggregators receive the same corrupted dataset, initialization, and sequence
of sampled observations.  Checkpoints are evaluated on the clean test set.

Besides ordinary test accuracy (the metric in the original notebooks), the
script records balanced accuracy and logistic loss.  Those diagnostics matter
because digit 1 versus all other MNIST digits is imbalanced and accuracy is a
piecewise-constant, relatively noisy convergence proxy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t

from reproduce_small_b_accuracy import (
    METHODS,
    geometric_median,
    load_mnist,
    make_reverse_mask,
    radius_estimator,
)


@dataclass(frozen=True)
class Task:
    epsilon: float
    batch_size: int
    seed: int
    method: str
    steps: int
    learning_rate: float
    budget: int
    checkpoints: tuple[int, ...]
    mnist_path: str


@dataclass(frozen=True)
class Point:
    epsilon: float
    batch_size: int
    num_minibatches: int
    seed: int
    method: str
    iteration: int
    learning_rate: float
    budget: int
    test_accuracy: float
    balanced_accuracy: float
    positive_recall: float
    negative_recall: float
    test_logistic_loss: float
    weight_norm: float


def evaluate(
    weights: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    task: Task,
    iteration: int,
    num_minibatches: int,
) -> Point:
    logits = test_images @ weights
    predictions = (logits >= 0.0).astype(np.float64)
    positives = test_labels == 1.0
    negatives = ~positives
    positive_recall = float(np.mean(predictions[positives] == 1.0))
    negative_recall = float(np.mean(predictions[negatives] == 0.0))
    return Point(
        epsilon=task.epsilon,
        batch_size=task.batch_size,
        num_minibatches=num_minibatches,
        seed=task.seed,
        method=task.method,
        iteration=iteration,
        learning_rate=task.learning_rate,
        budget=task.budget,
        test_accuracy=float(np.mean(predictions == test_labels)),
        balanced_accuracy=0.5 * (positive_recall + negative_recall),
        positive_recall=positive_recall,
        negative_recall=negative_recall,
        test_logistic_loss=float(
            np.mean(np.logaddexp(0.0, logits) - test_labels * logits)
        ),
        weight_norm=float(np.linalg.norm(weights)),
    )


def train(task: Task) -> list[Point]:
    train_images, train_labels, test_images, test_labels = load_mnist(task.mnist_path)
    n = len(train_labels)
    if task.budget % task.batch_size:
        raise ValueError("budget must be divisible by every minibatch size")
    if not task.checkpoints or task.checkpoints[0] < 0:
        raise ValueError("checkpoints must be nonnegative")
    if task.checkpoints[-1] > task.steps:
        raise ValueError("a checkpoint exceeds the requested number of steps")

    num_minibatches = task.budget // task.batch_size
    reverse_mask = make_reverse_mask(n, task.epsilon, task.seed)
    rng = np.random.RandomState(20_000 + task.seed)
    weights = rng.rand(784) * 2.0 - 1.0
    aggregator = geometric_median if task.method == "geomed" else radius_estimator
    checkpoint_set = set(task.checkpoints)
    points: list[Point] = []

    if 0 in checkpoint_set:
        points.append(
            evaluate(weights, test_images, test_labels, task, 0, num_minibatches)
        )

    for iteration in range(1, task.steps + 1):
        indices = rng.choice(n, task.budget, replace=False)
        images = train_images[indices].reshape(-1, 784).astype(np.float64) / 255.0
        selected_reverse = reverse_mask[indices]
        if np.any(selected_reverse):
            images[selected_reverse] = 1.0 - images[selected_reverse]
        labels = train_labels[indices]

        logits = images @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        example_gradients = (probabilities - labels)[:, None] * images
        minibatch_gradients = example_gradients.reshape(
            num_minibatches, task.batch_size, 784
        ).mean(axis=1)
        weights -= task.learning_rate * aggregator(minibatch_gradients)

        if iteration in checkpoint_set:
            points.append(
                evaluate(
                    weights,
                    test_images,
                    test_labels,
                    task,
                    iteration,
                    num_minibatches,
                )
            )
    return points


def write_points(path: Path, points: Iterable[Point]) -> None:
    rows = sorted(
        points,
        key=lambda p: (p.epsilon, p.batch_size, p.seed, p.method, p.iteration),
    )
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_points(path: Path) -> list[Point]:
    if not path.exists():
        return []
    points: list[Point] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            points.append(
                Point(
                    epsilon=float(row["epsilon"]),
                    batch_size=int(row["batch_size"]),
                    num_minibatches=int(row["num_minibatches"]),
                    seed=int(row["seed"]),
                    method=row["method"],
                    iteration=int(row["iteration"]),
                    learning_rate=float(row["learning_rate"]),
                    budget=int(row["budget"]),
                    test_accuracy=float(row["test_accuracy"]),
                    balanced_accuracy=float(row["balanced_accuracy"]),
                    positive_recall=float(row["positive_recall"]),
                    negative_recall=float(row["negative_recall"]),
                    test_logistic_loss=float(row["test_logistic_loss"]),
                    weight_norm=float(row["weight_norm"]),
                )
            )
    return points


def t_interval(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    if len(values) < 2:
        return center, center, center
    spread = stdev(values)
    if spread == 0.0:
        return center, center, center
    half_width = float(student_t.ppf(0.975, len(values) - 1)) * spread / math.sqrt(
        len(values)
    )
    return center, center - half_width, center + half_width


def summarize(points: list[Point], path: Path) -> list[dict[str, float | int]]:
    keyed = {
        (p.epsilon, p.batch_size, p.seed, p.method, p.iteration): p for p in points
    }
    groups = sorted({(p.epsilon, p.batch_size, p.iteration) for p in points})
    rows: list[dict[str, float | int]] = []
    for epsilon, batch_size, iteration in groups:
        seeds = sorted(
            {
                p.seed
                for p in points
                if p.epsilon == epsilon
                and p.batch_size == batch_size
                and p.iteration == iteration
            }
        )
        pairs = []
        for seed in seeds:
            gm = keyed.get((epsilon, batch_size, seed, "geomed", iteration))
            radius = keyed.get((epsilon, batch_size, seed, "radius", iteration))
            if gm is not None and radius is not None:
                pairs.append((gm, radius))
        if not pairs:
            continue
        gaps = [gm.test_accuracy - radius.test_accuracy for gm, radius in pairs]
        gap_mean, gap_low, gap_high = t_interval(gaps)
        rows.append(
            {
                "epsilon": epsilon,
                "batch_size": batch_size,
                "num_minibatches": pairs[0][0].num_minibatches,
                "iteration": iteration,
                "paired_seeds": len(pairs),
                "geomed_mean_accuracy": mean(gm.test_accuracy for gm, _ in pairs),
                "radius_mean_accuracy": mean(r.test_accuracy for _, r in pairs),
                "mean_accuracy_gap_geomed_minus_radius": gap_mean,
                "accuracy_gap_ci95_low": gap_low,
                "accuracy_gap_ci95_high": gap_high,
                "geomed_mean_balanced_accuracy": mean(
                    gm.balanced_accuracy for gm, _ in pairs
                ),
                "radius_mean_balanced_accuracy": mean(
                    r.balanced_accuracy for _, r in pairs
                ),
                "geomed_mean_test_loss": mean(
                    gm.test_logistic_loss for gm, _ in pairs
                ),
                "radius_mean_test_loss": mean(
                    r.test_logistic_loss for _, r in pairs
                ),
            }
        )
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plot_trajectories(
    points: list[Point], summary: list[dict[str, float | int]], output_dir: Path
) -> None:
    if not points or not summary:
        return
    import matplotlib.pyplot as plt

    conditions = sorted({(p.epsilon, p.batch_size) for p in points})
    colors = {"geomed": "#2563eb", "radius": "#dc2626"}
    labels = {"geomed": "GeoMed", "radius": "Radius"}
    fig, axes = plt.subplots(
        len(conditions), 2, figsize=(11.0, 3.5 * len(conditions)), squeeze=False
    )

    for row_index, (epsilon, batch_size) in enumerate(conditions):
        condition_points = [
            p
            for p in points
            if p.epsilon == epsilon and p.batch_size == batch_size
        ]
        iterations = sorted({p.iteration for p in condition_points})
        accuracy_axis = axes[row_index, 0]
        gap_axis = axes[row_index, 1]

        for method in METHODS:
            means = []
            lows = []
            highs = []
            for iteration in iterations:
                values = [
                    p.test_accuracy
                    for p in condition_points
                    if p.method == method and p.iteration == iteration
                ]
                center, low, high = t_interval(values)
                means.append(center)
                lows.append(low)
                highs.append(high)
            accuracy_axis.plot(
                iterations,
                means,
                color=colors[method],
                linewidth=2,
                marker="o",
                markersize=3,
                label=labels[method],
            )
            accuracy_axis.fill_between(
                iterations, lows, highs, color=colors[method], alpha=0.14
            )

        condition_summary = sorted(
            (
                row
                for row in summary
                if float(row["epsilon"]) == epsilon
                and int(row["batch_size"]) == batch_size
            ),
            key=lambda row: int(row["iteration"]),
        )
        gap_iterations = [int(row["iteration"]) for row in condition_summary]
        gap = np.array(
            [
                100.0 * float(row["mean_accuracy_gap_geomed_minus_radius"])
                for row in condition_summary
            ]
        )
        gap_low = np.array(
            [100.0 * float(row["accuracy_gap_ci95_low"]) for row in condition_summary]
        )
        gap_high = np.array(
            [
                100.0 * float(row["accuracy_gap_ci95_high"])
                for row in condition_summary
            ]
        )
        gap_axis.plot(gap_iterations, gap, color="#6d28d9", marker="o", markersize=3)
        gap_axis.fill_between(gap_iterations, gap_low, gap_high, color="#6d28d9", alpha=0.14)
        gap_axis.axhline(0.0, color="black", linewidth=1)

        for axis in (accuracy_axis, gap_axis):
            axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
            axis.grid(alpha=0.22)
            axis.set_xlabel("SGD updates")
        accuracy_axis.set_ylabel("Clean test accuracy")
        accuracy_axis.set_title(f"epsilon={epsilon:g}, b={batch_size}")
        accuracy_axis.legend()
        gap_axis.set_ylabel("GeoMed - Radius (percentage points)")
        gap_axis.set_title("Paired accuracy difference (95% t-CI)")

    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_trajectories.png", dpi=200)
    plt.close(fig)

    # A convergence-focused view removes the highly variable random-start
    # checkpoint and makes the post-1,000-update separation legible.
    fig, axes = plt.subplots(
        len(conditions), 1, figsize=(8.0, 3.4 * len(conditions)), squeeze=False
    )
    for row_index, (epsilon, batch_size) in enumerate(conditions):
        axis = axes[row_index, 0]
        condition_points = [
            p
            for p in points
            if p.epsilon == epsilon
            and p.batch_size == batch_size
            and p.iteration >= 1000
        ]
        iterations = sorted({p.iteration for p in condition_points})
        for method in METHODS:
            means = []
            lows = []
            highs = []
            for iteration in iterations:
                values = [
                    p.test_accuracy
                    for p in condition_points
                    if p.method == method and p.iteration == iteration
                ]
                center, low, high = t_interval(values)
                means.append(center)
                lows.append(low)
                highs.append(high)
            axis.plot(
                iterations,
                means,
                color=colors[method],
                linewidth=2,
                marker="o",
                markersize=3,
                label=labels[method],
            )
            axis.fill_between(iterations, lows, highs, color=colors[method], alpha=0.14)
        axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
        axis.axhline(0.8865, color="#111827", linestyle=":", linewidth=1.2,
                     label="Always-not-1 baseline")
        axis.grid(alpha=0.22)
        axis.set_xlabel("SGD updates")
        axis.set_ylabel("Clean test accuracy")
        axis.set_title(f"epsilon={epsilon:g}, b={batch_size} (from update 1,000)")
        axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_trajectories_zoom.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(conditions), 1, figsize=(8.0, 3.4 * len(conditions)), squeeze=False
    )
    for row_index, (epsilon, batch_size) in enumerate(conditions):
        axis = axes[row_index, 0]
        condition_points = [
            p
            for p in points
            if p.epsilon == epsilon and p.batch_size == batch_size
        ]
        iterations = sorted({p.iteration for p in condition_points})
        for method in METHODS:
            means = []
            lows = []
            highs = []
            for iteration in iterations:
                values = [
                    p.balanced_accuracy
                    for p in condition_points
                    if p.method == method and p.iteration == iteration
                ]
                center, low, high = t_interval(values)
                means.append(center)
                lows.append(low)
                highs.append(high)
            axis.plot(
                iterations,
                means,
                color=colors[method],
                linewidth=2,
                marker="o",
                markersize=3,
                label=labels[method],
            )
            axis.fill_between(iterations, lows, highs, color=colors[method], alpha=0.14)
        axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
        axis.axhline(0.5, color="#111827", linestyle=":", linewidth=1.2,
                     label="Chance balanced accuracy")
        axis.grid(alpha=0.22)
        axis.set_xlabel("SGD updates")
        axis.set_ylabel("Clean test balanced accuracy")
        axis.set_title(f"epsilon={epsilon:g}, b={batch_size}")
        axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "balanced_accuracy_trajectories.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(
        len(conditions), 1, figsize=(8.0, 3.4 * len(conditions)), squeeze=False
    )
    for row_index, (epsilon, batch_size) in enumerate(conditions):
        axis = axes[row_index, 0]
        condition_points = [
            p
            for p in points
            if p.epsilon == epsilon and p.batch_size == batch_size
        ]
        iterations = sorted({p.iteration for p in condition_points})
        for method in METHODS:
            means = []
            lows = []
            highs = []
            for iteration in iterations:
                values = [
                    p.test_logistic_loss
                    for p in condition_points
                    if p.method == method and p.iteration == iteration
                ]
                center, low, high = t_interval(values)
                means.append(center)
                lows.append(low)
                highs.append(high)
            axis.plot(
                iterations,
                means,
                color=colors[method],
                linewidth=2,
                marker="o",
                markersize=3,
                label=labels[method],
            )
            axis.fill_between(iterations, lows, highs, color=colors[method], alpha=0.14)
        axis.axvline(5000, color="#6b7280", linestyle="--", linewidth=1)
        axis.grid(alpha=0.22)
        axis.set_xlabel("SGD updates")
        axis.set_ylabel("Clean test logistic loss")
        axis.set_title(f"epsilon={epsilon:g}, b={batch_size}")
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_trajectories.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist-path",
        default=str(Path.home() / ".keras" / "datasets" / "mnist.npz"),
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.1, 0.2])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[
            0,
            100,
            250,
            500,
            750,
            1000,
            1500,
            2000,
            2500,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/accuracy_trajectories"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mnist_path = Path(args.mnist_path)
    if not mnist_path.exists():
        raise FileNotFoundError(mnist_path)
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if checkpoints[-1] > args.steps:
        raise ValueError("the largest checkpoint exceeds --steps")
    for batch_size in args.batch_sizes:
        if args.budget % batch_size:
            raise ValueError(f"batch size {batch_size} does not divide budget {args.budget}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_trajectories.csv"
    existing = read_points(raw_path)
    expected_iterations = set(checkpoints)
    completed = set()
    for epsilon in args.epsilons:
        for batch_size in args.batch_sizes:
            for seed in args.seeds:
                for method in METHODS:
                    observed = {
                        p.iteration
                        for p in existing
                        if p.epsilon == epsilon
                        and p.batch_size == batch_size
                        and p.seed == seed
                        and p.method == method
                        and p.learning_rate == args.learning_rate
                        and p.budget == args.budget
                    }
                    if expected_iterations <= observed:
                        completed.add((epsilon, batch_size, seed, method))

    tasks = [
        Task(
            epsilon=epsilon,
            batch_size=batch_size,
            seed=seed,
            method=method,
            steps=args.steps,
            learning_rate=args.learning_rate,
            budget=args.budget,
            checkpoints=checkpoints,
            mnist_path=str(mnist_path),
        )
        for epsilon in args.epsilons
        for batch_size in args.batch_sizes
        for seed in args.seeds
        for method in METHODS
        if (epsilon, batch_size, seed, method) not in completed
    ]

    points = list(existing)
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(train, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                task = futures[future]
                task_points = future.result()
                # Replace a partial prior run of the same task, if present.
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
                    f"[{index}/{len(tasks)}] eps={task.epsilon:g} "
                    f"b={task.batch_size} seed={task.seed} {task.method} "
                    f"accuracy={endpoint.test_accuracy:.4f} "
                    f"loss={endpoint.test_logistic_loss:.4f}",
                    flush=True,
                )

    selected = [
        p
        for p in points
        if p.epsilon in args.epsilons
        and p.batch_size in args.batch_sizes
        and p.seed in args.seeds
        and p.method in METHODS
        and p.iteration in expected_iterations
        and p.learning_rate == args.learning_rate
        and p.budget == args.budget
    ]
    summary = summarize(selected, args.output_dir / "trajectory_summary.csv")
    plot_trajectories(selected, summary, args.output_dir)
    metadata = {
        "question": (
            "At learning rate 0.001, are the methods stable by 5000 updates, "
            "and does GeoMed converge faster when it has the higher endpoint accuracy?"
        ),
        "epsilons": args.epsilons,
        "batch_sizes": args.batch_sizes,
        "seeds": args.seeds,
        "steps": args.steps,
        "checkpoints": list(checkpoints),
        "learning_rate": args.learning_rate,
        "budget": args.budget,
        "methods": list(METHODS),
        "mnist_source": os.path.abspath(mnist_path),
        "pairing": (
            "Same corruption mask, initialization, and sampled minibatches for both "
            "aggregators within each condition and seed."
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
