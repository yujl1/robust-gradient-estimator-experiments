#!/usr/bin/env python3
"""Matched GeoMed/Radius trajectories for ten-class MNIST softmax regression.

The corruption is the same audited, oblivious dataset-level corruption used in
the binary experiment: an exact epsilon fraction of training images is selected
once, each selected image is replaced by its pixelwise reverse, and labels are
left unchanged.  Within a seed, the two aggregators share the corruption mask,
a standard all-zero softmax initialization, and the complete sequence of
sampled observations.
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
    macro_accuracy: float
    test_cross_entropy: float
    weight_norm: float


def load_mnist(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path)
    return (
        archive["x_train"],
        archive["y_train"].astype(np.int64),
        archive["x_test"].reshape(-1, 784).astype(np.float32) / 255.0,
        archive["y_test"].astype(np.int64),
    )


def evaluate(
    weights: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    task: Task,
    iteration: int,
    num_minibatches: int,
) -> Point:
    logits = test_images @ weights
    predictions = np.argmax(logits, axis=1)
    recalls = [
        float(np.mean(predictions[test_labels == label] == label))
        for label in range(10)
    ]
    log_normalizer = np.logaddexp.reduce(logits.astype(np.float64), axis=1)
    losses = log_normalizer - logits[np.arange(len(test_labels)), test_labels]
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
        macro_accuracy=mean(recalls),
        test_cross_entropy=float(np.mean(losses)),
        weight_norm=float(np.linalg.norm(weights)),
    )


def train(task: Task) -> list[Point]:
    train_images, train_labels, test_images, test_labels = load_mnist(task.mnist_path)
    n = len(train_labels)
    if task.budget % task.batch_size:
        raise ValueError("budget must be divisible by batch size")
    if not task.checkpoints or task.checkpoints[0] < 0:
        raise ValueError("checkpoints must be nonnegative")
    if task.checkpoints[-1] > task.steps:
        raise ValueError("a checkpoint exceeds the requested number of steps")

    num_minibatches = task.budget // task.batch_size
    reverse_mask = make_reverse_mask(n, task.epsilon, task.seed)
    rng = np.random.RandomState(20_000 + task.seed)
    # Zero is a neutral and fully reproducible start for convex multinomial
    # logistic regression. Reusing the binary experiment's [-1, 1] coordinate
    # scale would give the ten-column model oversized initial logits.
    weights = np.zeros((784, 10), dtype=np.float32)
    aggregator = geometric_median if task.method == "geomed" else radius_estimator
    checkpoint_set = set(task.checkpoints)
    points: list[Point] = []

    if 0 in checkpoint_set:
        points.append(
            evaluate(weights, test_images, test_labels, task, 0, num_minibatches)
        )

    row_indices = np.arange(task.budget)
    for iteration in range(1, task.steps + 1):
        indices = rng.choice(n, task.budget, replace=False)
        images = train_images[indices].reshape(-1, 784).astype(np.float32) / 255.0
        selected_reverse = reverse_mask[indices]
        if np.any(selected_reverse):
            images[selected_reverse] = 1.0 - images[selected_reverse]
        labels = train_labels[indices]

        logits = images @ weights
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        probabilities[row_indices, labels] -= 1.0
        image_batches = images.reshape(
            num_minibatches, task.batch_size, 784
        )
        error_batches = probabilities.reshape(
            num_minibatches, task.batch_size, 10
        )
        minibatch_gradients = np.einsum(
            "mbd,mbc->mdc", image_batches, error_batches, optimize=True
        ) / task.batch_size
        aggregate = aggregator(minibatch_gradients.reshape(num_minibatches, -1))
        weights -= task.learning_rate * aggregate.reshape(784, 10)

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
    with path.open(newline="") as stream:
        return [
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
                macro_accuracy=float(row["macro_accuracy"]),
                test_cross_entropy=float(row["test_cross_entropy"]),
                weight_norm=float(row["weight_norm"]),
            )
            for row in csv.DictReader(stream)
        ]


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
        accuracy_gaps = [gm.test_accuracy - r.test_accuracy for gm, r in pairs]
        macro_gaps = [gm.macro_accuracy - r.macro_accuracy for gm, r in pairs]
        accuracy_mean, accuracy_low, accuracy_high = t_interval(accuracy_gaps)
        macro_mean, macro_low, macro_high = t_interval(macro_gaps)
        rows.append(
            {
                "epsilon": epsilon,
                "batch_size": batch_size,
                "num_minibatches": pairs[0][0].num_minibatches,
                "iteration": iteration,
                "paired_seeds": len(pairs),
                "geomed_mean_accuracy": mean(g.test_accuracy for g, _ in pairs),
                "radius_mean_accuracy": mean(r.test_accuracy for _, r in pairs),
                "mean_accuracy_gap_geomed_minus_radius": accuracy_mean,
                "accuracy_gap_ci95_low": accuracy_low,
                "accuracy_gap_ci95_high": accuracy_high,
                "geomed_mean_macro_accuracy": mean(g.macro_accuracy for g, _ in pairs),
                "radius_mean_macro_accuracy": mean(r.macro_accuracy for _, r in pairs),
                "mean_macro_gap_geomed_minus_radius": macro_mean,
                "macro_gap_ci95_low": macro_low,
                "macro_gap_ci95_high": macro_high,
                "geomed_mean_cross_entropy": mean(
                    g.test_cross_entropy for g, _ in pairs
                ),
                "radius_mean_cross_entropy": mean(
                    r.test_cross_entropy for _, r in pairs
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


def plot(points: list[Point], summary: list[dict[str, float | int]], output_dir: Path) -> None:
    if not points or not summary:
        return
    import matplotlib.pyplot as plt

    colors = {"geomed": "#2563eb", "radius": "#dc2626"}
    labels = {"geomed": "GeoMed", "radius": "Radius"}
    iterations = sorted({p.iteration for p in points})
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for axis, (metric, title, ylabel) in zip(
        axes,
        (
            ("test_accuracy", "Ten-class test accuracy", "Accuracy"),
            ("macro_accuracy", "Macro-averaged class accuracy", "Macro accuracy"),
            ("test_cross_entropy", "Clean test cross-entropy", "Cross-entropy"),
        ),
    ):
        for method in METHODS:
            centers, lows, highs = [], [], []
            for iteration in iterations:
                values = [
                    float(getattr(p, metric))
                    for p in points
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
                linewidth=2.2,
                marker="o",
                markersize=3,
                label=labels[method],
            )
            axis.fill_between(iterations, lows, highs, color=colors[method], alpha=0.14)
        if metric != "test_cross_entropy":
            axis.axhline(0.1, color="#111827", linestyle=":", linewidth=1.2,
                         label="Chance")
        axis.grid(alpha=0.22)
        axis.set_xlabel("SGD updates")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend(fontsize=8)
    fig.suptitle(
        f"MNIST softmax regression: epsilon={points[0].epsilon:g}, "
        f"b={points[0].batch_size}, lr={points[0].learning_rate:g}"
    )
    fig.tight_layout()
    fig.savefig(output_dir / "multiclass_trajectories.png", dpi=200)
    plt.close(fig)

    condition_summary = sorted(summary, key=lambda row: int(row["iteration"]))
    x = [int(row["iteration"]) for row in condition_summary]
    gap = 100.0 * np.array(
        [float(row["mean_accuracy_gap_geomed_minus_radius"]) for row in condition_summary]
    )
    low = 100.0 * np.array(
        [float(row["accuracy_gap_ci95_low"]) for row in condition_summary]
    )
    high = 100.0 * np.array(
        [float(row["accuracy_gap_ci95_high"]) for row in condition_summary]
    )
    fig, axis = plt.subplots(figsize=(8.0, 4.2))
    axis.plot(x, gap, color="#6d28d9", linewidth=2.2, marker="o", markersize=3)
    axis.fill_between(x, low, high, color="#6d28d9", alpha=0.16)
    axis.axhline(0.0, color="black", linewidth=1)
    axis.grid(alpha=0.22)
    axis.set_xlabel("SGD updates")
    axis.set_ylabel("GeoMed - Radius accuracy (percentage points)")
    axis.set_title("Paired ten-class accuracy difference (95% t-CI)")
    fig.tight_layout()
    fig.savefig(output_dir / "multiclass_accuracy_gap.png", dpi=200)
    plt.close(fig)

    # Compact two-panel version sized for the workshop manuscript.
    fig, (accuracy_axis, gap_axis) = plt.subplots(1, 2, figsize=(8.0, 3.0))
    for method in METHODS:
        centers, lows, highs = [], [], []
        for iteration in iterations:
            values = [
                p.test_accuracy
                for p in points
                if p.method == method and p.iteration == iteration
            ]
            center, low_bound, high_bound = t_interval(values)
            centers.append(center)
            lows.append(low_bound)
            highs.append(high_bound)
        accuracy_axis.plot(
            iterations,
            centers,
            color=colors[method],
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            label=labels[method],
        )
        accuracy_axis.fill_between(
            iterations, lows, highs, color=colors[method], alpha=0.14
        )
    accuracy_axis.axhline(
        0.1, color="#111827", linestyle=":", linewidth=1.0, label="Chance"
    )
    accuracy_axis.set_xlabel("SGD updates")
    accuracy_axis.set_ylabel("Clean test accuracy")
    accuracy_axis.set_title("Ten-class accuracy")
    accuracy_axis.set_ylim(0.08, 0.93)
    accuracy_axis.grid(alpha=0.22)
    accuracy_axis.legend(fontsize=8, loc="lower right")

    late = np.array(x) >= 500
    late_x = np.array(x)[late]
    gap_axis.plot(
        late_x, gap[late], color="#6d28d9", linewidth=2.0,
        marker="o", markersize=2.8
    )
    gap_axis.fill_between(late_x, low[late], high[late], color="#6d28d9", alpha=0.16)
    gap_axis.axhline(0.0, color="black", linewidth=1.0)
    gap_axis.set_xlabel("SGD updates")
    gap_axis.set_ylabel("GeoMed - Radius (points)")
    gap_axis.set_title("Paired gap after 500 updates")
    gap_floor = min(0.0, float(np.min(low[late])))
    gap_ceiling = max(0.0, float(np.max(high[late])))
    gap_padding = max(0.15, 0.08 * (gap_ceiling - gap_floor))
    gap_axis.set_ylim(gap_floor - gap_padding, gap_ceiling + gap_padding)
    gap_axis.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_dir / "multiclass_paper_figure.png", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist-path", default=str(Path.home() / ".keras" / "datasets" / "mnist.npz")
    )
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--workers", type=int, default=1)
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
        default=Path("output/experiments/multiclass_eps01_lr001"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mnist_path = Path(args.mnist_path)
    if not mnist_path.exists():
        raise FileNotFoundError(mnist_path)
    if args.budget % args.batch_size:
        raise ValueError("batch size must divide budget")
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if checkpoints[-1] > args.steps:
        raise ValueError("the largest checkpoint exceeds --steps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_trajectories.csv"
    existing = read_points(raw_path)
    expected_iterations = set(checkpoints)
    completed = set()
    for seed in args.seeds:
        for method in METHODS:
            observed = {
                p.iteration
                for p in existing
                if p.epsilon == args.epsilon
                and p.batch_size == args.batch_size
                and p.seed == seed
                and p.method == method
                and p.learning_rate == args.learning_rate
                and p.budget == args.budget
            }
            if expected_iterations <= observed:
                completed.add((seed, method))

    tasks = [
        Task(
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            seed=seed,
            method=method,
            steps=args.steps,
            learning_rate=args.learning_rate,
            budget=args.budget,
            checkpoints=checkpoints,
            mnist_path=str(mnist_path),
        )
        for seed in args.seeds
        for method in METHODS
        if (seed, method) not in completed
    ]

    points = list(existing)
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
                    f"[{index}/{len(tasks)}] seed={task.seed} {task.method} "
                    f"accuracy={endpoint.test_accuracy:.4f} "
                    f"macro={endpoint.macro_accuracy:.4f} "
                    f"loss={endpoint.test_cross_entropy:.4f}",
                    flush=True,
                )

    selected = [
        p
        for p in points
        if p.epsilon == args.epsilon
        and p.batch_size == args.batch_size
        and p.seed in args.seeds
        and p.method in METHODS
        and p.iteration in expected_iterations
        and p.learning_rate == args.learning_rate
        and p.budget == args.budget
    ]
    summary = summarize(selected, args.output_dir / "trajectory_summary.csv")
    plot(selected, summary, args.output_dir)
    metadata = {
        "question": (
            "In ten-class MNIST softmax regression at epsilon=0.1 and lr=0.01, "
            "how do GeoMed and Radius compare during training?"
        ),
        "task": "ten-class MNIST softmax regression without intercept",
        "corruption": (
            "An exact epsilon fraction of training images is selected once and "
            "pixelwise reversed; labels are unchanged; the clean test set is used."
        ),
        "epsilon": args.epsilon,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "steps": args.steps,
        "checkpoints": list(checkpoints),
        "learning_rate": args.learning_rate,
        "budget": args.budget,
        "methods": list(METHODS),
        "mnist_source": os.path.abspath(mnist_path),
        "pairing": (
            "Same corruption mask, initialization, and sampled observations for both "
            "aggregators within every seed."
        ),
        "initialization": "All-zero 784-by-10 weight matrix.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
