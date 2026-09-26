#!/usr/bin/env python3
"""Reproduce the MNIST GeoMed-versus-Radius minibatch-size comparison.

This script deliberately does not read any previously reported result table.
It reconstructs the experiment from the original notebook's model and
training choices:

* binary MNIST classification (digit 1 versus all other digits),
* logistic regression without an intercept,
* reverse-image corruption with labels unchanged,
* 640 sampled observations per SGD iteration,
* sampling without replacement within an iteration,
* geometric-median (Weiszfeld) or sample-centered radius aggregation.

The two methods receive identical corruptions, initializations, and sampled
minibatches within every (seed, epsilon, minibatch-size) comparison.
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


METHODS = ("geomed", "radius")


@dataclass(frozen=True)
class Task:
    epsilon: float
    batch_size: int
    seed: int
    method: str
    steps: int
    learning_rate: float
    budget: int
    mnist_path: str


@dataclass(frozen=True)
class Result:
    epsilon: float
    batch_size: int
    num_minibatches: int
    seed: int
    method: str
    steps: int
    learning_rate: float
    budget: int
    test_accuracy: float


def geometric_median(
    points: np.ndarray, tolerance: float = 1e-5, max_iterations: int = 100
) -> np.ndarray:
    """Match the Weiszfeld implementation used in the original notebooks."""
    center = points.mean(axis=0)
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - center, axis=1)
        nonzero = distances > tolerance
        if not np.any(nonzero):
            break
        weights = 1.0 / distances[nonzero]
        new_center = (points[nonzero] * weights[:, None]).sum(axis=0) / weights.sum()
        if np.linalg.norm(new_center - center) < tolerance:
            return new_center
        center = new_center
    return center


def radius_estimator(points: np.ndarray) -> np.ndarray:
    """Return the observed point with the smallest strict-majority radius."""
    centered = points - points.mean(axis=0)
    gram = centered @ centered.T
    squared_norms = np.diag(gram)
    squared_distances = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * gram
    )
    # Floating-point roundoff can create tiny negative squared distances.
    np.maximum(squared_distances, 0.0, out=squared_distances)
    np.fill_diagonal(squared_distances, 0.0)
    majority_index = len(points) // 2
    radii = np.partition(squared_distances, majority_index, axis=1)[
        :, majority_index
    ]
    return points[int(np.argmin(radii))]


def load_mnist(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path)
    train_images = archive["x_train"]
    train_labels = (archive["y_train"] == 1).astype(np.float64)
    test_images = archive["x_test"].reshape(-1, 784).astype(np.float64) / 255.0
    test_labels = (archive["y_test"] == 1).astype(np.float64)
    return train_images, train_labels, test_images, test_labels


def make_reverse_mask(n: int, epsilon: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(10_000 + seed)
    mask = np.zeros(n, dtype=bool)
    count = int(epsilon * n)
    if count:
        mask[rng.choice(n, count, replace=False)] = True
    return mask


def train(task: Task) -> Result:
    train_images, train_labels, test_images, test_labels = load_mnist(task.mnist_path)
    n = len(train_labels)
    if task.budget % task.batch_size:
        raise ValueError("budget must be divisible by every minibatch size")
    num_minibatches = task.budget // task.batch_size
    reverse_mask = make_reverse_mask(n, task.epsilon, task.seed)

    # A separate, paired stream controls initialization and all sampled batches.
    rng = np.random.RandomState(20_000 + task.seed)
    weights = rng.rand(784) * 2.0 - 1.0
    aggregator = geometric_median if task.method == "geomed" else radius_estimator

    for _ in range(task.steps):
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

    predictions = (test_images @ weights >= 0.0).astype(np.float64)
    accuracy = float(np.mean(predictions == test_labels))
    return Result(
        epsilon=task.epsilon,
        batch_size=task.batch_size,
        num_minibatches=num_minibatches,
        seed=task.seed,
        method=task.method,
        steps=task.steps,
        learning_rate=task.learning_rate,
        budget=task.budget,
        test_accuracy=accuracy,
    )


def write_results(path: Path, results: Iterable[Result]) -> None:
    rows = sorted(results, key=lambda r: (r.epsilon, r.batch_size, r.seed, r.method))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def read_results(path: Path) -> list[Result]:
    if not path.exists():
        return []
    results: list[Result] = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            results.append(
                Result(
                    epsilon=float(row["epsilon"]),
                    batch_size=int(row["batch_size"]),
                    num_minibatches=int(row["num_minibatches"]),
                    seed=int(row["seed"]),
                    method=row["method"],
                    steps=int(row["steps"]),
                    learning_rate=float(row["learning_rate"]),
                    budget=int(row["budget"]),
                    test_accuracy=float(row["test_accuracy"]),
                )
            )
    return results


def summarize(results: list[Result], path: Path) -> list[dict[str, float | int]]:
    keyed = {(r.epsilon, r.batch_size, r.seed, r.method): r for r in results}
    groups = sorted({(r.epsilon, r.batch_size) for r in results})
    summary: list[dict[str, float | int]] = []
    for epsilon, batch_size in groups:
        seeds = sorted(
            {
                r.seed
                for r in results
                if r.epsilon == epsilon and r.batch_size == batch_size
            }
        )
        paired = []
        for seed in seeds:
            gm = keyed.get((epsilon, batch_size, seed, "geomed"))
            radius = keyed.get((epsilon, batch_size, seed, "radius"))
            if gm and radius:
                paired.append((gm.test_accuracy, radius.test_accuracy))
        if not paired:
            continue
        gaps = [gm - radius for gm, radius in paired]
        gap_sd = stdev(gaps) if len(gaps) > 1 else 0.0
        if len(gaps) > 1 and gap_sd > 0.0:
            standard_error = gap_sd / math.sqrt(len(gaps))
            critical_value = float(student_t.ppf(0.975, len(gaps) - 1))
            half_width = critical_value * standard_error
            t_statistic = mean(gaps) / standard_error
            p_value = float(2.0 * student_t.sf(abs(t_statistic), len(gaps) - 1))
        elif len(gaps) > 1:
            half_width = 0.0
            p_value = 0.0 if mean(gaps) != 0.0 else 1.0
        else:
            half_width = 0.0
            p_value = float("nan")
        summary.append(
            {
                "epsilon": epsilon,
                "batch_size": batch_size,
                "num_minibatches": next(
                    r.num_minibatches
                    for r in results
                    if r.epsilon == epsilon and r.batch_size == batch_size
                ),
                "paired_seeds": len(paired),
                "geomed_mean_accuracy": mean(gm for gm, _ in paired),
                "radius_mean_accuracy": mean(radius for _, radius in paired),
                "mean_gap_geomed_minus_radius": mean(gaps),
                "gap_ci95_low": mean(gaps) - half_width,
                "gap_ci95_high": mean(gaps) + half_width,
                "paired_t_pvalue": p_value,
                "geomed_wins": sum(gap > 0 for gap in gaps),
                "radius_wins": sum(gap < 0 for gap in gaps),
                "ties": sum(gap == 0 for gap in gaps),
            }
        )
    if summary:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    return summary


def plot_summary(summary: list[dict[str, float | int]], path: Path) -> None:
    if not summary:
        return
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for epsilon in sorted({float(row["epsilon"]) for row in summary}):
        rows = [row for row in summary if float(row["epsilon"]) == epsilon]
        rows.sort(key=lambda row: int(row["batch_size"]))
        sizes = np.array([int(row["batch_size"]) for row in rows])
        gm = np.array([float(row["geomed_mean_accuracy"]) for row in rows])
        radius = np.array([float(row["radius_mean_accuracy"]) for row in rows])
        gap = np.array([float(row["mean_gap_geomed_minus_radius"]) for row in rows])
        low = np.array([float(row["gap_ci95_low"]) for row in rows])
        high = np.array([float(row["gap_ci95_high"]) for row in rows])

        axes[0].plot(sizes, gm, marker="o", label=f"GeoMed, eps={epsilon:g}")
        axes[0].plot(sizes, radius, marker="s", linestyle="--", label=f"Radius, eps={epsilon:g}")
        axes[1].errorbar(
            sizes,
            100.0 * gap,
            yerr=np.vstack((100.0 * (gap - low), 100.0 * (high - gap))),
            marker="o",
            capsize=3,
            label=f"eps={epsilon:g}",
        )

    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted({int(row["batch_size"]) for row in summary}))
        axis.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        axis.grid(alpha=0.25)
        axis.set_xlabel("Minibatch size b")
    axes[0].set_ylabel("Mean test accuracy")
    axes[0].set_title("Paired MNIST reruns")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_ylabel("GeoMed - Radius (percentage points)")
    axes[1].set_title("Paired accuracy difference (95% t-CI)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist-path",
        default=str(Path.home() / ".keras" / "datasets" / "mnist.npz"),
    )
    parser.add_argument("--epsilons", nargs="+", type=float, default=[0.2])
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=[2, 4, 8, 16, 32, 64, 128]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=5001)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/small_b_accuracy"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.mnist_path).exists():
        raise FileNotFoundError(args.mnist_path)
    for batch_size in args.batch_sizes:
        if args.budget % batch_size:
            raise ValueError(f"batch size {batch_size} does not divide budget {args.budget}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_results.csv"
    existing = read_results(raw_path)
    completed = {
        (r.epsilon, r.batch_size, r.seed, r.method, r.steps, r.learning_rate, r.budget)
        for r in existing
    }
    tasks = [
        Task(
            epsilon=epsilon,
            batch_size=batch_size,
            seed=seed,
            method=method,
            steps=args.steps,
            learning_rate=args.learning_rate,
            budget=args.budget,
            mnist_path=args.mnist_path,
        )
        for epsilon in args.epsilons
        for batch_size in args.batch_sizes
        for seed in args.seeds
        for method in METHODS
        if (
            epsilon,
            batch_size,
            seed,
            method,
            args.steps,
            args.learning_rate,
            args.budget,
        )
        not in completed
    ]

    results = list(existing)
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(train, task): task for task in tasks}
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                write_results(raw_path, results)
                print(
                    f"[{index}/{len(tasks)}] eps={result.epsilon:g} "
                    f"b={result.batch_size} seed={result.seed} "
                    f"{result.method} accuracy={result.test_accuracy:.4f}",
                    flush=True,
                )

    summary = summarize(results, args.output_dir / "summary.csv")
    plot_summary(summary, args.output_dir / "accuracy_comparison.png")
    metadata = {
        "claim_under_test": (
            "GeoMed has higher test accuracy for small b, and the difference "
            "shrinks for larger b."
        ),
        "epsilons": args.epsilons,
        "batch_sizes": args.batch_sizes,
        "seeds": args.seeds,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "budget": args.budget,
        "methods": list(METHODS),
        "mnist_source": os.path.abspath(args.mnist_path),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
