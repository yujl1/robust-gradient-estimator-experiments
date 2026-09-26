#!/usr/bin/env python3
"""Compare GeoMed and Radius under several static MNIST corruptions.

Every corruption is applied to an exact epsilon fraction of the 60,000 MNIST
training observations once, before optimization.  Within each seed, both
aggregators—and all corruption conditions—share the selected observations,
zero initialization, and complete sequence of sampled training indices.

The requested static target-class attack is represented in two forms:

* target_prototype: every selected observation is replaced by the same
  deterministic, central-looking training image of the target digit and given
  the target label;
* target_diverse: every selected observation is replaced once by a fixed draw
  from the target-class training examples and given the target label.

The other conditions are a clean control, reverse-image feature corruption,
targeted label poisoning, and cyclic label poisoning.  None is adaptive to the
optimization trajectory.
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

from reproduce_small_b_accuracy import METHODS, geometric_median, radius_estimator


ATTACKS = (
    "none",
    "reverse_pixels",
    "target_prototype",
    "target_diverse",
    "target_label",
    "cyclic_label",
)

ATTACK_LABELS = {
    "none": "Clean",
    "reverse_pixels": "Reverse pixels",
    "target_prototype": "Target prototype",
    "target_diverse": "Target diverse",
    "target_label": "Target label",
    "cyclic_label": "Cyclic label",
}


@dataclass(frozen=True)
class Task:
    attack: str
    epsilon: float
    target_class: int
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
    attack: str
    epsilon: float
    target_class: int
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
    target_recall: float
    nontarget_to_target_rate: float
    predicted_target_fraction: float
    weight_norm: float


def load_mnist(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    archive = np.load(path)
    return (
        archive["x_train"],
        archive["y_train"].astype(np.int64),
        archive["x_test"].reshape(-1, 784).astype(np.float32) / 255.0,
        archive["y_test"].astype(np.int64),
    )


def exact_corruption_mask(n: int, epsilon: float, seed: int) -> np.ndarray:
    """Select exactly floor(epsilon*n) observations, identically across attacks."""
    rng = np.random.RandomState(10_000 + seed)
    mask = np.zeros(n, dtype=bool)
    count = int(epsilon * n)
    if count:
        mask[rng.choice(n, count, replace=False)] = True
    return mask


def central_target_prototype(
    train_images: np.ndarray, train_labels: np.ndarray, target_class: int
) -> tuple[np.ndarray, int]:
    """Choose the target example nearest its class's pixelwise mean."""
    candidate_indices = np.flatnonzero(train_labels == target_class)
    candidates = train_images[candidate_indices].reshape(-1, 784).astype(np.float32)
    class_mean = candidates.mean(axis=0)
    distances = np.einsum(
        "ij,ij->i", candidates - class_mean, candidates - class_mean, optimize=True
    )
    local_index = int(np.argmin(distances))
    return candidates[local_index].astype(np.uint8), int(candidate_indices[local_index])


def fixed_target_sources(
    n: int,
    mask: np.ndarray,
    train_labels: np.ndarray,
    target_class: int,
    seed: int,
) -> np.ndarray:
    """Map each corrupted row to a target-class source once, before training."""
    rng = np.random.RandomState(30_000 + 97 * target_class + seed)
    candidates = np.flatnonzero(train_labels == target_class)
    mapping = np.full(n, -1, dtype=np.int64)
    mapping[mask] = rng.choice(candidates, int(mask.sum()), replace=True)
    return mapping


def apply_corruption(
    attack: str,
    indices: np.ndarray,
    images: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    target_class: int,
    prototype: np.ndarray,
    target_sources: np.ndarray,
    train_images: np.ndarray,
) -> None:
    """Apply a precommitted dataset corruption to one sampled training batch."""
    selected = mask[indices]
    if not np.any(selected) or attack == "none":
        return
    if attack == "reverse_pixels":
        images[selected] = 1.0 - images[selected]
    elif attack == "target_prototype":
        images[selected] = prototype.astype(np.float32) / 255.0
        labels[selected] = target_class
    elif attack == "target_diverse":
        source_indices = target_sources[indices[selected]]
        images[selected] = (
            train_images[source_indices].reshape(-1, 784).astype(np.float32) / 255.0
        )
        labels[selected] = target_class
    elif attack == "target_label":
        labels[selected] = target_class
    elif attack == "cyclic_label":
        labels[selected] = (labels[selected] + 1) % 10
    else:
        raise ValueError(f"unknown attack: {attack}")


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
    nontarget = test_labels != task.target_class
    return Point(
        attack=task.attack,
        epsilon=task.epsilon,
        target_class=task.target_class,
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
        target_recall=recalls[task.target_class],
        nontarget_to_target_rate=float(
            np.mean(predictions[nontarget] == task.target_class)
        ),
        predicted_target_fraction=float(np.mean(predictions == task.target_class)),
        weight_norm=float(np.linalg.norm(weights)),
    )


def train(task: Task) -> list[Point]:
    train_images, train_labels, test_images, test_labels = load_mnist(task.mnist_path)
    n = len(train_labels)
    if task.attack not in ATTACKS:
        raise ValueError(f"unknown attack: {task.attack}")
    if task.budget % task.batch_size:
        raise ValueError("budget must be divisible by batch size")
    if task.checkpoints[-1] > task.steps:
        raise ValueError("a checkpoint exceeds the requested number of steps")

    effective_epsilon = 0.0 if task.attack == "none" else task.epsilon
    mask = exact_corruption_mask(n, effective_epsilon, task.seed)
    prototype, _ = central_target_prototype(
        train_images, train_labels, task.target_class
    )
    target_sources = fixed_target_sources(
        n, mask, train_labels, task.target_class, task.seed
    )

    num_minibatches = task.budget // task.batch_size
    rng = np.random.RandomState(20_000 + task.seed)
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
        labels = train_labels[indices].copy()
        apply_corruption(
            task.attack,
            indices,
            images,
            labels,
            mask,
            task.target_class,
            prototype,
            target_sources,
            train_images,
        )

        logits = images @ weights
        logits -= np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities, axis=1, keepdims=True)
        probabilities[row_indices, labels] -= 1.0
        image_batches = images.reshape(num_minibatches, task.batch_size, 784)
        error_batches = probabilities.reshape(num_minibatches, task.batch_size, 10)
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
        key=lambda p: (
            p.batch_size,
            p.attack,
            p.epsilon,
            p.seed,
            p.method,
            p.iteration,
        ),
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
                attack=row["attack"],
                epsilon=float(row["epsilon"]),
                target_class=int(row["target_class"]),
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
                target_recall=float(row["target_recall"]),
                nontarget_to_target_rate=float(row["nontarget_to_target_rate"]),
                predicted_target_fraction=float(row["predicted_target_fraction"]),
                weight_norm=float(row["weight_norm"]),
            )
            for row in csv.DictReader(stream)
        ]


def interval(values: list[float]) -> tuple[float, float, float]:
    center = mean(values)
    if len(values) < 2 or stdev(values) == 0.0:
        return center, center, center
    half_width = (
        float(student_t.ppf(0.975, len(values) - 1))
        * stdev(values)
        / math.sqrt(len(values))
    )
    return center, center - half_width, center + half_width


def paired_pvalue(values: list[float]) -> float:
    if len(values) < 2:
        return float("nan")
    spread = stdev(values)
    if spread == 0.0:
        return 0.0 if mean(values) != 0.0 else 1.0
    statistic = mean(values) / (spread / math.sqrt(len(values)))
    return float(2.0 * student_t.sf(abs(statistic), len(values) - 1))


def summarize(points: list[Point], path: Path) -> list[dict[str, float | int | str]]:
    keyed = {
        (p.attack, p.epsilon, p.batch_size, p.seed, p.method, p.iteration): p
        for p in points
    }
    groups = sorted(
        {(p.batch_size, p.attack, p.epsilon, p.iteration) for p in points}
    )
    rows: list[dict[str, float | int | str]] = []
    for batch_size, attack, epsilon, iteration in groups:
        seeds = sorted(
            {
                p.seed
                for p in points
                if p.batch_size == batch_size
                and p.attack == attack
                and p.epsilon == epsilon
                and p.iteration == iteration
            }
        )
        pairs = []
        for seed in seeds:
            gm = keyed.get((attack, epsilon, batch_size, seed, "geomed", iteration))
            radius = keyed.get(
                (attack, epsilon, batch_size, seed, "radius", iteration)
            )
            if gm is not None and radius is not None:
                pairs.append((gm, radius))
        if not pairs:
            continue
        gaps = [gm.test_accuracy - radius.test_accuracy for gm, radius in pairs]
        gap_mean, gap_low, gap_high = interval(gaps)
        macro_gaps = [gm.macro_accuracy - radius.macro_accuracy for gm, radius in pairs]
        target_fpr_gaps = [
            gm.nontarget_to_target_rate - radius.nontarget_to_target_rate
            for gm, radius in pairs
        ]
        rows.append(
            {
                "attack": attack,
                "epsilon": epsilon,
                "target_class": pairs[0][0].target_class,
                "batch_size": batch_size,
                "num_minibatches": pairs[0][0].num_minibatches,
                "iteration": iteration,
                "paired_seeds": len(pairs),
                "geomed_mean_accuracy": mean(g.test_accuracy for g, _ in pairs),
                "radius_mean_accuracy": mean(r.test_accuracy for _, r in pairs),
                "mean_accuracy_gap_geomed_minus_radius": gap_mean,
                "accuracy_gap_ci95_low": gap_low,
                "accuracy_gap_ci95_high": gap_high,
                "accuracy_gap_pvalue": paired_pvalue(gaps),
                "geomed_wins": sum(value > 0 for value in gaps),
                "radius_wins": sum(value < 0 for value in gaps),
                "ties": sum(value == 0 for value in gaps),
                "geomed_mean_macro_accuracy": mean(
                    g.macro_accuracy for g, _ in pairs
                ),
                "radius_mean_macro_accuracy": mean(
                    r.macro_accuracy for _, r in pairs
                ),
                "mean_macro_gap_geomed_minus_radius": mean(macro_gaps),
                "geomed_mean_cross_entropy": mean(
                    g.test_cross_entropy for g, _ in pairs
                ),
                "radius_mean_cross_entropy": mean(
                    r.test_cross_entropy for _, r in pairs
                ),
                "geomed_mean_target_recall": mean(
                    g.target_recall for g, _ in pairs
                ),
                "radius_mean_target_recall": mean(
                    r.target_recall for _, r in pairs
                ),
                "geomed_mean_nontarget_to_target_rate": mean(
                    g.nontarget_to_target_rate for g, _ in pairs
                ),
                "radius_mean_nontarget_to_target_rate": mean(
                    r.nontarget_to_target_rate for _, r in pairs
                ),
                "mean_target_fpr_gap_geomed_minus_radius": mean(target_fpr_gaps),
            }
        )
    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def summarize_attack_effects(
    points: list[Point], path: Path
) -> list[dict[str, float | int | str]]:
    """Compare each attack with the paired clean run at matching checkpoints."""
    keyed = {
        (p.attack, p.batch_size, p.seed, p.method, p.iteration): p for p in points
    }
    groups = sorted(
        {
            (p.batch_size, p.attack, p.epsilon, p.iteration)
            for p in points
            if p.attack != "none"
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for batch_size, attack, epsilon, iteration in groups:
        per_seed = []
        for seed in sorted({p.seed for p in points}):
            gm_attack = keyed.get((attack, batch_size, seed, "geomed", iteration))
            r_attack = keyed.get((attack, batch_size, seed, "radius", iteration))
            gm_clean = keyed.get(("none", batch_size, seed, "geomed", iteration))
            r_clean = keyed.get(("none", batch_size, seed, "radius", iteration))
            if all(x is not None for x in (gm_attack, r_attack, gm_clean, r_clean)):
                per_seed.append((gm_attack, r_attack, gm_clean, r_clean))
        if not per_seed:
            continue
        gm_changes = [a.test_accuracy - c.test_accuracy for a, _, c, _ in per_seed]
        r_changes = [a.test_accuracy - c.test_accuracy for _, a, _, c in per_seed]
        differences = [g - r for g, r in zip(gm_changes, r_changes)]
        diff_mean, diff_low, diff_high = interval(differences)
        rows.append(
            {
                "attack": attack,
                "epsilon": epsilon,
                "batch_size": batch_size,
                "iteration": iteration,
                "paired_seeds": len(per_seed),
                "geomed_mean_accuracy_change_attack_minus_clean": mean(gm_changes),
                "radius_mean_accuracy_change_attack_minus_clean": mean(r_changes),
                "mean_difference_in_differences": diff_mean,
                "difference_in_differences_ci95_low": diff_low,
                "difference_in_differences_ci95_high": diff_high,
                "difference_in_differences_pvalue": paired_pvalue(differences),
            }
        )
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def plot(
    points: list[Point],
    summary: list[dict[str, float | int | str]],
    output_dir: Path,
) -> None:
    if not points or not summary:
        return
    import matplotlib.pyplot as plt

    colors = {"geomed": "#2563eb", "radius": "#dc2626"}
    labels = {"geomed": "GeoMed", "radius": "Radius"}
    attacks = [attack for attack in ATTACKS if any(p.attack == attack for p in points)]
    batch_sizes = sorted({p.batch_size for p in points})

    for batch_size in batch_sizes:
        ncols = 3
        nrows = math.ceil(len(attacks) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(12.0, 3.25 * nrows), squeeze=False
        )
        for axis, attack in zip(axes.flat, attacks):
            attack_points = [
                p for p in points if p.attack == attack and p.batch_size == batch_size
            ]
            iterations = sorted({p.iteration for p in attack_points})
            for method in METHODS:
                centers, lows, highs = [], [], []
                for iteration in iterations:
                    values = [
                        p.test_accuracy
                        for p in attack_points
                        if p.method == method and p.iteration == iteration
                    ]
                    center, low, high = interval(values)
                    centers.append(center)
                    lows.append(low)
                    highs.append(high)
                axis.plot(
                    iterations,
                    centers,
                    color=colors[method],
                    linewidth=2.0,
                    marker="o",
                    markersize=2.5,
                    label=labels[method],
                )
                axis.fill_between(
                    iterations, lows, highs, color=colors[method], alpha=0.13
                )
            axis.set_title(ATTACK_LABELS[attack])
            axis.set_xlabel("SGD updates")
            axis.set_ylabel("Clean test accuracy")
            axis.set_ylim(0.08, 0.95)
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8, loc="lower right")
        for axis in axes.flat[len(attacks) :]:
            axis.axis("off")
        fig.suptitle(
            f"Static MNIST corruptions: b={batch_size}, paired mean and 95% t-CI"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"accuracy_trajectories_b{batch_size}.png", dpi=220)
        plt.close(fig)

        attack_summary = [
            row for row in summary if int(row["batch_size"]) == batch_size
        ]
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(12.0, 3.25 * nrows), squeeze=False
        )
        for axis, attack in zip(axes.flat, attacks):
            rows = sorted(
                (row for row in attack_summary if row["attack"] == attack),
                key=lambda row: int(row["iteration"]),
            )
            x = np.array([int(row["iteration"]) for row in rows])
            gap = 100.0 * np.array(
                [float(row["mean_accuracy_gap_geomed_minus_radius"]) for row in rows]
            )
            low = 100.0 * np.array(
                [float(row["accuracy_gap_ci95_low"]) for row in rows]
            )
            high = 100.0 * np.array(
                [float(row["accuracy_gap_ci95_high"]) for row in rows]
            )
            axis.plot(x, gap, color="#6d28d9", linewidth=2.0, marker="o", markersize=2.5)
            axis.fill_between(x, low, high, color="#6d28d9", alpha=0.15)
            axis.axhline(0.0, color="black", linewidth=1.0)
            axis.set_title(ATTACK_LABELS[attack])
            axis.set_xlabel("SGD updates")
            axis.set_ylabel("GeoMed - Radius (points)")
            axis.grid(alpha=0.22)
        for axis in axes.flat[len(attacks) :]:
            axis.axis("off")
        fig.suptitle(
            f"Paired aggregator gaps across static corruptions: b={batch_size}"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"accuracy_gaps_b{batch_size}.png", dpi=220)
        plt.close(fig)

        endpoints = []
        for attack in attacks:
            rows = [
                row
                for row in attack_summary
                if row["attack"] == attack
            ]
            if rows:
                endpoints.append(max(rows, key=lambda row: int(row["iteration"])))
        positions = np.arange(len(endpoints))
        gm = 100.0 * np.array([float(row["geomed_mean_accuracy"]) for row in endpoints])
        radius = 100.0 * np.array([float(row["radius_mean_accuracy"]) for row in endpoints])
        width = 0.36
        fig, axis = plt.subplots(figsize=(9.5, 4.2))
        axis.bar(positions - width / 2, gm, width, color=colors["geomed"], label="GeoMed")
        axis.bar(positions + width / 2, radius, width, color=colors["radius"], label="Radius")
        axis.set_xticks(
            positions,
            [ATTACK_LABELS[str(row["attack"])] for row in endpoints],
            rotation=18,
            ha="right",
        )
        axis.set_ylabel("Mean clean test accuracy (%)")
        axis.set_ylim(0, 100)
        axis.set_title(f"Endpoint comparison at b={batch_size}")
        axis.grid(axis="y", alpha=0.22)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"endpoint_comparison_b{batch_size}.png", dpi=240)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mnist-path", default=str(Path.home() / ".keras" / "datasets" / "mnist.npz")
    )
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--target-class", type=int, default=0)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[0, 25, 50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/static_corruption_suite"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mnist_path = Path(args.mnist_path)
    if not mnist_path.exists():
        raise FileNotFoundError(mnist_path)
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")
    if not 0 <= args.target_class <= 9:
        raise ValueError("target class must lie in {0,...,9}")
    for batch_size in args.batch_sizes:
        if args.budget % batch_size:
            raise ValueError(f"batch size {batch_size} does not divide budget")
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if not checkpoints or checkpoints[0] < 0 or checkpoints[-1] > args.steps:
        raise ValueError("checkpoints must lie between zero and --steps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_trajectories.csv"
    existing = read_points(raw_path)
    expected_iterations = set(checkpoints)
    completed = set()
    for attack in args.attacks:
        epsilon = 0.0 if attack == "none" else args.epsilon
        for batch_size in args.batch_sizes:
            for seed in args.seeds:
                for method in METHODS:
                    observed = {
                        p.iteration
                        for p in existing
                        if p.attack == attack
                        and p.epsilon == epsilon
                        and p.target_class == args.target_class
                        and p.batch_size == batch_size
                        and p.seed == seed
                        and p.method == method
                        and p.learning_rate == args.learning_rate
                        and p.budget == args.budget
                    }
                    if expected_iterations <= observed:
                        completed.add((attack, batch_size, seed, method))

    tasks = [
        Task(
            attack=attack,
            epsilon=0.0 if attack == "none" else args.epsilon,
            target_class=args.target_class,
            batch_size=batch_size,
            seed=seed,
            method=method,
            steps=args.steps,
            learning_rate=args.learning_rate,
            budget=args.budget,
            checkpoints=checkpoints,
            mnist_path=str(mnist_path),
        )
        for attack in args.attacks
        for batch_size in args.batch_sizes
        for seed in args.seeds
        for method in METHODS
        if (attack, batch_size, seed, method) not in completed
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
                        p.attack == task.attack
                        and p.epsilon == task.epsilon
                        and p.target_class == task.target_class
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
                    f"[{index}/{len(tasks)}] {task.attack} seed={task.seed} "
                    f"{task.method} accuracy={endpoint.test_accuracy:.4f} "
                    f"macro={endpoint.macro_accuracy:.4f} "
                    f"target_fpr={endpoint.nontarget_to_target_rate:.4f}",
                    flush=True,
                )

    allowed_epsilons = {
        attack: 0.0 if attack == "none" else args.epsilon for attack in args.attacks
    }
    selected = [
        p
        for p in points
        if p.attack in args.attacks
        and p.epsilon == allowed_epsilons[p.attack]
        and p.target_class == args.target_class
        and p.batch_size in args.batch_sizes
        and p.seed in args.seeds
        and p.method in METHODS
        and p.iteration in expected_iterations
        and p.learning_rate == args.learning_rate
        and p.budget == args.budget
    ]
    summary = summarize(selected, args.output_dir / "trajectory_summary.csv")
    summarize_attack_effects(selected, args.output_dir / "attack_effect_summary.csv")
    plot(selected, summary, args.output_dir)

    train_images, train_labels, _, _ = load_mnist(str(mnist_path))
    _, prototype_index = central_target_prototype(
        train_images, train_labels, args.target_class
    )
    metadata = {
        "question": (
            "Does the GeoMed-versus-Radius distinction at small minibatch size "
            "persist across several fixed, dataset-level MNIST corruptions?"
        ),
        "task": "ten-class MNIST linear softmax regression without intercept",
        "attacks": {
            "none": "Unmodified training set; epsilon recorded as zero.",
            "reverse_pixels": (
                "For selected rows, replace x by 1-x and retain the original label."
            ),
            "target_prototype": (
                "Replace every selected row by the same clean target-class prototype "
                "and the target label."
            ),
            "target_diverse": (
                "Replace each selected row once by a seeded target-class training "
                "example and the target label."
            ),
            "target_label": (
                "Retain each selected image and replace its label by the target label."
            ),
            "cyclic_label": (
                "Retain each selected image and replace y by (y+1) modulo 10."
            ),
        },
        "obliviousness": (
            "The exact corruption mask and all replacements are fixed before training; "
            "the adversary never observes iterates or sampled minibatches."
        ),
        "epsilon": args.epsilon,
        "included_attacks": args.attacks,
        "target_class": args.target_class,
        "target_prototype_rule": (
            "MNIST target-class training image nearest the class pixelwise mean in "
            "Euclidean distance."
        ),
        "target_prototype_training_index": prototype_index,
        "batch_sizes": args.batch_sizes,
        "seeds": args.seeds,
        "steps": args.steps,
        "checkpoints": list(checkpoints),
        "learning_rate": args.learning_rate,
        "budget": args.budget,
        "methods": list(METHODS),
        "mnist_source": os.path.abspath(mnist_path),
        "pairing": (
            "Within a seed, all conditions share the selected corruption rows, zero "
            "initialization, and full sequence of sampled observations; methods also "
            "share each fixed corrupted dataset."
        ),
        "initialization": "All-zero 784-by-10 weight matrix.",
        "evaluation": "Standard unmodified 10,000-image MNIST test set.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
