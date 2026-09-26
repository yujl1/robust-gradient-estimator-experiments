#!/usr/bin/env python3
"""Paired GeoMed/Radius experiments on a balanced Gaussian mixture.

For exactly half of each generated dataset,

    y = 1,  x ~ N(1_d, I_d),

and for the other half,

    y = 0,  x ~ N(-1_d, I_d).

The experiment mirrors the static-corruption MNIST comparison with binary
logistic regression: all corruptions are committed before training, and within
each seed every method/attack pair shares its base dataset, corruption rows,
zero initialization, and complete sampled-index stream.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from reproduce_small_b_accuracy import METHODS, geometric_median, radius_estimator


ATTACKS = (
    "none",
    "sign_flip",
    "target_prototype",
    "target_diverse",
    "target_label",
    "label_flip",
)

ATTACK_LABELS = {
    "none": "Clean",
    "sign_flip": "Sign-flip features",
    "target_prototype": "Target prototype",
    "target_diverse": "Target diverse",
    "target_label": "Target label",
    "label_flip": "Label flip",
}


@dataclass(frozen=True)
class Task:
    attack: str
    epsilon: float
    dimension: int
    n_train: int
    n_test: int
    batch_size: int
    seed: int
    method: str
    steps: int
    learning_rate: float
    budget: int
    checkpoints: tuple[int, ...]


@dataclass(frozen=True)
class Point:
    attack: str
    epsilon: float
    dimension: int
    n_train: int
    n_test: int
    batch_size: int
    num_minibatches: int
    seed: int
    method: str
    iteration: int
    learning_rate: float
    budget: int
    test_accuracy: float
    test_logistic_loss: float
    positive_recall: float
    negative_to_positive_rate: float
    predicted_positive_fraction: float
    weight_norm: float
    weight_alignment_with_ones: float


def make_balanced_gaussian_data(
    seed: int, n_train: int, n_test: int, dimension: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate independent, exactly balanced train and test samples."""
    if n_train % 2 or n_test % 2:
        raise ValueError("n_train and n_test must both be even")
    if dimension < 1:
        raise ValueError("dimension must be positive")

    def generate(n: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
        half = n // 2
        positive = rng.normal(1.0, 1.0, size=(half, dimension)).astype(np.float32)
        negative = rng.normal(-1.0, 1.0, size=(half, dimension)).astype(np.float32)
        features = np.concatenate((positive, negative), axis=0)
        labels = np.concatenate(
            (np.ones(half, dtype=np.float32), np.zeros(half, dtype=np.float32))
        )
        permutation = rng.permutation(n)
        return features[permutation], labels[permutation]

    train_rng = np.random.RandomState(40_000 + 1_009 * dimension + seed)
    test_rng = np.random.RandomState(50_000 + 1_009 * dimension + seed)
    train_features, train_labels = generate(n_train, train_rng)
    test_features, test_labels = generate(n_test, test_rng)
    return train_features, train_labels, test_features, test_labels


def exact_corruption_mask(n: int, epsilon: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(10_000 + seed)
    mask = np.zeros(n, dtype=bool)
    count = int(epsilon * n)
    if count:
        mask[rng.choice(n, count, replace=False)] = True
    return mask


def central_positive_prototype(
    train_features: np.ndarray, train_labels: np.ndarray
) -> tuple[np.ndarray, int]:
    candidate_indices = np.flatnonzero(train_labels == 1.0)
    candidates = train_features[candidate_indices]
    class_mean = candidates.mean(axis=0)
    differences = candidates - class_mean
    distances = np.einsum("ij,ij->i", differences, differences, optimize=True)
    local_index = int(np.argmin(distances))
    return candidates[local_index].copy(), int(candidate_indices[local_index])


def fixed_positive_sources(
    n: int, mask: np.ndarray, train_labels: np.ndarray, seed: int, dimension: int
) -> np.ndarray:
    rng = np.random.RandomState(30_000 + 1_009 * dimension + seed)
    candidates = np.flatnonzero(train_labels == 1.0)
    mapping = np.full(n, -1, dtype=np.int64)
    mapping[mask] = rng.choice(candidates, int(mask.sum()), replace=True)
    return mapping


def apply_corruption(
    attack: str,
    indices: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    prototype: np.ndarray,
    target_sources: np.ndarray,
    train_features: np.ndarray,
) -> None:
    selected = mask[indices]
    if attack == "none" or not np.any(selected):
        return
    if attack == "sign_flip":
        features[selected] *= -1.0
    elif attack == "target_prototype":
        features[selected] = prototype
        labels[selected] = 1.0
    elif attack == "target_diverse":
        features[selected] = train_features[target_sources[indices[selected]]]
        labels[selected] = 1.0
    elif attack == "target_label":
        labels[selected] = 1.0
    elif attack == "label_flip":
        labels[selected] = 1.0 - labels[selected]
    else:
        raise ValueError(f"unknown attack: {attack}")


def evaluate(
    weights: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    task: Task,
    iteration: int,
    num_minibatches: int,
) -> Point:
    logits = test_features @ weights
    predictions = (logits >= 0.0).astype(np.float32)
    positive = test_labels == 1.0
    negative = ~positive
    weight_norm = float(np.linalg.norm(weights))
    if weight_norm == 0.0:
        alignment = 0.0
    else:
        alignment = float(np.sum(weights) / (weight_norm * math.sqrt(task.dimension)))
    losses = np.logaddexp(0.0, logits.astype(np.float64)) - test_labels * logits
    return Point(
        attack=task.attack,
        epsilon=task.epsilon,
        dimension=task.dimension,
        n_train=task.n_train,
        n_test=task.n_test,
        batch_size=task.batch_size,
        num_minibatches=num_minibatches,
        seed=task.seed,
        method=task.method,
        iteration=iteration,
        learning_rate=task.learning_rate,
        budget=task.budget,
        test_accuracy=float(np.mean(predictions == test_labels)),
        test_logistic_loss=float(np.mean(losses)),
        positive_recall=float(np.mean(predictions[positive] == 1.0)),
        negative_to_positive_rate=float(np.mean(predictions[negative] == 1.0)),
        predicted_positive_fraction=float(np.mean(predictions == 1.0)),
        weight_norm=weight_norm,
        weight_alignment_with_ones=alignment,
    )


def train(task: Task) -> list[Point]:
    train_features, train_labels, test_features, test_labels = (
        make_balanced_gaussian_data(
            task.seed, task.n_train, task.n_test, task.dimension
        )
    )
    effective_epsilon = 0.0 if task.attack == "none" else task.epsilon
    mask = exact_corruption_mask(task.n_train, effective_epsilon, task.seed)
    prototype, _ = central_positive_prototype(train_features, train_labels)
    target_sources = fixed_positive_sources(
        task.n_train, mask, train_labels, task.seed, task.dimension
    )

    if task.budget % task.batch_size:
        raise ValueError("budget must be divisible by batch size")
    num_minibatches = task.budget // task.batch_size
    aggregator = geometric_median if task.method == "geomed" else radius_estimator
    rng = np.random.RandomState(20_000 + task.seed)
    weights = np.zeros(task.dimension, dtype=np.float32)
    checkpoints = set(task.checkpoints)
    points: list[Point] = []
    if 0 in checkpoints:
        points.append(
            evaluate(
                weights, test_features, test_labels, task, 0, num_minibatches
            )
        )

    for iteration in range(1, task.steps + 1):
        indices = rng.choice(task.n_train, task.budget, replace=False)
        features = train_features[indices].copy()
        labels = train_labels[indices].copy()
        apply_corruption(
            task.attack,
            indices,
            features,
            labels,
            mask,
            prototype,
            target_sources,
            train_features,
        )
        logits = features @ weights
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        example_gradients = (probabilities - labels)[:, None] * features
        minibatch_gradients = example_gradients.reshape(
            num_minibatches, task.batch_size, task.dimension
        ).mean(axis=1)
        weights -= task.learning_rate * aggregator(minibatch_gradients)

        if iteration in checkpoints:
            points.append(
                evaluate(
                    weights,
                    test_features,
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
            p.dimension,
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
                dimension=int(row["dimension"]),
                n_train=int(row["n_train"]),
                n_test=int(row["n_test"]),
                batch_size=int(row["batch_size"]),
                num_minibatches=int(row["num_minibatches"]),
                seed=int(row["seed"]),
                method=row["method"],
                iteration=int(row["iteration"]),
                learning_rate=float(row["learning_rate"]),
                budget=int(row["budget"]),
                test_accuracy=float(row["test_accuracy"]),
                test_logistic_loss=float(row["test_logistic_loss"]),
                positive_recall=float(row["positive_recall"]),
                negative_to_positive_rate=float(row["negative_to_positive_rate"]),
                predicted_positive_fraction=float(row["predicted_positive_fraction"]),
                weight_norm=float(row["weight_norm"]),
                weight_alignment_with_ones=float(row["weight_alignment_with_ones"]),
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


def summarize(
    points: list[Point], path: Path
) -> list[dict[str, float | int | str]]:
    keyed = {
        (p.dimension, p.attack, p.epsilon, p.seed, p.method, p.iteration): p
        for p in points
    }
    groups = sorted(
        {(p.dimension, p.attack, p.epsilon, p.iteration) for p in points}
    )
    rows: list[dict[str, float | int | str]] = []
    for dimension, attack, epsilon, iteration in groups:
        seeds = sorted(
            {
                p.seed
                for p in points
                if p.dimension == dimension
                and p.attack == attack
                and p.epsilon == epsilon
                and p.iteration == iteration
            }
        )
        pairs = []
        for seed in seeds:
            gm = keyed.get(
                (dimension, attack, epsilon, seed, "geomed", iteration)
            )
            radius = keyed.get(
                (dimension, attack, epsilon, seed, "radius", iteration)
            )
            if gm is not None and radius is not None:
                pairs.append((gm, radius))
        if not pairs:
            continue
        accuracy_gaps = [g.test_accuracy - r.test_accuracy for g, r in pairs]
        loss_gaps = [g.test_logistic_loss - r.test_logistic_loss for g, r in pairs]
        alignment_gaps = [
            g.weight_alignment_with_ones - r.weight_alignment_with_ones
            for g, r in pairs
        ]
        gap_mean, gap_low, gap_high = interval(accuracy_gaps)
        rows.append(
            {
                "dimension": dimension,
                "bayes_accuracy": float(norm.cdf(math.sqrt(dimension))),
                "attack": attack,
                "epsilon": epsilon,
                "batch_size": pairs[0][0].batch_size,
                "num_minibatches": pairs[0][0].num_minibatches,
                "iteration": iteration,
                "paired_seeds": len(pairs),
                "geomed_mean_accuracy": mean(g.test_accuracy for g, _ in pairs),
                "radius_mean_accuracy": mean(r.test_accuracy for _, r in pairs),
                "mean_accuracy_gap_geomed_minus_radius": gap_mean,
                "accuracy_gap_ci95_low": gap_low,
                "accuracy_gap_ci95_high": gap_high,
                "accuracy_gap_pvalue": paired_pvalue(accuracy_gaps),
                "geomed_wins": sum(value > 0.0 for value in accuracy_gaps),
                "radius_wins": sum(value < 0.0 for value in accuracy_gaps),
                "ties": sum(value == 0.0 for value in accuracy_gaps),
                "geomed_mean_logistic_loss": mean(
                    g.test_logistic_loss for g, _ in pairs
                ),
                "radius_mean_logistic_loss": mean(
                    r.test_logistic_loss for _, r in pairs
                ),
                "mean_loss_gap_geomed_minus_radius": mean(loss_gaps),
                "geomed_mean_alignment": mean(
                    g.weight_alignment_with_ones for g, _ in pairs
                ),
                "radius_mean_alignment": mean(
                    r.weight_alignment_with_ones for _, r in pairs
                ),
                "mean_alignment_gap_geomed_minus_radius": mean(alignment_gaps),
                "geomed_mean_positive_recall": mean(
                    g.positive_recall for g, _ in pairs
                ),
                "radius_mean_positive_recall": mean(
                    r.positive_recall for _, r in pairs
                ),
                "geomed_mean_negative_to_positive_rate": mean(
                    g.negative_to_positive_rate for g, _ in pairs
                ),
                "radius_mean_negative_to_positive_rate": mean(
                    r.negative_to_positive_rate for _, r in pairs
                ),
            }
        )
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def summarize_attack_effects(
    points: list[Point], path: Path
) -> list[dict[str, float | int | str]]:
    keyed = {
        (p.dimension, p.attack, p.seed, p.method, p.iteration): p for p in points
    }
    groups = sorted(
        {
            (p.dimension, p.attack, p.epsilon, p.iteration)
            for p in points
            if p.attack != "none"
        }
    )
    rows: list[dict[str, float | int | str]] = []
    for dimension, attack, epsilon, iteration in groups:
        per_seed = []
        for seed in sorted({p.seed for p in points}):
            values = (
                keyed.get((dimension, attack, seed, "geomed", iteration)),
                keyed.get((dimension, attack, seed, "radius", iteration)),
                keyed.get((dimension, "none", seed, "geomed", iteration)),
                keyed.get((dimension, "none", seed, "radius", iteration)),
            )
            if all(value is not None for value in values):
                per_seed.append(values)
        if not per_seed:
            continue
        gm_changes = [a.test_accuracy - c.test_accuracy for a, _, c, _ in per_seed]
        radius_changes = [a.test_accuracy - c.test_accuracy for _, a, _, c in per_seed]
        differences = [g - r for g, r in zip(gm_changes, radius_changes)]
        diff_mean, diff_low, diff_high = interval(differences)
        rows.append(
            {
                "dimension": dimension,
                "attack": attack,
                "epsilon": epsilon,
                "iteration": iteration,
                "paired_seeds": len(per_seed),
                "geomed_mean_accuracy_change_attack_minus_clean": mean(gm_changes),
                "radius_mean_accuracy_change_attack_minus_clean": mean(radius_changes),
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
    dimensions = sorted({p.dimension for p in points})

    for dimension in dimensions:
        ncols = 3
        nrows = math.ceil(len(attacks) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(12.0, 3.25 * nrows), squeeze=False
        )
        for axis, attack in zip(axes.flat, attacks):
            selected = [
                p for p in points if p.dimension == dimension and p.attack == attack
            ]
            iterations = sorted({p.iteration for p in selected})
            for method in METHODS:
                centers, lows, highs = [], [], []
                for iteration in iterations:
                    values = [
                        p.test_accuracy
                        for p in selected
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
                    markersize=2.6,
                    label=labels[method],
                )
                axis.fill_between(
                    iterations, lows, highs, color=colors[method], alpha=0.14
                )
            bayes = float(norm.cdf(math.sqrt(dimension)))
            axis.axhline(
                bayes, color="#111827", linestyle=":", linewidth=1.1,
                label="Bayes"
            )
            axis.set_title(ATTACK_LABELS[attack])
            axis.set_xlabel("SGD updates")
            axis.set_ylabel("Clean test accuracy")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=7.5, loc="lower right")
        for axis in axes.flat[len(attacks) :]:
            axis.axis("off")
        fig.suptitle(
            f"Balanced Gaussian mixture: d={dimension}, b={points[0].batch_size}"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"accuracy_trajectories_d{dimension}.png", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(12.0, 3.25 * nrows), squeeze=False
        )
        for axis, attack in zip(axes.flat, attacks):
            selected = [
                p for p in points if p.dimension == dimension and p.attack == attack
            ]
            iterations = sorted({p.iteration for p in selected})
            for method in METHODS:
                centers, lows, highs = [], [], []
                for iteration in iterations:
                    values = [
                        p.test_logistic_loss
                        for p in selected
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
                    markersize=2.6,
                    label=labels[method],
                )
                axis.fill_between(
                    iterations, lows, highs, color=colors[method], alpha=0.14
                )
            axis.set_title(ATTACK_LABELS[attack])
            axis.set_xlabel("SGD updates")
            axis.set_ylabel("Clean logistic loss")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=7.5)
        for axis in axes.flat[len(attacks) :]:
            axis.axis("off")
        fig.suptitle(
            f"Gaussian-mixture test loss: d={dimension}, b={points[0].batch_size}"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"loss_trajectories_d{dimension}.png", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(12.0, 3.25 * nrows), squeeze=False
        )
        for axis, attack in zip(axes.flat, attacks):
            selected = [
                p for p in points if p.dimension == dimension and p.attack == attack
            ]
            iterations = sorted({p.iteration for p in selected})
            for method in METHODS:
                centers, lows, highs = [], [], []
                for iteration in iterations:
                    values = [
                        p.weight_alignment_with_ones
                        for p in selected
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
                    markersize=2.6,
                    label=labels[method],
                )
                axis.fill_between(
                    iterations, lows, highs, color=colors[method], alpha=0.14
                )
            axis.axhline(1.0, color="#111827", linestyle=":", linewidth=1.1)
            axis.set_title(ATTACK_LABELS[attack])
            axis.set_xlabel("SGD updates")
            axis.set_ylabel(r"Cosine alignment with $\mathbf{1}_d$")
            axis.set_ylim(-0.05, 1.05)
            axis.grid(alpha=0.22)
            axis.legend(fontsize=7.5, loc="lower right")
        for axis in axes.flat[len(attacks) :]:
            axis.axis("off")
        fig.suptitle(
            f"Gaussian-mixture direction alignment: d={dimension}, "
            f"b={points[0].batch_size}"
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"alignment_trajectories_d{dimension}.png", dpi=220)
        plt.close(fig)

    endpoints = []
    for dimension in dimensions:
        for attack in attacks:
            rows = [
                row
                for row in summary
                if int(row["dimension"]) == dimension and row["attack"] == attack
            ]
            if rows:
                endpoints.append(max(rows, key=lambda row: int(row["iteration"])))
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5.0 * len(dimensions), 4.1), squeeze=False)
    for axis, dimension in zip(axes.flat, dimensions):
        rows = [row for row in endpoints if int(row["dimension"]) == dimension]
        positions = np.arange(len(rows))
        gaps = 100.0 * np.array(
            [float(row["mean_accuracy_gap_geomed_minus_radius"]) for row in rows]
        )
        lows = 100.0 * np.array([float(row["accuracy_gap_ci95_low"]) for row in rows])
        highs = 100.0 * np.array([float(row["accuracy_gap_ci95_high"]) for row in rows])
        axis.errorbar(
            positions,
            gaps,
            yerr=np.vstack((gaps - lows, highs - gaps)),
            fmt="o",
            color="#6d28d9",
            capsize=3,
        )
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.set_xticks(
            positions,
            [ATTACK_LABELS[str(row["attack"])] for row in rows],
            rotation=25,
            ha="right",
        )
        axis.set_ylabel("GeoMed - Radius accuracy (points)")
        axis.set_title(f"d={dimension}")
        axis.grid(axis="y", alpha=0.22)
    fig.suptitle("Endpoint paired accuracy gaps across dimensions")
    fig.tight_layout()
    fig.savefig(output_dir / "endpoint_accuracy_gaps.png", dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks", nargs="+", choices=ATTACKS, default=list(ATTACKS))
    parser.add_argument("--dimensions", nargs="+", type=int, default=[2, 10, 100])
    parser.add_argument("--n-train", type=int, default=60_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--budget", type=int, default=640)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        type=int,
        default=[
            0, 1, 2, 5, 10, 25, 50, 100, 200, 300, 500, 750, 1000,
            1500, 2000, 3000,
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/gaussian_static_corruption_suite"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")
    if args.n_train % 2 or args.n_test % 2:
        raise ValueError("training and test sizes must be even")
    if args.budget > args.n_train:
        raise ValueError("budget cannot exceed n_train when sampling without replacement")
    if args.budget % args.batch_size:
        raise ValueError("batch size must divide budget")
    if any(dimension < 1 for dimension in args.dimensions):
        raise ValueError("all dimensions must be positive")
    checkpoints = tuple(sorted(set(args.checkpoints)))
    if not checkpoints or checkpoints[0] < 0 or checkpoints[-1] > args.steps:
        raise ValueError("checkpoints must lie between zero and --steps")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw_trajectories.csv"
    existing = read_points(raw_path)
    expected_iterations = set(checkpoints)
    completed = set()
    for dimension in args.dimensions:
        for attack in args.attacks:
            epsilon = 0.0 if attack == "none" else args.epsilon
            for seed in args.seeds:
                for method in METHODS:
                    observed = {
                        p.iteration
                        for p in existing
                        if p.dimension == dimension
                        and p.n_train == args.n_train
                        and p.n_test == args.n_test
                        and p.attack == attack
                        and p.epsilon == epsilon
                        and p.batch_size == args.batch_size
                        and p.seed == seed
                        and p.method == method
                        and p.learning_rate == args.learning_rate
                        and p.budget == args.budget
                    }
                    if expected_iterations <= observed:
                        completed.add((dimension, attack, seed, method))

    tasks = [
        Task(
            attack=attack,
            epsilon=0.0 if attack == "none" else args.epsilon,
            dimension=dimension,
            n_train=args.n_train,
            n_test=args.n_test,
            batch_size=args.batch_size,
            seed=seed,
            method=method,
            steps=args.steps,
            learning_rate=args.learning_rate,
            budget=args.budget,
            checkpoints=checkpoints,
        )
        for dimension in args.dimensions
        for attack in args.attacks
        for seed in args.seeds
        for method in METHODS
        if (dimension, attack, seed, method) not in completed
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
                        p.dimension == task.dimension
                        and p.n_train == task.n_train
                        and p.n_test == task.n_test
                        and p.attack == task.attack
                        and p.epsilon == task.epsilon
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
                    f"[{index}/{len(tasks)}] d={task.dimension} {task.attack} "
                    f"seed={task.seed} {task.method} "
                    f"accuracy={endpoint.test_accuracy:.5f} "
                    f"loss={endpoint.test_logistic_loss:.5f} "
                    f"alignment={endpoint.weight_alignment_with_ones:.5f}",
                    flush=True,
                )

    allowed_epsilons = {
        attack: 0.0 if attack == "none" else args.epsilon for attack in args.attacks
    }
    selected = [
        p
        for p in points
        if p.dimension in args.dimensions
        and p.n_train == args.n_train
        and p.n_test == args.n_test
        and p.attack in args.attacks
        and p.epsilon == allowed_epsilons[p.attack]
        and p.batch_size == args.batch_size
        and p.seed in args.seeds
        and p.method in METHODS
        and p.iteration in expected_iterations
        and p.learning_rate == args.learning_rate
        and p.budget == args.budget
    ]
    summary = summarize(selected, args.output_dir / "trajectory_summary.csv")
    summarize_attack_effects(selected, args.output_dir / "attack_effect_summary.csv")
    plot(selected, summary, args.output_dir)

    prototype_indices = {}
    class_counts = {}
    for dimension in args.dimensions:
        per_dimension = {}
        per_dimension_counts = {}
        for seed in args.seeds:
            train_features, train_labels, _, _ = make_balanced_gaussian_data(
                seed, args.n_train, args.n_test, dimension
            )
            _, prototype_index = central_positive_prototype(
                train_features, train_labels
            )
            per_dimension[str(seed)] = prototype_index
            per_dimension_counts[str(seed)] = {
                "positive": int(np.sum(train_labels == 1.0)),
                "negative": int(np.sum(train_labels == 0.0)),
            }
        prototype_indices[str(dimension)] = per_dimension
        class_counts[str(dimension)] = per_dimension_counts

    metadata = {
        "question": (
            "Does the finite-horizon GeoMed-versus-Radius distinction at small "
            "minibatch size persist for balanced Gaussian-mixture data under "
            "several fixed corruptions?"
        ),
        "data_generating_process": {
            "positive_half": "y=1 and x ~ N(1_d, I_d)",
            "negative_half": "y=0 and x ~ N(-1_d, I_d)",
            "balance": "Exactly half of train and test observations in each class.",
        },
        "bayes_accuracy_by_dimension": {
            str(dimension): float(norm.cdf(math.sqrt(dimension)))
            for dimension in args.dimensions
        },
        "model": "Binary logistic regression without an intercept.",
        "attacks": {
            "none": "Unmodified training set; epsilon recorded as zero.",
            "sign_flip": "For selected rows, replace x by -x and retain y.",
            "target_prototype": (
                "Replace every selected row by the same central positive-class "
                "training point and label 1."
            ),
            "target_diverse": (
                "Replace each selected row once by a seeded positive-class "
                "training point and label 1."
            ),
            "target_label": "Retain x and replace the selected label by 1.",
            "label_flip": "Retain x and replace the selected label y by 1-y.",
        },
        "obliviousness": (
            "All corruption rows and replacements are fixed before training; the "
            "adversary never observes iterates or sampled minibatches."
        ),
        "included_attacks": args.attacks,
        "dimensions": args.dimensions,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "class_counts": class_counts,
        "epsilon": args.epsilon,
        "corrupted_training_rows": int(args.epsilon * args.n_train),
        "target_class": 1,
        "target_prototype_rule": (
            "Positive-class training point nearest the positive-class sample mean "
            "in Euclidean distance."
        ),
        "target_prototype_training_indices": prototype_indices,
        "batch_size": args.batch_size,
        "seeds": args.seeds,
        "steps": args.steps,
        "checkpoints": list(checkpoints),
        "learning_rate": args.learning_rate,
        "budget": args.budget,
        "methods": list(METHODS),
        "pairing": (
            "Within a seed and dimension, all conditions share the generated base "
            "data, selected corruption rows, zero initialization, and full sequence "
            "of sampled training indices. Methods share each fixed corrupted dataset."
        ),
        "initialization": "All-zero d-vector.",
        "evaluation": "Independent clean balanced test sample for each seed.",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "working_directory": os.path.abspath(Path.cwd()),
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
