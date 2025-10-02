"""Visualize the decision boundary of a 2D linear classifier.

Two scattered clusters are sampled on either side of the decision
boundary, leaving a low-density strip around the separating line so the
margin is easy to see.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class LinearClassifier:
    """Simple 2D linear classifier of the form sign(w · x + b)."""

    weights: np.ndarray
    bias: float

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) >= 0).astype(int)


def _sample_class_points(
    classifier: "LinearClassifier",
    center: np.ndarray,
    label: int,
    n_points: int,
    rng: np.random.Generator,
    spread: float,
    buffer: float,
) -> np.ndarray:
    """Draw class samples, rejecting those that fall too close to the boundary."""

    weights_norm = np.linalg.norm(classifier.weights)
    if weights_norm == 0:
        raise ValueError("Weights must be non-zero to define a decision boundary.")

    collected: list[np.ndarray] = []
    target_sign = -1 if label == 0 else 1

    while sum(chunk.shape[0] for chunk in collected) < n_points:
        batch_size = max(8, (n_points - sum(chunk.shape[0] for chunk in collected)) * 2)
        candidates = center + rng.normal(scale=spread, size=(batch_size, 2))
        scores = classifier.decision_function(candidates)
        distances = np.abs(scores) / weights_norm

        keep_mask = (scores * target_sign >= 0) & (distances >= buffer)
        kept = candidates[keep_mask]
        if kept.size:
            collected.append(kept)

    stacked = np.vstack(collected)
    return stacked[:n_points]


def generate_samples(
    classifier: LinearClassifier,
    n_samples: int,
    seed: int,
    margin: float,
    spread: float,
    buffer: float,
) -> tuple[np.ndarray, np.ndarray]:
    if np.allclose(classifier.weights, 0.0):
        raise ValueError("Weights must be non-zero to define a decision boundary.")

    n_samples = max(n_samples, 2)
    rng = np.random.default_rng(seed)

    weights = classifier.weights
    weights_norm = np.linalg.norm(weights)
    normal = weights / weights_norm

    boundary_point = -classifier.bias / (weights_norm**2) * weights

    n_class0 = n_samples // 2
    n_class1 = n_samples - n_class0

    center0 = boundary_point - margin * normal
    center1 = boundary_point + margin * normal

    class0 = _sample_class_points(
        classifier,
        center=center0,
        label=0,
        n_points=n_class0,
        rng=rng,
        spread=spread,
        buffer=buffer,
    )
    class1 = _sample_class_points(
        classifier,
        center=center1,
        label=1,
        n_points=n_class1,
        rng=rng,
        spread=spread,
        buffer=buffer,
    )

    labels = np.concatenate((np.zeros(n_class0, dtype=int), np.ones(n_class1, dtype=int)))
    samples = np.vstack((class0, class1))

    return samples, labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the decision boundary of a linear classifier in 2D",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=2,
        default=(1.0, -1.0),
        metavar=("W1", "W2"),
        help="Classifier weights (w1, w2).",
    )
    parser.add_argument(
        "--bias",
        type=float,
        default=0.0,
        help="Classifier bias term.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of points to sample (split evenly across classes).",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1.5,
        help="Offset from the decision boundary to each class centre.",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=0.75,
        help="Standard deviation of the (isotropic) class spread.",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.6,
        help="Minimum distance each point must keep from the decision boundary.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="linear_classifier_decision_boundary.png",
        help="Filename to save inside the figures directory (PNG extension forced).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving the PNG.",
    )
    return parser.parse_args()


def compute_decision_boundary(
    weights: np.ndarray, bias: float, x_values: np.ndarray
) -> np.ndarray:
    w1, w2 = weights
    if np.allclose(weights, 0.0):
        raise ValueError("Weights must be non-zero to define a decision boundary.")
    if np.isclose(w2, 0.0):
        raise ValueError("Cannot compute y-values when w2 is zero; rotate the weights.")
    return -(w1 / w2) * x_values - bias / w2


def plot_decision_boundary(
    classifier: LinearClassifier,
    X: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    show_plot: bool,
) -> None:
    predictions = classifier.predict(X)

    x_min, x_max = X[:, 0].min() - 0.75, X[:, 0].max() + 0.75
    y_min, y_max = X[:, 1].min() - 0.75, X[:, 1].max() + 0.75

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(grid_points).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx, yy, Z, levels=2, alpha=0.2, cmap="coolwarm")

    boundary_x = np.array([x_min, x_max])
    try:
        boundary_y = compute_decision_boundary(classifier.weights, classifier.bias, boundary_x)
        ax.plot(boundary_x, boundary_y, color="black", linewidth=2, label="Decision boundary")
    except ValueError:
        x_vertical = -classifier.bias / classifier.weights[0]
        ax.axvline(x=x_vertical, color="black", linewidth=2, label="Decision boundary")

    ax.scatter(
        X[labels == 0, 0],
        X[labels == 0, 1],
        color="tab:blue",
        label="Class 0",
        edgecolor="k",
        alpha=0.85,
    )
    ax.scatter(
        X[labels == 1, 0],
        X[labels == 1, 1],
        color="tab:red",
        label="Class 1",
        edgecolor="k",
        alpha=0.85,
    )

    misclassified = labels != predictions
    if misclassified.any():
        ax.scatter(
            X[misclassified, 0],
            X[misclassified, 1],
            facecolors="none",
            edgecolors="yellow",
            linewidths=1.5,
            marker="o",
            label="Misclassified",
        )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Linear Classifier Decision Boundary")
    ax.legend(loc="upper right")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")

    fig.tight_layout()

    fig.savefig(output_path, dpi=300)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    weights = np.array(args.weights, dtype=float)
    classifier = LinearClassifier(weights=weights, bias=args.bias)
    X, labels = generate_samples(
        classifier=classifier,
        n_samples=args.samples,
        seed=args.seed,
        margin=args.margin,
        spread=args.spread,
        buffer=args.buffer,
    )
    output_path = Path("figures") / args.output
    if output_path.suffix.lower() != ".png":
        output_path = output_path.with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plot_decision_boundary(
        classifier=classifier,
        X=X,
        labels=labels,
        output_path=output_path,
        show_plot=args.show,
    )


if __name__ == "__main__":
    main()
