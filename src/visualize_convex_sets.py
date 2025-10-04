import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull


def plot_convex_example():
    """Plot a simple convex set (triangle) with line segments between points."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Define a convex set (triangle)
    points = np.array([[1, 1], [4, 2], [2.5, 4]])

    # Plot the convex hull
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], "b-", linewidth=2)

    # Fill the polygon
    polygon = Polygon(points, alpha=0.3, facecolor="blue", edgecolor="blue")
    ax.add_patch(polygon)

    # Plot points
    ax.plot(points[:, 0], points[:, 1], "ro", markersize=10, label="Points")

    # Draw line segments between all pairs of points
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            ax.plot(
                [points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                "g--",
                linewidth=1.5,
                alpha=0.7,
            )

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(
        "Convex Set\n(All line segments lie within the set)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("convex_set_example.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_nonconvex_example():
    """Plot a non-convex set (crescent shape)."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Create a crescent shape (non-convex)
    theta = np.linspace(0, np.pi, 50)
    x_outer = 3 * np.cos(theta) + 3
    y_outer = 3 * np.sin(theta) + 2.5

    x_inner = 2 * np.cos(theta) + 3.5
    y_inner = 2 * np.sin(theta) + 2.5

    x_crescent = np.concatenate([x_outer, x_inner[::-1]])
    y_crescent = np.concatenate([y_outer, y_inner[::-1]])

    # Plot the crescent
    polygon = Polygon(
        np.column_stack([x_crescent, y_crescent]),
        alpha=0.3,
        facecolor="red",
        edgecolor="red",
        linewidth=2,
    )
    ax.add_patch(polygon)

    # Select two points where line segment goes outside
    p1 = np.array([1.5, 2.5])
    p2 = np.array([4.5, 2.5])

    ax.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        "b--",
        linewidth=2,
        label="Line segment outside set",
    )
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "go", markersize=10, label="Points in set")

    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(
        "Non-Convex Set\n(Line segment can lie outside the set)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("nonconvex_set_example.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_xor_convexity_problem():
    """Illustrate why XOR is not linearly separable using convexity."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # XOR data points
    pos_points = np.array([[0, 1], [1, 0]])
    neg_points = np.array([[0, 0], [1, 1]])

    # Plot positive examples
    ax.plot(
        pos_points[:, 0],
        pos_points[:, 1],
        "b+",
        markersize=20,
        markeredgewidth=3,
        label="Positive class (+1)",
    )

    # Plot negative examples
    ax.plot(
        neg_points[:, 0],
        neg_points[:, 1],
        "ro",
        markersize=15,
        markerfacecolor="red",
        label="Negative class (-1)",
    )

    # Draw line segment between positive examples
    ax.plot(
        [pos_points[0, 0], pos_points[1, 0]],
        [pos_points[0, 1], pos_points[1, 1]],
        "b--",
        linewidth=2,
        alpha=0.7,
        label="Line between positive examples",
    )

    # Draw line segment between negative examples
    ax.plot(
        [neg_points[0, 0], neg_points[1, 0]],
        [neg_points[0, 1], neg_points[1, 1]],
        "r--",
        linewidth=2,
        alpha=0.7,
        label="Line between negative examples",
    )

    # Mark intersection point
    intersection = np.array([0.5, 0.5])
    ax.plot(
        intersection[0],
        intersection[1],
        "k*",
        markersize=20,
        label="Intersection (0.5, 0.5)",
    )

    # Add annotations
    ax.annotate("(0,1)", xy=(0, 1), xytext=(-0.15, 1.1), fontsize=11)
    ax.annotate("(1,0)", xy=(1, 0), xytext=(1.05, -0.1), fontsize=11)
    ax.annotate("(0,0)", xy=(0, 0), xytext=(-0.15, -0.15), fontsize=11)
    ax.annotate("(1,1)", xy=(1, 1), xytext=(1.05, 1.05), fontsize=11)
    ax.annotate(
        "Contradiction!\nMust be both\n+1 and -1",
        xy=(0.5, 0.5),
        xytext=(0.6, 0.7),
        fontsize=10,
        ha="left",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"),
    )

    ax.set_xlim(-0.3, 1.3)
    ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(
        "XOR Problem: Why It's Not Linearly Separable\n"
        + "Convexity argument shows contradiction at (0.5, 0.5)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("xor_convexity_problem.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_halfspace_convexity():
    """Illustrate that half-spaces are convex."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Define a decision boundary
    x = np.linspace(-1, 5, 100)
    y_boundary = 0.5 * x + 1

    # Fill positive half-space
    ax.fill_between(
        x, y_boundary, 5, alpha=0.3, color="blue", label="Positive half-space"
    )

    # Plot decision boundary
    ax.plot(x, y_boundary, "k-", linewidth=2, label="Decision boundary")

    # Select random points in the positive half-space
    np.random.seed(42)
    points = np.array([[1, 3], [3, 4], [2, 2.5], [4, 3.5]])

    # Plot points
    ax.plot(
        points[:, 0], points[:, 1], "ro", markersize=10, label="Points in half-space"
    )

    # Draw line segments between some pairs to show convexity
    for i in range(len(points) - 1):
        ax.plot(
            [points[i, 0], points[i + 1, 0]],
            [points[i, 1], points[i + 1, 1]],
            "g--",
            linewidth=1.5,
            alpha=0.7,
        )

    # Add text
    ax.text(
        3.5,
        4.5,
        "All line segments\nstay in half-space",
        fontsize=11,
        ha="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax.set_xlim(-1, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title("Half-Space is Convex", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig("halfspace_convexity.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_weighted_average():
    """Illustrate weighted average of points in a convex set."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    # Define a convex set (quadrilateral)
    points = np.array([[1, 1], [4, 1.5], [4, 4], [1.5, 3.5]])

    # Plot the convex hull
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        ax.plot(points[simplex, 0], points[simplex, 1], "b-", linewidth=2)

    # Fill the polygon
    polygon = Polygon(points, alpha=0.2, facecolor="blue", edgecolor="blue")
    ax.add_patch(polygon)

    # Plot original points
    ax.plot(points[:, 0], points[:, 1], "ro", markersize=10, label="Original points")

    # Compute weighted averages
    weights1 = np.array([0.25, 0.25, 0.25, 0.25])
    avg1 = np.dot(weights1, points)

    weights2 = np.array([0.5, 0.3, 0.1, 0.1])
    avg2 = np.dot(weights2, points)

    weights3 = np.array([0.1, 0.1, 0.6, 0.2])
    avg3 = np.dot(weights3, points)

    # Plot weighted averages
    ax.plot(avg1[0], avg1[1], "g*", markersize=15, label="Weighted average 1")
    ax.plot(avg2[0], avg2[1], "m*", markersize=15, label="Weighted average 2")
    ax.plot(avg3[0], avg3[1], "c*", markersize=15, label="Weighted average 3")

    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(
        "Weighted Averages Lie Within Convex Set", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("weighted_average_convex.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Generate all convex set visualizations."""
    print("Generating convex set visualizations...")

    print("1. Plotting convex set example...")
    plot_convex_example()

    print("2. Plotting non-convex set example...")
    plot_nonconvex_example()

    print("3. Plotting XOR convexity problem...")
    plot_xor_convexity_problem()

    print("4. Plotting half-space convexity...")
    plot_halfspace_convexity()

    print("5. Plotting weighted average demonstration...")
    plot_weighted_average()

    print("\nAll visualizations saved successfully!")


if __name__ == "__main__":
    main()
