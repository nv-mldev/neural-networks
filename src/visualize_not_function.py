import matplotlib.pyplot as plt
import numpy as np


def plot_input_space():
    """
    Visualizes the Input Space for a Perceptron solving the NOT function.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data for NOT function:
    # Input 0 (x₁=0) -> Output 1 (positive class)
    # Input 1 (x₁=1) -> Output 0 (negative class)
    # Both points have x₀=1 (bias component)
    ax.plot(
        [1],
        [0],
        "o",
        markersize=10,
        markerfacecolor="b",
        markeredgecolor="k",
        label="Positive Class (x₁=0, Target=1)",
    )
    ax.plot(
        [1],
        [1],
        "s",
        markersize=10,
        markerfacecolor="r",
        markeredgecolor="k",
        label="Negative Class (x₁=1, Target=0)",
    )

    # Decision boundary as sloped line through origin: w₀*x₀ + w₁*x₁ = 0
    # This gives x₁ = -(w₀/w₁)*x₀. Let's use w₀=1, w₁=-2
    # so slope = -1/(-2) = 0.5. Decision line: x₁ = 0.5*x₀
    w0, w1 = 1, -2
    x0_range = np.linspace(-0.5, 1.5, 100)
    x1_boundary = -(w0 / w1) * x0_range  # x₁ = 0.5*x₀
    ax.plot(
        x0_range,
        x1_boundary,
        color="k",
        linestyle="--",
        linewidth=2,
        label="Decision Boundary (x₁ = 0.5·x₀)",
    )

    # Shade the half-spaces
    x0_fill = np.linspace(-0.5, 1.5, 100)
    x1_fill_boundary = -(w0 / w1) * x0_fill

    ax.fill_between(
        x0_fill,
        x1_fill_boundary,
        2,
        color="r",
        alpha=0.1,
        label="Negative Region (w·x < 0)",
    )
    ax.fill_between(
        x0_fill,
        -1.2,
        x1_fill_boundary,
        color="b",
        alpha=0.1,
        label="Positive Region (w·x ≥ 0)",
    )

    # Highlight a reference point on the decision boundary and
    # draw the position/normal vectors
    boundary_point = np.array([1.0, -(w0 / w1) * 1.0])

    # Position vector from origin to point on boundary
    ax.arrow(
        0,
        0,
        boundary_point[0],
        boundary_point[1],
        head_width=0.04,
        head_length=0.08,
        fc="#2e7d32",
        ec="#2e7d32",
        linestyle="-",
        linewidth=2.5,
        length_includes_head=True,
        label="Position Vector r_P",
    )
    # Mark the point where position vector ends
    ax.scatter(
        boundary_point[0],
        boundary_point[1],
        color="#2e7d32",
        s=50,
        marker="o",
        zorder=5,
        edgecolor="white",
        linewidth=1.5,
    )
    ax.text(
        boundary_point[0] * 0.6,
        boundary_point[1] * 0.6 + 0.1,
        "r_P",
        fontsize=11,
        color="#2e7d32",
        fontweight="bold",
    )

    # Unit normal vector from origin
    w_magnitude = np.sqrt(w0**2 + w1**2)
    unit_normal = np.array([w0, w1]) / w_magnitude
    unit_scale = 0.6
    ax.arrow(
        0,
        0,
        unit_scale * unit_normal[0],
        unit_scale * unit_normal[1],
        head_width=0.04,
        head_length=0.08,
        fc="red",
        ec="red",
        linestyle="-",
        linewidth=2.5,
        length_includes_head=True,
        label="Unit Normal n̂",
    )
    ax.text(
        unit_scale * unit_normal[0] * 0.7,
        unit_scale * unit_normal[1] * 0.7 - 0.1,
        "n̂",
        fontsize=11,
        color="red",
        fontweight="bold",
    )

    # Normal vector from boundary point
    normal_scale = 0.35
    normal_vector = normal_scale * np.array([w0, w1])
    ax.arrow(
        boundary_point[0],
        boundary_point[1],
        normal_vector[0],
        normal_vector[1],
        head_width=0.05,
        head_length=0.1,
        fc="k",
        ec="k",
        length_includes_head=True,
        label="Normal Vector w",
    )

    # Set up proper coordinate system with axes through origin
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-1.2, 2)

    # Draw x and y axes through origin
    ax.axhline(y=0, color="black", linewidth=1.5, alpha=0.8)
    ax.axvline(x=0, color="black", linewidth=1.5, alpha=0.8)

    # Set ticks and labels for both axes
    ax.set_xticks([-0.5, 0, 0.5, 1, 1.5])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1, 1.5, 2])

    # Add grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.2)

    # Annotations and labels
    ax.set_title("Input Space for NOT Function", fontsize=16, pad=20)
    ax.set_xlabel("Bias Component (x₀)", fontsize=12)
    ax.set_ylabel("Input Value (x₁)", fontsize=12)

    # Add axis labels at the ends of axes
    ax.text(1.4, 0.05, "x₀", fontsize=14, ha="center", va="bottom")
    ax.text(0.05, 1.9, "x₁", fontsize=14, ha="left", va="center")

    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9)

    plt.savefig("not_input_space.png")
    print("Generated not_input_space.png")


def plot_weight_space():
    """
    Visualizes the Weight Space for a Perceptron solving the NOT function.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    w0 = np.linspace(-2, 2, 400)

    # With sloped boundary through origin, we need different constraints
    # Point (1,0): w₀*1 + w₁*0 = w₀ > 0 (positive class)
    # Point (1,1): w₀*1 + w₁*1 = w₀ + w₁ < 0 (negative class)
    ax.axvline(
        x=0,
        color="g",
        linestyle="-",
        linewidth=2,
        label="Constraint 1: w₀ > 0",
    )

    # Constraint 2: w₀ + w₁ < 0 => w₁ < -w₀
    w1_constraint2 = -w0
    ax.plot(
        w0,
        w1_constraint2,
        color="m",
        linestyle="-",
        linewidth=2,
        label="Constraint 2: w₀ + w₁ < 0",
    )

    # Shade the feasible region (solution space)
    # We need w₀ > 0 and w₁ < -w₀
    mask = w0 >= 0
    ax.fill_between(
        w0,
        -2,
        w1_constraint2,
        where=mask.tolist(),
        color="gray",
        alpha=0.5,
        label="Feasible Region (Solution Space)",
        interpolate=True,
    )

    # Plot an example valid weight vector
    example_w = (1, -2)
    ax.plot(
        example_w[0],
        example_w[1],
        "ko",
        markersize=10,
        label=f"Example Solution w=({example_w[0]}, {example_w[1]})",
    )

    # Set up coordinate system
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal", adjustable="box")

    # Draw x and y axes through origin with proper styling
    ax.axhline(0, color="black", linewidth=1.5, alpha=0.8)
    ax.axvline(0, color="black", linewidth=1.5, alpha=0.8)

    # Set ticks and labels
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])

    # Add grid
    ax.grid(True, which="major", linestyle="-", linewidth=0.5, alpha=0.3)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.3, alpha=0.2)

    # Annotations and labels
    ax.set_title("Weight Space for NOT Function", fontsize=16, pad=20)
    ax.set_xlabel("Bias Weight (w₀)", fontsize=12)
    ax.set_ylabel("Input Weight (w₁)", fontsize=12)

    # Add axis labels at the ends of axes
    ax.text(1.9, 0.1, "w₀", fontsize=14, ha="center", va="bottom")
    ax.text(0.1, 1.9, "w₁", fontsize=14, ha="left", va="center")

    ax.legend()

    plt.savefig("not_weight_space.png")
    print("Generated not_weight_space.png")


if __name__ == "__main__":
    plot_input_space()
    plot_weight_space()
