import matplotlib.pyplot as plt
import numpy as np


def plot_and_input_space():
    """
    Visualizes the Input Space for a Perceptron solving the AND function.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Data for AND function:
    # (0,0) -> 0, (0,1) -> 0, (1,0) -> 0 (Negative class)
    # (1,1) -> 1 (Positive class)
    ax.plot(
        [0, 0, 1],
        [0, 1, 0],
        "s",
        markersize=10,
        markerfacecolor="r",
        markeredgecolor="k",
        label="Negative Class (Target=0)",
    )
    ax.plot(
        [1],
        [1],
        "o",
        markersize=10,
        markerfacecolor="b",
        markeredgecolor="k",
        label="Positive Class (Target=1)",
    )

    # A valid weight vector w = (w0, w1, w2) = (b, w1, w2) is (-0.3, 0.2, 0.2)
    # The decision boundary is w0 + w1*x1 + w2*x2 = 0
    # -0.3 + 0.2*x1 + 0.2*x2 = 0
    # 0.2*x2 = -0.2*x1 + 0.3
    # x2 = -x1 + 1.5
    x1_vals = np.linspace(-0.5, 2.5, 100)
    x2_vals = -x1_vals + 1.5
    ax.plot(
        x1_vals, x2_vals, "k--", linewidth=2, label="Decision Boundary (x₂ = -x₁ + 1.5)"
    )

    # Shade the half-spaces
    ax.fill_between(
        x1_vals, x2_vals, 2.5, color="b", alpha=0.1, label="Positive Region (w·x+b ≥ 0)"
    )
    ax.fill_between(
        x1_vals,
        -0.5,
        x2_vals,
        color="r",
        alpha=0.1,
        label="Negative Region (w·x+b < 0)",
    )

    # Weight vector w = (w1, w2) is normal to the boundary
    ax.arrow(0.75, 0.75, 0.2, 0.2, head_width=0.05, head_length=0.1, fc="k", ec="k")
    ax.text(1.0, 1.0, "w", ha="center", fontsize=12)

    # Annotations and labels
    ax.set_title("Input Space for AND Function", fontsize=16)
    ax.set_xlabel("Input (x₁)", fontsize=12)
    ax.set_ylabel("Input (x₂)", fontsize=12)
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(-0.5, 2)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    plt.savefig("and_input_space.png")
    print("Generated and_input_space.png")


def plot_and_weight_space():
    """
    Visualizes a 2D slice of the Weight Space for the AND function with clearer annotations.
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    # The weight space is 3D (w0, w1, w2). Let's fix w0 to a valid value, e.g., w0 = -0.3
    # The constraints become:
    # 1. w2 < 0.3
    # 2. w1 < 0.3
    # 3. w1 + w2 >= 0.3

    w_vals = np.linspace(-1, 1, 400)

    # Plot constraint boundaries
    ax.axhline(
        0.3, color="c", linestyle="--", linewidth=2, label="Boundary for w₂ < 0.3"
    )
    ax.axvline(
        0.3, color="m", linestyle="--", linewidth=2, label="Boundary for w₁ < 0.3"
    )
    ax.plot(
        w_vals,
        0.3 - w_vals,
        color="g",
        linestyle="--",
        linewidth=2,
        label="Boundary for w₁ + w₂ ≥ 0.3",
    )

    # Add arrows to indicate the valid side of each constraint
    ax.arrow(0.0, 0.3, 0, -0.1, head_width=0.02, head_length=0.02, fc="c", ec="c")
    ax.text(0.0, 0.35, "w₂ < 0.3", color="c", ha="center")

    ax.arrow(0.3, 0.0, -0.1, 0, head_width=0.02, head_length=0.02, fc="m", ec="m")
    ax.text(0.35, 0.0, "w₁ < 0.3", color="m", va="center")

    ax.arrow(0.15, 0.15, 0.05, 0.05, head_width=0.02, head_length=0.02, fc="g", ec="g")
    ax.text(0.1, 0.25, "w₁ + w₂ ≥ 0.3", color="g")

    # Shade the feasible region
    w1_fill = np.linspace(0, 0.3, 200)  # More precise range for the triangle
    ax.fill_between(
        w1_fill, 0.3 - w1_fill, 0.3, color="gray", alpha=0.5, label="Feasible Region"
    )

    # Plot an example valid weight vector
    example_w = (0.2, 0.2)
    ax.plot(
        example_w[0],
        example_w[1],
        "ko",
        markersize=10,
        label=f"Example Solution w=({example_w[0]}, {example_w[1]})",
    )

    # Annotations and labels
    ax.set_title("Weight Space for AND Function (slice at w₀ = -0.3)", fontsize=16)
    ax.set_xlabel("Input Weight (w₁)", fontsize=12)
    ax.set_ylabel("Input Weight (w₂)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="lower left")
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal", adjustable="box")

    plt.savefig("and_weight_space.png")
    print("Generated updated and_weight_space.png")


if __name__ == "__main__":
    plot_and_input_space()
    plot_and_weight_space()
