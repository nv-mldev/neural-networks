import matplotlib.pyplot as plt
import numpy as np


def plot_input_space():
    """
    Visualizes the Input Space for a Perceptron solving the NOT function.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Data for NOT function:
    # Input 0 (x1=0) -> Output 1 (positive class)
    # Input 1 (x1=1) -> Output 0 (negative class)
    ax.plot(
        [0],
        [1],
        "o",
        markersize=10,
        markerfacecolor="b",
        markeredgecolor="k",
        label="Positive Class (Input=0, Target=1)",
    )
    ax.plot(
        [1],
        [1],
        "s",
        markersize=10,
        markerfacecolor="r",
        markeredgecolor="k",
        label="Negative Class (Input=1, Target=0)",
    )

    # A valid weight vector w = (w0, w1) = (b, w1) is (0.5, -1)
    # The decision boundary is w0*x0 + w1*x1 = 0. With x0=1, this is w0 + w1*x1 = 0
    # So, x1 = -w0 / w1. For our weights, x1 = -0.5 / -1 = 0.5
    decision_boundary_x = 0.5
    ax.axvline(
        x=decision_boundary_x,
        color="k",
        linestyle="--",
        linewidth=2,
        label="Decision Boundary (x₁ = 0.5)",
    )

    # Shade the half-spaces
    ax.fill_betweenx(
        np.linspace(0, 2, 100),
        decision_boundary_x,
        2,
        color="b",
        alpha=0.1,
        label="Positive Region (w·x+b ≥ 0)",
    )
    ax.fill_betweenx(
        np.linspace(0, 2, 100),
        0,
        decision_boundary_x,
        color="r",
        alpha=0.1,
        label="Negative Region (w·x+b < 0)",
    )

    # Annotations and labels
    ax.set_title("Input Space for NOT Function", fontsize=16)
    ax.set_xlabel("Input Value (x₁)", fontsize=12)
    ax.set_ylabel("Dummy Input for Bias (x₀)", fontsize=12)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 2)
    ax.set_yticks([1])
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend(loc="upper right")

    # Add weight vector arrow
    # w = (w1) = (-1). It's a 1D vector. In our 2D plot, the normal to the boundary is (-1, 0)
    # But the weight vector w is (w0, w1) = (0.5, -1). The direction normal to the boundary is (w1)
    # The vector w = (w1) = -1 points towards the negative region. We plot the normal vector.
    ax.arrow(
        decision_boundary_x,
        1.5,
        -0.5,
        0,
        head_width=0.05,
        head_length=0.1,
        fc="k",
        ec="k",
        label="Weight Vector w Normal",
    )
    ax.text(0, 1.6, "w (normal to boundary)", ha="center")

    plt.savefig("not_input_space.png")
    print("Generated not_input_space.png")


def plot_weight_space():
    """
    Visualizes the Weight Space for a Perceptron solving the NOT function.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    w0 = np.linspace(-2, 2, 400)

    # Constraint 1: from x=(1,0), t=1 => w0*1 + w1*0 >= 0 => w0 >= 0
    ax.axvline(x=0, color="g", linestyle="-", linewidth=2, label="Constraint 1: w₀ ≥ 0")

    # Constraint 2: from x=(1,1), t=0 => w0*1 + w1*1 < 0 => w1 < -w0
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
    # We need w0 > 0 and w1 < -w0
    ax.fill_between(
        w0,
        -2,
        w1_constraint2,
        where=w0 >= 0,
        color="gray",
        alpha=0.5,
        label="Feasible Region (Solution Space)",
    )

    # Plot an example valid weight vectormpy as np
    example_w = (0.5, -1)
    ax.plot(
        example_w[0],
        example_w[1],
        "ko",
        markersize=10,
        label=f"Example Solution w=({example_w[0]}, {example_w[1]})",
    )

    # Annotations and labels
    ax.set_title("Weight Space for NOT Function", fontsize=16)
    ax.set_xlabel("Bias Weight (w₀)", fontsize=12)
    ax.set_ylabel("Input Weight (w₁)", fontsize=12)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect("equal", adjustable="box")

    plt.savefig("not_weight_space.png")
    print("Generated not_weight_space.png")


if __name__ == "__main__":
    plot_input_space()
    plot_weight_space()
