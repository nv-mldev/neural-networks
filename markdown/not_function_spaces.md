# Visualizing Perceptron Spaces: The NOT Function Example

Understanding a Perceptron involves more than just its learning rule; it requires grasping the geometry of how it makes decisions. We can visualize this geometry in two primary ways: the **Input Space** and the **Weight Space**. This document explores both, using the simple logical NOT function as our running example.

The NOT function is defined as:
-   Input `0` -> Output `1`
-   Input `1` -> Output `0`

To handle the bias term elegantly, we use the "bias trick" by adding a dummy input `x₀` that is always 1. Our data points `(x₀, x₁)` are therefore:
-   For input `0`: `(1, 0)` -> Target `1` (Positive Class)
-   For input `1`: `(1, 1)` -> Target `0` (Negative Class)

The Perceptron's decision rule is `y = 1` if `w₀x₀ + w₁x₁ ≥ 0`, and `y = 0` otherwise.

---

## 1. The Input Space (or Data Space)

The Input Space is the most common way to visualize a classifier. It's a geometric representation of the data points. The goal of the Perceptron is to find a **decision boundary**—a line in 2D, or a hyperplane in higher dimensions—that perfectly separates the positive and negative data points.

The equation for this decision boundary is `w₀x₀ + w₁x₁ = 0`. Since `x₀` is always 1, this simplifies to `w₀ + w₁x₁ = 0`.

Let's pick a set of weights that solves the NOT function, for example, `w₀ = 0.5` and `w₁ = -1`.
-   For input `(1, 0)`: `0.5*1 + (-1)*0 = 0.5 ≥ 0` -> Output `1`. Correct.
-   For input `(1, 1)`: `0.5*1 + (-1)*1 = -0.5 < 0` -> Output `0`. Correct.

With these weights, the decision boundary is `0.5 - 1*x₁ = 0`, which solves to `x₁ = 0.5`.

### Input Space Visualization

The plot below shows this input space.

![Input Space for NOT Function](not_input_space.png)

**Key Takeaways:**

1.  **Axes and Points:** The horizontal axis is our real input `x₁`, and the vertical axis is the dummy bias input `x₀`. The blue circle at `(0, 1)` is our positive data point, and the red square at `(1, 1)` is our negative data point.
2.  **Decision Boundary:** The dashed black line at `x₁ = 0.5` is the decision boundary. This line perfectly separates the two classes.
3.  **Half-Spaces:** The boundary divides the space into two regions (half-spaces). The blue shaded region on the left is the "positive region," where any point will be classified as 1. The red shaded region on the right is the "negative region," where any point will be classified as 0.
4.  **The Role of Weights:** The weights `(w₀, w₁)` define the decision boundary. The weight vector `w` is always perpendicular (normal) to the decision boundary and points into the positive half-space.

In essence, the input space shows **one solution** (one decision boundary) that correctly partitions the entire dataset.

---

## 2. The Weight Space

While the input space plots the data, the **Weight Space** plots the possible solutions. The axes of this space are the weights themselves (`w₀` and `w₁`). Every point in this space represents a different Perceptron model (a different decision boundary).

Each data point from our training set imposes a constraint on what the weights can be. A valid solution must satisfy all constraints simultaneously.

**Deriving the Constraints:**

1.  **Data Point (1, 0), Target 1:**
    -   We need `w₀(1) + w₁(0) ≥ 0`.
    -   This simplifies to the constraint: **`w₀ ≥ 0`**.

2.  **Data Point (1, 1), Target 0:**
    -   We need `w₀(1) + w₁(1) < 0`.
    -   This simplifies to the constraint: **`w₀ + w₁ < 0`**, or **`w₁ < -w₀`**.

The Perceptron learning algorithm is guaranteed to find a weight vector `(w₀, w₁)` that satisfies both of these inequalities, because the problem is linearly separable.

### Weight Space Visualization

The plot below shows this weight space.

![Weight Space for NOT Function](not_weight_space.png)

**Key Takeaways:**

1.  **Axes:** The axes represent the values of the bias weight `w₀` and the input weight `w₁`.
2.  **Constraints as Lines:** Each inequality defines a half-space in the weight space. The green line (`w₀ = 0`) and the magenta line (`w₁ = -w₀`) are the boundaries of these constraints.
3.  **Feasible Region (Solution Space):** The shaded gray area is the **feasible region** (also called the *version space*). **Any point `(w₀, w₁)` chosen from this region is a valid weight vector that solves the NOT function.**
4.  **Example Solution:** The black dot at `(0.5, -1)` is the specific solution we used for the input space visualization. Notice that it lies comfortably within the feasible region, satisfying both constraints.

### Input Space vs. Weight Space

-   **Input Space:** Shows data points. A single point in weight space (a single solution) corresponds to a single decision boundary line in input space.
-   **Weight Space:** Shows solutions (weights). A single data point from the input space corresponds to a constraint line that defines a half-space of valid solutions in weight space.

By visualizing both spaces, we gain a complete geometric understanding of how a simple classifier like the Perceptron works. It's a search for a valid point in the weight space, which in turn defines a separating hyperplane in the input space.
