# Visualizing Perceptron Spaces: The AND Function Example

This document explores the Input Space and Weight Space for a Perceptron trained to learn the logical AND function. The AND function is a classic example of a linearly separable problem, making it ideal for understanding the geometry of a Perceptron.

The AND function for two inputs is defined as:

| x₁ | x₂ | Target |
|----|----|--------|
| 0  | 0  | 0      |
| 0  | 1  | 0      |
| 1  | 0  | 0      |
| 1  | 1  | 1      |

We use the "bias trick" by adding a dummy input `x₀ = 1`. The Perceptron's decision rule is `y = 1` if `w₀x₀ + w₁x₁ + w₂x₂ ≥ 0`, and `y = 0` otherwise.

---

## 1. The Input Space

The Input Space shows our data points and the decision boundary that separates them. For the AND function, we have three "negative" points (target=0) and one "positive" point (target=1). The Perceptron must find a line that isolates the single positive point from the three negative ones.

Let's consider a valid set of weights that solves this problem: `w₀ = -0.3`, `w₁ = 0.2`, `w₂ = 0.2`.

The decision boundary is given by the equation `w₀ + w₁x₁ + w₂x₂ = 0`.
Plugging in our weights:
`-0.3 + 0.2x₁ + 0.2x₂ = 0`

Solving for `x₂`, we get the line:
`x₂ = -x₁ + 1.5`

### Input Space Visualization

The plot below shows the four data points and the decision boundary.

![Input Space for AND Function](and_input_space.png)

**Key Takeaways:**

1.  **Data Points:** The three red squares are the negative class points `(0,0)`, `(0,1)`, and `(1,0)`. The single blue circle at `(1,1)` is the positive class point.
2.  **Decision Boundary:** The dashed black line (`x₂ = -x₁ + 1.5`) is a valid decision boundary. It successfully separates the input space.
3.  **Half-Spaces:** The blue shaded area is the "positive region." Any point falling in this region will be classified as 1. The red shaded area is the "negative region." All three negative points lie here.
4.  **Weight Vector:** The weight vector `w = (w₁, w₂)` is normal (perpendicular) to the decision boundary and points into the positive half-space, as indicated by the black arrow.

---

## 2. The Weight Space

The Weight Space represents the set of all possible solutions. For the AND function, the weights are `(w₀, w₁, w₂)`. This is a 3D space. Each point in this space is a unique Perceptron model.

Each of our four data points imposes a constraint on the possible values of `(w₀, w₁, w₂)`.

**Deriving the Constraints:**

1.  `x=(0,0), t=0` => `w₀*1 + w₁*0 + w₂*0 < 0` => **`w₀ < 0`**
2.  `x=(0,1), t=0` => `w₀*1 + w₁*0 + w₂*1 < 0` => **`w₀ + w₂ < 0`**
3.  `x=(1,0), t=0` => `w₀*1 + w₁*1 + w₂*0 < 0` => **`w₀ + w₁ < 0`**
4.  `x=(1,1), t=1` => `w₀*1 + w₁*1 + w₂*1 ≥ 0` => **`w₀ + w₁ + w₂ ≥ 0`**

A valid solution must satisfy all four inequalities. To visualize this, we can take a 2D "slice" of the 3D space by fixing one weight. Let's fix the bias weight to a valid value, `w₀ = -0.3`, which satisfies the first constraint.

The remaining constraints on `w₁` and `w₂` become:
1.  `-0.3 + w₂ < 0` => **`w₂ < 0.3`**
2.  `-0.3 + w₁ < 0` => **`w₁ < 0.3`**
3.  `-0.3 + w₁ + w₂ ≥ 0` => **`w₁ + w₂ ≥ 0.3`**

### Weight Space Visualization (2D Slice)

The plot below shows the feasible region for `w₁` and `w₂` when `w₀ = -0.3`. The new visualization includes arrows to explicitly show the valid side of each constraint boundary, removing any ambiguity.

![Weight Space for AND Function](and_weight_space.png)

**Key Takeaways:**

1.  **Axes:** The axes represent the values of the input weights `w₁` and `w₂`.
2.  **Constraints:** The dashed lines represent the boundaries of our three inequalities.
    -   The cyan arrow shows that valid solutions must be **below** the `w₂ = 0.3` line.
    -   The magenta arrow shows that valid solutions must be **to the left of** the `w₁ = 0.3` line.
    -   The green arrow shows that valid solutions must be **above** the `w₁ + w₂ = 0.3` line.
3.  **Feasible Region:** The shaded gray area is the **single, triangular solution space** formed by the intersection of all three constraints. **Any point `(w₁, w₂)` picked from this region, when combined with `w₀ = -0.3`, will solve the AND function.**
4.  **Example Solution:** The black dot at `(0.2, 0.2)` corresponds to the specific decision boundary we plotted in the input space. As you can see, it lies correctly within the shaded feasible region, satisfying all three conditions.

By analyzing both spaces, we see the elegant duality of the Perceptron: the learning algorithm's job is to find a single point in the multi-dimensional weight space's feasible region. That single point, in turn, defines a simple line (or hyperplane) in the input space that correctly solves the classification problem.
