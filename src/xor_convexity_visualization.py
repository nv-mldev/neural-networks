import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import os

# XOR data points
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])  # XOR output

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 8))

# Set up custom colormap
cmap = ListedColormap(["#ff9999", "#9999ff"])

# Scatter plot for XOR data points
scatter = ax.scatter(
    X[:, 0], X[:, 1], c=y, cmap=cmap, s=200, edgecolors="k", linewidths=1.5
)

# Add labels to each point
for i, (x, label) in enumerate(zip(X, y)):
    ax.annotate(
        f"({int(x[0])}, {int(x[1])}) → {int(label)}",
        xy=(x[0], x[1]),
        xytext=(x[0] + 0.05, x[1] + 0.05),
        fontsize=14,
    )

# Connect points of same class to show non-convexity
# Class 0 (negative) points: (0,0) and (1,1)
x_neg = [0, 1]
y_neg = [0, 1]
ax.plot(x_neg, y_neg, "r--", linewidth=2, label="Class 0 connection", alpha=0.7)

# Class 1 (positive) points: (0,1) and (1,0)
x_pos = [0, 1]
y_pos = [1, 0]
ax.plot(x_pos, y_pos, "b--", linewidth=2, label="Class 1 connection", alpha=0.7)

# Mark the intersection point
ax.plot(0.5, 0.5, "ko", markersize=10)
ax.annotate(
    "Intersection Point (0.5, 0.5)",
    xy=(0.5, 0.5),
    xytext=(0.7, 0.7),
    fontsize=12,
    arrowprops=dict(arrowstyle="->", lw=1.5, color="black"),
    color="black",
)


# Function to generate grid of points
def make_grid(x, y, h=0.01):
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# Create grid
xx, yy = make_grid(X[:, 0], X[:, 1], h=0.01)

# Plot any line (showing it cannot separate)
# For demonstration, let's try different lines
lines = [{"m": 0, "b": 0.5, "label": "y = 0.5"}, {"m": 1, "b": 0, "label": "y = x"}]

# Plot each line and show why it fails
for i, line in enumerate(lines):
    m, b = line["m"], line["b"]
    x_vals = np.array([-0.5, 1.5])
    y_vals = m * x_vals + b
    ax.plot(x_vals, y_vals, linestyle=":", linewidth=2, label=line["label"], alpha=0.5)

# Legend
legend_elements = [
    Patch(facecolor="#ff9999", edgecolor="k", label="Class 0"),
    Patch(facecolor="#9999ff", edgecolor="k", label="Class 1"),
]

# Styling
ax.set_xlim(-0.25, 1.25)
ax.set_ylim(-0.25, 1.25)
ax.set_xlabel("X₁", fontsize=16)
ax.set_ylabel("X₂", fontsize=16)
ax.set_title(
    "XOR Problem: Violation of Convexity Requirement", fontsize=18, fontweight="bold"
)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.tick_params(axis="both", which="major", labelsize=14)
ax.grid(True, linestyle="--", alpha=0.7)

# Add explanation text box
props = dict(boxstyle="round", facecolor="white", alpha=0.7)
explanation = (
    "XOR Problem: Convexity Violation\n\n"
    "• Red dotted line connects Class 0 points\n"
    "• Blue dotted line connects Class 1 points\n"
    "• The lines intersect at (0.5, 0.5)\n"
    "• This intersection proves non-convexity\n\n"
    "For linear separability, each class must\n"
    "form a convex set (no such intersection)."
)
ax.text(
    1.3,
    0.5,
    explanation,
    transform=ax.transData,
    fontsize=14,
    verticalalignment="center",
    bbox=props,
)

# Custom legend with the lines
ax.legend(
    handles=legend_elements
    + [
        plt.Line2D([0], [0], color="r", linestyle="--", label="Class 0 connection"),
        plt.Line2D([0], [0], color="b", linestyle="--", label="Class 1 connection"),
        plt.Line2D(
            [0],
            [0],
            color="k",
            marker="o",
            linestyle="None",
            label="Intersection Point",
        ),
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, -0.05),
    ncol=3,
    fontsize=12,
)

# Save the figure
plt.tight_layout()

# Ensure the figures directory exists
os.makedirs("figures", exist_ok=True)

# Save to the figures directory (from project root)
output_path = "figures/xor_convexity_violation.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"XOR convexity visualization created and saved to '{output_path}'")
