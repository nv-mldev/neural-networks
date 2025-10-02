import numpy as np
import matplotlib.pyplot as plt

# Example data: 2D points, linearly separable
np.random.seed(0)
# Positive examples
X_pos = np.array([[2, 4], [3, 5], [4, 4.5]])
# Negative examples
X_neg = np.array([[5, 1.5], [6, 2], [5.5, 0.5]])

# Optimal weight vector and margin

# Find closest points between classes (support vectors)
from scipy.spatial.distance import cdist

dist_matrix = cdist(X_pos, X_neg)
min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
sv_pos = X_pos[min_idx[0]]
sv_neg = X_neg[min_idx[1]]

# Optimal weight vector is the direction between support vectors
w = sv_pos - sv_neg
w = w / np.linalg.norm(w)
# Decision boundary is halfway between support vectors
midpoint = (sv_pos + sv_neg) / 2
b = -np.dot(w, midpoint)
# Optimal margin is half the distance between support vectors projected onto w
margin = np.dot(w, sv_pos - sv_neg) / 2

# Decision boundary: w^T x + b = 0
# Margin boundaries: w^T x + b = ±margin
xx = np.linspace(1, 7, 200)


def boundary_line(offset):
    # w1*x + w2*y + b = offset => y = (offset - w1*x - b)/w2
    return (offset - w[0] * xx - b) / w[1]


plt.figure(figsize=(7, 5))
# Plot positive and negative examples
plt.scatter(
    X_pos[:, 0], X_pos[:, 1], color="blue", marker="+", s=100, label="Positive (+1)"
)
plt.scatter(
    X_neg[:, 0], X_neg[:, 1], color="red", marker="o", s=80, label="Negative (-1)"
)

# Plot decision boundary and margins
plt.plot(xx, boundary_line(0), "k-", lw=2, label="Decision Boundary")
plt.plot(xx, boundary_line(margin), "g--", lw=2, label="Margin +γ (Support Vector)")
plt.plot(xx, boundary_line(-margin), "g--", lw=2, label="Margin -γ (Support Vector)")

# Fill margin region
plt.fill_between(
    xx, boundary_line(-margin), boundary_line(margin), color="green", alpha=0.1
)

plt.xlim(1, 7)
plt.ylim(0, 6)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.title("Optimal Separating Hyperplane with Margin $\gamma$")
plt.legend(loc="upper right")
plt.grid(True, linestyle=":")
plt.tight_layout()
plt.savefig("optimal_margin.png", dpi=300)
plt.show()
