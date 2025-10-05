import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Two points
x1 = np.array([1.0, 2.0])
x2 = np.array([4.0, 5.0])

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.25)

# Plot points and line segment
ax.scatter(*x1, color="blue", label="x1")
ax.scatter(*x2, color="green", label="x2")
ax.plot([x1[0], x2[0]], [x1[1], x2[1]], "r-", label="Line segment")

# Movable point (convex combo)
(point,) = ax.plot([], [], "ro", markersize=10, label="λx1+(1−λ)x2")

# Slider axis
ax_lambda = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_lambda, "λ", -1.0, 2.0, valinit=0.5, valstep=0.01)


# Update function
def update(val):
    lam = slider.val
    combo = lam * x1 + (1 - lam) * x2
    point.set_data([combo[0]], [combo[1]])
    fig.canvas.draw_idle()


slider.on_changed(update)

# Initial update
update(0.5)

ax.set_title("Interactive Convex Combination")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
ax.axis("equal")
ax.legend()
plt.show()
