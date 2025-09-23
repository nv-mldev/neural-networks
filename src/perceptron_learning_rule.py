import numpy as np


X = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 0]])
t = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

X_bias = np.vstack((np.ones((1, 4)), X))
w_bias = np.random.rand(5, 2)
print(X_bias)


def perceptron_learning_rule(X, t, w, learning_rate=0.1, epochs=1):
    for epoch in range(epochs):
        Y = np.dot(X.transpose(), w)
        Y[Y >= 0] = 1  # step activation function
        Y[Y < 0] = 0  # step activation function
        E = t - Y  # error calculation 
        w_update = learning_rate * np.dot(X, E)
        print(E, w_update)


perceptron_learning_rule(X_bias, t, w_bias, epochs=1)
