import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + 1.5 * X ** 2 + np.random.randn(100, 1)
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Modified Data with Quadratic Relationship')
plt.show()
X_b = np.c_[np.ones((100, 1)), X, X ** 2]
def gradient_descent(X, y, iterations=1000, learning_rate=0.01, stopping_threshold=1e-5):
    m, n = X.shape
    theta = np.random.randn(n, 1)
    for i in range(iterations):
        gradients = 2 / m * X.T.dot(X.dot(theta) - y)
        theta = theta - learning_rate * gradients
        loss = np.mean((X.dot(theta) - y) ** 2)
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss}")
        if loss < stopping_threshold:
            print(f"Converged at iteration {i}, Loss: {loss}")
            break
    return theta
theta = gradient_descent(X_b, y)
print("Optimal Parameters:")
print("Bias (theta_0):", theta[0][0])
print("Linear Coefficient (theta_1):", theta[1][0])
print("Quadratic Coefficient (theta_2):", theta[2][0])
X_new = np.linspace(0, 2, 100).reshape(-1, 1)
X_new_b = np.c_[np.ones((100, 1)), X_new, X_new ** 2]
y_pred = X_new_b.dot(theta)
plt.scatter(X, y)
plt.plot(X_new, y_pred, color='red', label='Regression Curve')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Quadratic Regression with Gradient Descent')
plt.legend()
plt.show()