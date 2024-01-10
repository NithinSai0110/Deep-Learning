import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data')
plt.show()
X_b = np.c_[np.ones((100, 1)), X]
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
print("Weight (theta_1):", theta[1][0])
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with Gradient Descent')
plt.legend()
plt.show()
