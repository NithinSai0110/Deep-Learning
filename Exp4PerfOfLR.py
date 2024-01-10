import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
data = load_iris()
X = data.data[:, 0]
y = data.target
x_mean = np.mean(X)
y_mean = np.mean(y)
cross_deviation = np.sum((X - x_mean) * (y - y_mean))
x_deviation = np.sum((X - x_mean) ** 2)
b1 = cross_deviation / x_deviation
b0 = y_mean - (b1 * x_mean)
def plot_regression_line():
    plt.scatter(X, y, color='blue', label='Data Points')
    plt.plot(X, b0 + b1 * X, color='red', label='Regression Line')
    plt.xlabel('Sepal Length')
    plt.ylabel('Species')
    plt.title('Linear Regression on Iris Dataset')
    plt.legend()
    plt.show()
print("Regression Coefficients:")
print("Intercept (b0):", b0)
print("Slope (b1):", b1)
plot_regression_line()