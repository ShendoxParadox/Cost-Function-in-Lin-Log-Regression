# %%
import numpy as np
import matplotlib.pyplot as plt

def linear_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    bias = 0

    mse_values = []  # to store MSE values for each epoch

    for epoch in range(epochs):
        h = X.dot(theta) + bias
        error = h - y

        gradients = (1/m) * X.T.dot(error)
        bias_gradient = np.mean(error)

        theta -= learning_rate * gradients
        bias -= learning_rate * bias_gradient

        mse = np.mean((error ** 2))
        mse_values.append(mse)

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, MSE: {mse}')

    return theta, bias, mse_values

def predict(X, theta, bias):
    return X.dot(theta) + bias

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((len(X), 1)), X]

theta, bias, mse_values = linear_regression(X_b, y)

y_pred = predict(X_b, theta, bias)

mse = np.mean((y_pred - y) ** 2)
print(f'Final MSE: {mse}')

# Visualize the convergence of the loss function
plt.plot(range(len(mse_values)), mse_values)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Convergence of Loss Function (MSE)')
plt.show()

# Visualize the regression line
plt.scatter(X, y, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with MSE')
plt.legend()
plt.show()

# %%
