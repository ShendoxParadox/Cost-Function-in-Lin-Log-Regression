# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
def linear_regression(X, y, learning_rate=0.01, epochs=1000):
    # Initialize weights and bias
    m, n = X.shape
    theta = np.zeros((n, 1))
    bias = 0

    # Gradient Descent
    for epoch in range(epochs):
        # Hypothesis (predicted values)
        h = X.dot(theta) + bias

        # Calculate the error
        error = h - y

        # Compute gradients
        gradients = (1/m) * X.T.dot(error)
        bias_gradient = np.mean(error)

        # Update weights and bias
        theta -= learning_rate * gradients
        bias -= learning_rate * bias_gradient

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((error ** 2))

        # Print MSE every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, MSE: {mse}')

    return theta, bias

def predict(X, theta, bias):
    return X.dot(theta) + bias

# %%
# Generate some random data for demonstration purposes
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Add a bias term to X
X_b = np.c_[np.ones((len(X), 1)), X]

# Train the linear regression model
theta, bias = linear_regression(X_b, y)

# Make predictions
y_pred = predict(X_b, theta, bias)

# Calculate Mean Squared Error (MSE)
mse = np.mean((y_pred - y) ** 2)
print(f'Final MSE: {mse}')

# %%
# Visualize the regression line
plt.scatter(X, y, label='Actual Data')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression with MSE')
plt.legend()
plt.show()

# %%
