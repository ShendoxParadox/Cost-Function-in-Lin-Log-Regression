# %%
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate=0.01, epochs=1000):
    m, n = X.shape
    theta = np.zeros((n, 1))
    bias = 0

    cost_values = []  # to store cost values for each epoch

    for epoch in range(epochs):
        z = X.dot(theta) + bias
        h = sigmoid(z)

        # Calculate the logistic loss (cross-entropy loss)
        cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        cost_values.append(cost)

        # Compute gradients
        gradients = (1/m) * X.T.dot(h - y)
        bias_gradient = np.mean(h - y)

        # Update weights and bias
        theta -= learning_rate * gradients
        bias -= learning_rate * bias_gradient

        # Print cost every 100 epochs
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Cost: {cost}')

    return theta, bias, cost_values

def predict(X, theta, bias):
    z = X.dot(theta) + bias
    h = sigmoid(z)
    return (h >= 0.5).astype(int)

# Generate synthetic data for binary classification
np.random.seed(42)
X_positive = np.random.randn(50, 2) + np.array([2, 2])
X_negative = np.random.randn(50, 2) + np.array([-2, -2])
X = np.vstack([X_positive, X_negative])
y = np.vstack([np.ones((50, 1)), np.zeros((50, 1))])

# Add bias term to X
X_b = np.c_[np.ones((len(X), 1)), X]

# Shuffle the data
shuffle_index = np.random.permutation(len(X_b))
X_shuffled, y_shuffled = X_b[shuffle_index], y[shuffle_index]

# Train logistic regression model
theta, bias, cost_values = logistic_regression(X_shuffled, y_shuffled)

# Make predictions
y_pred = predict(X_shuffled, theta, bias)

# Visualize the decision boundary
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', marker='o', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression - Decision Boundary')

# Plot decision boundary
boundary_x = np.array([np.min(X[:, 0]), np.max(X[:, 0])])
boundary_y = (-1/theta[2]) * (theta[1] * boundary_x + bias)
plt.plot(boundary_x, boundary_y, color='red', label='Decision Boundary')

plt.legend()
plt.show()

# Visualize the convergence of the cost function
plt.plot(range(len(cost_values)), cost_values)
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.title('Convergence of Cost Function (Logistic Regression)')
plt.show()

# %%
