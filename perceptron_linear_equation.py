import numpy as np

# Generate dataset
np.random.seed(42)
X = np.random.rand(10, 3)  # 10 samples, 3 features (x1, x2, x3)
y_true = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + 5  # True y

# Initialize weights and bias
weights = np.random.randn(3)  # w1, w2, w3
bias = np.random.randn()      # b
learning_rate = 0.01
epochs = 100

# Training loop
mse_history = []
for epoch in range(epochs):
    # Predict
    y_pred = np.dot(X, weights) + bias
    
    # Compute MSE
    mse = np.mean((y_pred - y_true) ** 2)
    mse_history.append(mse)
    
    # Gradient descent
    error = y_pred - y_true
    grad_weights = np.dot(X.T, error) / len(X)
    grad_bias = np.mean(error)
    
    weights -= learning_rate * grad_weights
    bias -= learning_rate * grad_bias
    
    # Print MSE every 10 epochs
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: MSE = {mse:.6f}")

# Final results
print("\nFinal Weights:", weights)
print("Final Bias:", bias)
print("Final MSE:", mse_history[-1])

# Compare true and predicted y
print("\nSample Predictions:")
for i in range(3):
    print(f"True y: {y_true[i]:.4f}, Predicted y: {np.dot(X[i], weights) + bias:.4f}")