import numpy as np

def generate_data(n, num_samples=10):
    np.random.seed(42)
    X = np.random.rand(num_samples, n)  # Random x ∈ [0, 1]
    weights_true = np.random.uniform(-1, 1, n)  # True weights ∈ [-1, 1]
    bias_true = 5  # True bias
    y_true = np.dot(X, weights_true) + bias_true  # True y
    return X, y_true, weights_true, bias_true

def train_perceptron(X, y_true, n, learning_rate=0.01, epochs=100):
    weights = np.random.randn(n)  # Initialize weights randomly
    bias = np.random.randn()      # Initialize bias randomly
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
    
    return weights, bias, mse_history

# Test for n=4 and n=5
for n in [4, 5]:
    print(f"\n--- {n}-Feature Perceptron ---")
    X, y_true, weights_true, bias_true = generate_data(n)
    print(f"True Weights: {weights_true}, True Bias: {bias_true}")
    
    weights, bias, mse_history = train_perceptron(X, y_true, n)
    
    print("\nFinal Learned Weights:", weights)
    print("Final Learned Bias:", bias)
    print("Final MSE:", mse_history[-1])
    
    # Compare true and predicted y
    print("\nSample Predictions:")
    for i in range(3):
        print(f"True y: {y_true[i]:.4f}, Predicted y: {np.dot(X[i], weights) + bias:.4f}")