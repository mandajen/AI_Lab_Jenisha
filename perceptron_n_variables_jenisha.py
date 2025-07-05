import numpy as np

def generate_truth_table(n, gate_type):
    inputs = np.array(np.meshgrid(*[[0, 1]] * n)).T.reshape(-1, n)
    if gate_type == "AND":
        outputs = np.all(inputs, axis=1).astype(int)
    elif gate_type == "OR":
        outputs = np.any(inputs, axis=1).astype(int)
    return inputs, outputs

def step_function(x):
    return 1 if x >= 0 else 0

def train_perceptron(n, gate_type, learning_rate=0.1, max_epochs=100):
    inputs, outputs = generate_truth_table(n, gate_type)
    weights = np.zeros(n)
    bias = 0
    
    for epoch in range(max_epochs):
        errors = 0
        for x, y_true in zip(inputs, outputs):
            y_pred = step_function(np.dot(weights, x) + bias)
            error = y_true - y_pred
            weights += learning_rate * error * x
            bias += learning_rate * error
            errors += int(error != 0)
        
        if errors == 0:
            print(f"Converged in {epoch + 1} epochs.")
            break
    
    accuracy = np.mean([step_function(np.dot(weights, x) + bias) == y_true for x, y_true in zip(inputs, outputs)])
    return weights, bias, accuracy

# Test for n=3 and n=4
for n in [3, 4]:
    print(f"\n--- {n}-Input Gates ---")
    for gate in ["AND", "OR"]:
        print(f"\nTraining {gate} Gate:")
        weights, bias, accuracy = train_perceptron(n, gate)
        print(f"Final Weights: {weights}, Bias: {bias}")
        print(f"Accuracy: {accuracy * 100:.2f}%")