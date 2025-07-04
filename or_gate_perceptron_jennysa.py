def step_function(x):
    return 1 if x >= 0 else 0


    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    weights = [0.5, 0.5]
    bias = 0.2

    expected = [0, 1, 1, 1]  # OR gate output

    learning_rate = 0.1
    epochs = 10

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}:")
        for i in range(4):
            x1, x2 = inputs[i]
            target = expected[i]
            total_input = weights[0] * x1 + weights[1] * x2 + bias
            output = step_function(total_input)
            error = target - output

          
            weights[0] += learning_rate * error * x1
            weights[1] += learning_rate * error * x2
            bias += learning_rate * error

            print(f"Input: [{x1}, {x2}], Output: {output}, Expected: {target}, Error: {error}")

        print(f"Weights: {weights}, Bias: {bias}\n")


