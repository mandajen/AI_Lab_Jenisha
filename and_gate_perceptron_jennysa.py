       def step_function(x):
    return 1 if x >= 0 else 0


    weights = [0.5, 1.0]
    bias = -0.7

    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    expected_outputs = [0, 0, 0, 1]

    epochs = 10
    learning_rate = 0.1

    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        for i in range(4):
            x1, x2 = inputs[i]
            target = expected_outputs[i]
            total_input = weights[0] * x1 + weights[1] * x2 + bias
            output = step_function(total_input)
            error = target - output

           
            weights[0] += learning_rate * error * x1
            weights[1] += learning_rate * error * x2
            bias += learning_rate * error

            print(f"Inputs: [{x1}, {x2}], Output: [{output}], Expected: {target}, Error: {error}")

        print(f"\nWeights: {weights}\nBias: {bias}\n")



