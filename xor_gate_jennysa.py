import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)


    inputs = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]
    targets = [0, 1, 1, 0]

    # Initial weights and biases
    w1, w2 = 0.34, 0.5
    w3, w4 = 0.5, 0.69
    b1, b2 = 0.3, 0.44

    w5, w6 = 0.5, 0.5
    b3 = 0.1

    learning_rate = 0.1
    epochs = 10009

    for epoch in range(epochs):
        for i in range(4):
            x1, x2 = inputs[i]
            y = targets[i]

          
            z1 = x1 * w1 + x2 * w2 + b1
            h1 = sigmoid(z1)

            z2 = x1 * w3 + x2 * w4 + b2
            h2 = sigmoid(z2)

            op = h1 * w5 + h2 * w6 + b3
            yhat = sigmoid(op)

          
            dL = yhat - y
            dyhat = sigmoid_derivative(yhat)

           
            d5 = dL * dyhat * h1
            d6 = dL * dyhat * h2
            db3 = dL * dyhat

           
            w5 -= learning_rate * d5
            w6 -= learning_rate * d6
            b3 -= learning_rate * db3

           
            dh1 = dL * dyhat * w5 * sigmoid_derivative(h1)
            dh2 = dL * dyhat * w6 * sigmoid_derivative(h2)

         
            w1 -= learning_rate * dh1 * x1
            w2 -= learning_rate * dh1 * x2
            b1 -= learning_rate * dh1

            w3 -= learning_rate * dh2 * x1
            w4 -= learning_rate * dh2 * x2
            b2 -= learning_rate * dh2

   
    print("\nTrained XOR Neural Network:")
    for i in range(4):
        x1, x2 = inputs[i]

        z1 = x1 * w1 + x2 * w2 + b1
        h1 = sigmoid(z1)

        z2 = x1 * w3 + x2 * w4 + b2
        h2 = sigmoid(z2)

        op = h1 * w5 + h2 * w6 + b3
        yhat = sigmoid(op)

        print(f"Input: [{x1}, {x2}] -> Predicted: {int(yhat > 0.5)} ({yhat:.4f})")


