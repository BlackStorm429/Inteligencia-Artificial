import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, epochs=1000):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(num_inputs + 1)  # +1 para o bias

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels):
        for _ in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)

def generate_truth_table(n, func):
    table = np.array([[int(j) for j in format(i, '0' + str(n) + 'b')] for i in range(2**n)])
    if func == "XOR":
        results = np.array([sum(row) % 2 for row in table], dtype=int)
    else:
        raise ValueError("Func must be 'XOR'")
    return table, results

def plot_results(inputs, labels, perceptron, func):
    plt.figure(figsize=(10, 6))
    plt.scatter(inputs[:, 0], inputs[:, 1], c=labels, cmap='viridis', marker='o', label='Training Data')
    plt.xlabel("Input 1")
    plt.ylabel("Input 2")
    plt.title(f"Perceptron Results for {func}")
    x_values = np.array([inputs[:, 0].min(), inputs[:, 0].max()])
    y_values = (-perceptron.weights[0] - perceptron.weights[1] * x_values) / perceptron.weights[2]
    plt.plot(x_values, y_values, label='Decision Boundary', c='r')
    plt.legend()
    plt.show()

# Solicitação do número de entradas e da função booleana
n = 2
func = "XOR"

# Geração da tabela verdade e treinamento do Perceptron
inputs, labels = generate_truth_table(n, func)
perceptron = Perceptron(num_inputs=n, epochs=1000)
perceptron.train(inputs, labels)

# Plotagem dos resultados para função XOR
plot_results(inputs, labels, perceptron, func)
