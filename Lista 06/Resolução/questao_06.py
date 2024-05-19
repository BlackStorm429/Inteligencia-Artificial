import numpy as np
import matplotlib.pyplot as plt

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função de ativação sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Função para treinar a rede neural com backpropagation
def train(X, y, num_epochs, learning_rate):
    input_size = X.shape[1]
    hidden_size = 4
    output_size = 1

    # Inicialização aleatória dos pesos
    np.random.seed(1)
    weights_input_hidden = np.random.rand(input_size, hidden_size)
    weights_hidden_output = np.random.rand(hidden_size, output_size)

    # Loop de treinamento
    for epoch in range(num_epochs):
        # Feedforward
        hidden_layer_input = np.dot(X, weights_input_hidden)
        hidden_layer_output = sigmoid(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
        output_layer_output = sigmoid(output_layer_input)

        # Backpropagation
        output_error = y - output_layer_output
        output_delta = output_error * sigmoid_derivative(output_layer_output)

        hidden_error = output_delta.dot(weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)

        # Atualização dos pesos
        weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
        weights_input_hidden += X.T.dot(hidden_delta) * learning_rate

    return weights_input_hidden, weights_hidden_output

# Função para prever a saída da rede neural
def predict(X, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    return output_layer_output

# Função para calcular o resultado de funções booleanas
def boolean_function(inputs, function):
    if function == 'AND':
        return int(all(inputs))
    elif function == 'OR':
        return int(any(inputs))
    elif function == 'XOR':
        return int(np.logical_xor.reduce(inputs))
    else:
        raise ValueError("Função booleana não suportada.")

# Função principal
def main():
    # Parâmetros
    num_epochs = 10000
    learning_rate = 0.1

    # Solicitar ao usuário o número de entradas
    num_inputs = int(input("Enter the number of inputs: "))

    # Solicitar ao usuário a função booleana desejada
    function = input("Enter the boolean function (AND, OR, XOR): ")

    # Dados de entrada
    X = np.array([[i for i in range(2**num_inputs)]])
    X = (X.T & (1 << np.arange(num_inputs))) > 0
    y = np.array([[boolean_function(inputs, function)] for inputs in X])

    # Treinamento da rede neural
    weights_input_hidden, weights_hidden_output = train(X, y, num_epochs, learning_rate)

    # Predição e exibição dos resultados
    for i in range(len(X)):
        prediction = predict(X[i], weights_input_hidden, weights_hidden_output)
        print(f"Entrada: {X[i]}, Saída Prevista: {prediction}, Saída Real: {y[i]}")

    # Plotagem dos resultados
    plt.plot(X, y, label='Real')
    plt.plot(X, predict(X, weights_input_hidden, weights_hidden_output), label='Previsto')
    plt.xlabel('Entrada')
    plt.ylabel('Saída')
    plt.title(f'Função {function}')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
