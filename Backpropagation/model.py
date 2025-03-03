#!/usr/bin/env python

import random

class Neuron:
    def __init__(self, bias=None):
        self.bias = bias if bias is not None else random.uniform(-0.5, 0.5)
        self.weights = []
        self.delta = 0
        self.output = 0
        self.input_sum = 0
    
    def activate(self, inputs):
        self.input_sum = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        self.output = self.tanh(self.input_sum)
        return self.output

    @staticmethod
    def tanh(x):
        x = max(min(x, 10), -10)  # Prevent overflow
        exp_2x = 2.7182818284590452353602874713527 ** (2 * x)  # e^(2x)
        return (exp_2x - 1) / (exp_2x + 1)

    @staticmethod
    def tanh_derivative(tanh_output):
        return 1.0 - tanh_output ** 2


class Layer:
    def __init__(self, neuron_count, input_count, bias_value=None):
        self.neurons = [Neuron(bias_value) for _ in range(neuron_count)]
        
        # Initialize weights for each neuron
        for neuron in self.neurons:
            neuron.weights = [random.uniform(-0.5, 0.5) for _ in range(input_count)]
    
    def forward(self, inputs):
        return [neuron.activate(inputs) for neuron in self.neurons]


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_layer = None  # Just a placeholder for inputs
        self.hidden_layer = Layer(hidden_size, input_size, 0.5)
        self.output_layer = Layer(output_size, hidden_size, 0.7)
        self.learning_rate = 0.1
        
    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = self.hidden_layer.forward(inputs)
        self.outputs = self.output_layer.forward(self.hidden_outputs)
        return self.outputs
        
    def backward(self, targets):
        # Calculate output layer deltas
        for i, neuron in enumerate(self.output_layer.neurons):
            error = targets[i] - neuron.output
            neuron.delta = error * Neuron.tanh_derivative(neuron.output)
        
        # Calculate hidden layer deltas
        for i, neuron in enumerate(self.hidden_layer.neurons):
            error = 0.0
            for j, output_neuron in enumerate(self.output_layer.neurons):
                error += output_neuron.delta * output_neuron.weights[i]
            neuron.delta = error * Neuron.tanh_derivative(neuron.output)
        
        # Update output layer weights
        for i, neuron in enumerate(self.output_layer.neurons):
            for j in range(len(neuron.weights)):
                neuron.weights[j] += self.learning_rate * neuron.delta * self.hidden_outputs[j]
            neuron.bias += self.learning_rate * neuron.delta
        
        # Update hidden layer weights
        for i, neuron in enumerate(self.hidden_layer.neurons):
            for j in range(len(neuron.weights)):
                neuron.weights[j] += self.learning_rate * neuron.delta * self.inputs[j]
            neuron.bias += self.learning_rate * neuron.delta

    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            self.backward(targets)
            
            if epoch % 100 == 0:
                error = self.calculate_error(outputs, targets)
                print(f"Epoch {epoch}: Total error = {error:.4f}")
        
        # Final results
        final_outputs = self.forward(inputs)
        final_error = self.calculate_error(final_outputs, targets)
        return final_outputs, final_error
    
    def calculate_error(self, outputs, targets):
        return sum(0.5 * (targets[i] - outputs[i])**2 for i in range(len(outputs)))
        
    def get_network_state(self):
        state = {
            'inputs': self.inputs,
            'hidden': {
                'inputs': [n.input_sum for n in self.hidden_layer.neurons],
                'outputs': self.hidden_outputs
            },
            'outputs': {
                'inputs': [n.input_sum for n in self.output_layer.neurons],
                'outputs': self.outputs
            }
        }
        return state


if __name__ == "__main__":
    # Initialize network
    network = NeuralNetwork(input_size=2, hidden_size=2, output_size=2)
    
    # Define inputs and targets
    inputs = [0.05, 0.10]
    targets = [0.01, 0.99]
    
    print(f"{'':=^50}")
    print("Initial state before training:")
    initial_outputs = network.forward(inputs)
    initial_error = network.calculate_error(initial_outputs, targets)
    
    state = network.get_network_state()
    print(f"{'':=^50}")
    print(f"\tHidden Layer Outputs: h1 = {state['hidden']['outputs'][0]:.4f}, h2 = {state['hidden']['outputs'][1]:.4f}")
    print(f"\tOutput Layer Inputs: o1 = {state['outputs']['inputs'][0]:.4f}, o2 = {state['outputs']['inputs'][1]:.4f}")
    print(f"\tOutput Layer Outputs: o1 = {state['outputs']['outputs'][0]:.4f}, o2 = {state['outputs']['outputs'][1]:.4f}")
    print(f"{'':=^50}")
    print(f"\tError for o1: {0.5 * (targets[0] - initial_outputs[0])**2:.4f}")
    print(f"\tError for o2: {0.5 * (targets[1] - initial_outputs[1])**2:.4f}")
    print(f"\nTotal Error: {initial_error:.4f}")
    print(f"{'':=^50}")
    
    # Train the network
    print("\nTraining network for 1000 epochs...\n")
    final_outputs, final_error = network.train(inputs, targets, 1000)
    
    # Display results after training
    print(f"{'':=^50}")
    print("Results after training:")
    state = network.get_network_state()
    print(f"{'':=^50}")
    print(f"\tHidden Layer Outputs: h1 = {state['hidden']['outputs'][0]:.4f}, h2 = {state['hidden']['outputs'][1]:.4f}")
    print(f"\tOutput Layer Inputs: o1 = {state['outputs']['inputs'][0]:.4f}, o2 = {state['outputs']['inputs'][1]:.4f}")
    print(f"\tOutput Layer Outputs: o1 = {state['outputs']['outputs'][0]:.4f}, o2 = {state['outputs']['outputs'][1]:.4f}")
    print(f"{'':=^50}")
    print(f"\tError for o1: {0.5 * (targets[0] - final_outputs[0])**2:.4f}")
    print(f"\tError for o2: {0.5 * (targets[1] - final_outputs[1])**2:.4f}")
    print(f"\nTotal Error: {final_error:.4f}")
    print(f"{'':=^50}")
