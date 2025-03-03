#!/usr/bin/env python

import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.hidden_weights = [[random.uniform(-0.5, 0.5) for _ in range(input_size)] for _ in range(hidden_size)]
        self.output_weights = [[random.uniform(-0.5, 0.5) for _ in range(hidden_size)] for _ in range(output_size)]
        self.hidden_biases = [0.5 for _ in range(hidden_size)]
        self.output_biases = [0.7 for _ in range(output_size)]
        self.learning_rate = 0.1
        
    @staticmethod
    def tanh(x):
        x = max(min(x, 10), -10)
        exp_2x = 2.7182818284590452353602874713527 ** (2 * x)
        return (exp_2x - 1) / (exp_2x + 1)
    
    @staticmethod
    def tanh_derivative(tanh_output):
        return 1.0 - tanh_output ** 2
    
    def forward(self, inputs):
        # Store inputs for backpropagation
        self.inputs = inputs
        
        # Calculate hidden layer outputs
        self.hidden_inputs = []
        self.hidden_outputs = []
        for i, weights in enumerate(self.hidden_weights):
            hidden_input = sum(w * inp for w, inp in zip(weights, inputs)) + self.hidden_biases[i]
            self.hidden_inputs.append(hidden_input)
            self.hidden_outputs.append(self.tanh(hidden_input))
        
        # Calculate output layer outputs
        self.output_inputs = []
        self.outputs = []
        for i, weights in enumerate(self.output_weights):
            output_input = sum(w * h for w, h in zip(weights, self.hidden_outputs)) + self.output_biases[i]
            self.output_inputs.append(output_input)
            self.outputs.append(self.tanh(output_input))
            
        return self.outputs
    
    def backward(self, targets):
        # Calculate output deltas
        output_deltas = []
        for i, output in enumerate(self.outputs):
            error = targets[i] - output
            output_deltas.append(error * self.tanh_derivative(output))
        
        # Calculate hidden deltas
        hidden_deltas = []
        for i in range(len(self.hidden_outputs)):
            error = sum(delta * self.output_weights[j][i] for j, delta in enumerate(output_deltas))
            hidden_deltas.append(error * self.tanh_derivative(self.hidden_outputs[i]))
        
        # Update output weights and biases
        for i, delta in enumerate(output_deltas):
            for j in range(len(self.output_weights[i])):
                self.output_weights[i][j] += self.learning_rate * delta * self.hidden_outputs[j]
            self.output_biases[i] += self.learning_rate * delta
        
        # Update hidden weights and biases
        for i, delta in enumerate(hidden_deltas):
            for j in range(len(self.hidden_weights[i])):
                self.hidden_weights[i][j] += self.learning_rate * delta * self.inputs[j]
            self.hidden_biases[i] += self.learning_rate * delta
    
    def calculate_error(self, outputs, targets):
        return sum(0.5 * (t - o) ** 2 for t, o in zip(targets, outputs))
    
    def train(self, inputs, targets, epochs):
        for epoch in range(epochs):
            outputs = self.forward(inputs)
            self.backward(targets)
            
            if epoch % 100 == 0:
                error = self.calculate_error(outputs, targets)
                print(f"Epoch {epoch}: Total error = {error:.4f}")
        
        final_outputs = self.forward(inputs)
        final_error = self.calculate_error(final_outputs, targets)
        return final_outputs, final_error
    
    def get_network_state(self):
        return {
            'inputs': self.inputs,
            'hidden': {
                'inputs': self.hidden_inputs,
                'outputs': self.hidden_outputs
            },
            'outputs': {
                'inputs': self.output_inputs,
                'outputs': self.outputs
            }
        }


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
    print(f"  Hidden Layer Outputs: h1 = {state['hidden']['outputs'][0]:.4f}, h2 = {state['hidden']['outputs'][1]:.4f}")
    print(f"  Output Layer Inputs: o1 = {state['outputs']['inputs'][0]:.4f}, o2 = {state['outputs']['inputs'][1]:.4f}")
    print(f"  Output Layer Outputs: o1 = {state['outputs']['outputs'][0]:.4f}, o2 = {state['outputs']['outputs'][1]:.4f}")
    print(f"{'':=^50}")
    print(f"  Error for o1: {0.5 * (targets[0] - initial_outputs[0])**2:.4f}")
    print(f"  Error for o2: {0.5 * (targets[1] - initial_outputs[1])**2:.4f}")
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
    print(f"  Hidden Layer Outputs: h1 = {state['hidden']['outputs'][0]:.4f}, h2 = {state['hidden']['outputs'][1]:.4f}")
    print(f"  Output Layer Inputs: o1 = {state['outputs']['inputs'][0]:.4f}, o2 = {state['outputs']['inputs'][1]:.4f}")
    print(f"  Output Layer Outputs: o1 = {state['outputs']['outputs'][0]:.4f}, o2 = {state['outputs']['outputs'][1]:.4f}")
    print(f"{'':=^50}")
    print(f"  Error for o1: {0.5 * (targets[0] - final_outputs[0])**2:.4f}")
    print(f"  Error for o2: {0.5 * (targets[1] - final_outputs[1])**2:.4f}")
    print(f"\nTotal Error: {final_error:.4f}")
    print(f"{'':=^50}")
