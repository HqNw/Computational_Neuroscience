#!/usr/bin/env python

import random
import math

def tanh(x) -> float:
  x = max(min(x, 10), -10)
  exp_2x = math.exp(2 * x)
  return (exp_2x - 1) / (exp_2x + 1)

  # Aleternative implementation
  return math.tanh(x) 


def init_weight() -> float:
  return random.uniform(-0.5, 0.5)

#@NOTE: Network structure
network = {
  'inputs': {'i1': 0.05, 'i2': 0.10},
  'biases': {'b1': 0.5, 'b2': 0.7},
  'weights': {
    'w1': init_weight(), 'w2': init_weight(),
    'w3': init_weight(), 'w4': init_weight(),
    'w5': init_weight(), 'w6': init_weight(),
    'w7': init_weight(), 'w8': init_weight()
  }
}

def forward_pass(network: dict) -> dict:
  # hidden layer values
  h1_input = network['inputs']['i1'] * network['weights']['w1'] + network['inputs']['i2'] * network['weights']['w3'] + network['biases']['b1']
  h2_input = network['inputs']['i1'] * network['weights']['w2'] + network['inputs']['i2'] * network['weights']['w4'] + network['biases']['b1']
  h1_output = tanh(h1_input)
  h2_output = tanh(h2_input)

  # output layer values
  o1_input = h1_output * network['weights']['w5'] + h2_output * network['weights']['w7'] + network['biases']['b2']
  o2_input = h1_output * network['weights']['w6'] + h2_output * network['weights']['w8'] + network['biases']['b2']

  # Activation function
  o1_output = tanh(o1_input)
  o2_output = tanh(o2_input)

  return {
    'h1_input': h1_input, 'h2_input': h2_input,
    'h1_output': h1_output, 'h2_output': h2_output,
    'o1_input': o1_input, 'o2_input': o2_input,
    'o1_output': o1_output, 'o2_output': o2_output
  }

def compute_error(results: dict) -> float:
  target_o1 = 0.01
  target_o2 = 0.99

  #@NOTE: Mean squared error
  error_o1 = 0.5 * (target_o1 - results['o1_output']) ** 2
  error_o2 = 0.5 * (target_o2 - results['o2_output']) ** 2
  return error_o1, error_o2


# Print results
results = forward_pass(network)
print(f"{"":=^50}")
print("Forward Pass Results:")
print(f"{"":=^50}")
print(f"  Hidden Layer Outputs: h1 = {results['h1_output']:.4f}, h2 = {results['h2_output']:.4f}")
print(f"  Output Layer Inputs: o1 = {results['o1_input']:.4f}, o2 = {results['o2_input']:.4f}")
print(f"  Output Layer Outputs: o1 = {results['o1_output']:.4f}, o2 = {results['o2_output']:.4f}")

error_o1, error_o2 = compute_error(results)
print(f"{"":=^50}")
print("Errors:")
print(f"{"":=^50}")
print(f"  Error for o1: {error_o1:.4f}")
print(f"  Error for o2: {error_o2:.4f}")

print(f"\nTotal Error: {error_o1 + error_o2:.4f}")
print(f"{"":=^50}")


#@TODO: Will implement in the future
def backwards_pass():
  pass


#@NOTE: the network can be trained like this after the backpropagation algorithm is implemented
# for i in range(1000):
#   results = forward_pass(network)
#   error = compute_error(results)
#   backwards_pass(error)
