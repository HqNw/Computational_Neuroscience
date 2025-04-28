import random
import math

class Neuron:
  def __init__(self, bias=None):
    self.bias = bias if bias is not None else random.uniform(-0.1, 0.1)
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
    x = max(min(x, 10), -10)
    exp_2x = math.exp(2 * x)
    return (exp_2x - 1) / (exp_2x + 1)
  
  @staticmethod
  def tanh_derivative(tanh_output):
    return 1.0 - tanh_output ** 2
  
  @staticmethod
  def softmax(values):
    max_val = max(values)
    exp_values = [math.exp(val - max_val) for val in values]
    sum_exp = sum(exp_values)
    return [val / sum_exp for val in exp_values]


class RecurrentLayer:
  def __init__(self, neuron_count, input_count, recurrent_count=None):
    recurrent_count = recurrent_count or neuron_count
    self.neurons = [Neuron() for _ in range(neuron_count)]
    
    for neuron in self.neurons:
      neuron.weights = [random.uniform(-0.1, 0.1) for _ in range(input_count)]
    
    self.recurrent_weights = [[random.uniform(-0.1, 0.1) for _ in range(recurrent_count)] 
                 for _ in range(neuron_count)]
    
    self.outputs = [0] * neuron_count
    self.prev_outputs = [0] * neuron_count
  
  def forward(self, inputs):
    self.prev_outputs = self.outputs.copy()
    
    new_outputs = []
    for i, neuron in enumerate(self.neurons):
      input_activation = sum(w * inp for w, inp in zip(neuron.weights, inputs))
      
      recurrent_activation = sum(w * out for w, out in zip(self.recurrent_weights[i], self.prev_outputs))
      
      neuron.input_sum = input_activation + recurrent_activation + neuron.bias
      
      neuron.output = Neuron.tanh(neuron.input_sum)
      new_outputs.append(neuron.output)
    
    self.outputs = new_outputs
    return new_outputs


class OutputLayer:
  def __init__(self, neuron_count, input_count):
    self.neurons = [Neuron() for _ in range(neuron_count)]
    
    for neuron in self.neurons:
      neuron.weights = [random.uniform(-0.1, 0.1) for _ in range(input_count)]
  
  def forward(self, inputs):
    outputs = [neuron.activate(inputs) for neuron in self.neurons]
    
    probabilities = Neuron.softmax([n.input_sum for n in self.neurons])
    
    for i, neuron in enumerate(self.neurons):
      neuron.output = probabilities[i]
    
    return probabilities


class RNN:
  def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
    self.hidden_layer = RecurrentLayer(hidden_size, input_size)
    self.output_layer = OutputLayer(output_size, hidden_size)
    self.learning_rate = learning_rate
    
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.output_size = output_size
  
  def forward(self, inputs_sequence):
    """Forward pass through the RNN for a sequence of inputs"""
    xs, hs, outputs, probabilities = [], [], [], []
    
    self.hidden_layer.prev_outputs = [0] * self.hidden_size
    
    for t in range(len(inputs_sequence)):
      x = inputs_sequence[t]
      xs.append(x)
      
      h = self.hidden_layer.forward(x)
      hs.append(h)
      
      p = self.output_layer.forward(h)
      probabilities.append(p)
    
    return xs, hs, probabilities
  
  def backward(self, xs, hs, ps, targets):
    """Backward pass through the RNN using backpropagation through time"""
    dWhy = [[0 for _ in range(self.hidden_size)] for _ in range(self.output_size)]
    dby = [0] * self.output_size
    dWxh = [[0 for _ in range(self.input_size)] for _ in range(self.hidden_size)]
    dWhh = [[0 for _ in range(self.hidden_size)] for _ in range(self.hidden_size)]
    dbh = [0] * self.hidden_size
    
    dhnext = [0] * self.hidden_size
    
    for t in reversed(range(len(xs))):
      dy = ps[t].copy()
      target_idx = targets[t].index(1)
      dy[target_idx] -= 1
      
      for i in range(self.output_size):
        for j in range(self.hidden_size):
          dWhy[i][j] += dy[i] * hs[t][j]
        dby[i] += dy[i]
      
      dh = [0] * self.hidden_size
      for i in range(self.hidden_size):
        for j in range(self.output_size):
          dh[i] += self.output_layer.neurons[j].weights[i] * dy[j]
      
      for i in range(self.hidden_size):
        dh[i] += dhnext[i]
      
      dtanh = [dh[i] * Neuron.tanh_derivative(hs[t][i]) for i in range(self.hidden_size)]
      
      for i in range(self.hidden_size):
        dbh[i] += dtanh[i]
      
      for i in range(self.hidden_size):
        for j in range(self.input_size):
          dWxh[i][j] += dtanh[i] * xs[t][j]
      
      if t > 0:
        for i in range(self.hidden_size):
          for j in range(self.hidden_size):
            dWhh[i][j] += dtanh[i] * hs[t-1][j]
      
      dhnext = [0] * self.hidden_size
      for i in range(self.hidden_size):
        for j in range(self.hidden_size):
          dhnext[j] += dtanh[i] * self.hidden_layer.recurrent_weights[i][j]
    
    def clip(gradients, min_val=-5, max_val=5):
      if isinstance(gradients[0], list):
        return [[max(min_val, min(max_val, g)) for g in row] for row in gradients]
      else:
        return [max(min_val, min(max_val, g)) for g in gradients]
    
    dWxh = clip(dWxh)
    dWhh = clip(dWhh)
    dWhy = clip(dWhy)
    dbh = clip(dbh)
    dby = clip(dby)
    
    return dWxh, dWhh, dWhy, dbh, dby
  
  def update_parameters(self, dWxh, dWhh, dWhy, dbh, dby):
    """Update model parameters using gradients"""
    for i in range(self.hidden_size):
      for j in range(self.input_size):
        self.hidden_layer.neurons[i].weights[j] -= self.learning_rate * dWxh[i][j]
    
    for i in range(self.hidden_size):
      for j in range(self.hidden_size):
        self.hidden_layer.recurrent_weights[i][j] -= self.learning_rate * dWhh[i][j]
    
    for i in range(self.output_size):
      for j in range(self.hidden_size):
        self.output_layer.neurons[i].weights[j] -= self.learning_rate * dWhy[i][j]
    
    for i in range(self.hidden_size):
      self.hidden_layer.neurons[i].bias -= self.learning_rate * dbh[i]
    
    for i in range(self.output_size):
      self.output_layer.neurons[i].bias -= self.learning_rate * dby[i]
  
  def train(self, inputs, targets, epochs=100):
    """Train the RNN on inputs and targets for multiple epochs"""
    losses = []
    
    for epoch in range(epochs):
      xs, hs, ps = self.forward(inputs)
      
      loss = 0
      for t in range(len(inputs)):
        target_idx = targets[t].index(1)
        loss -= math.log(ps[t][target_idx])
        
      dWxh, dWhh, dWhy, dbh, dby = self.backward(xs, hs, ps, targets)
      
      self.update_parameters(dWxh, dWhh, dWhy, dbh, dby)
      
      losses.append(loss)
      if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
    return losses
  
  def predict(self, input_vector, prev_hidden=None):
    """Make a prediction for a single input"""
    if prev_hidden is not None:
      self.hidden_layer.prev_outputs = prev_hidden
    
    h = self.hidden_layer.forward(input_vector)
    p = self.output_layer.forward(h)
    
    return p, h
  
  def get_network_state(self):
    """Get the current state of the network for inspection"""
    state = {
      'hidden': {
        'inputs': [n.input_sum for n in self.hidden_layer.neurons],
        'outputs': self.hidden_layer.outputs
      },
      'outputs': {
        'inputs': [n.input_sum for n in self.output_layer.neurons],
        'outputs': [n.output for n in self.output_layer.neurons]
      }
    }
    return state


if __name__ == "__main__":
  vocab = ["apple", "banana", "orange", "grape"]
  idx_to_word = {i: word for i, word in enumerate(vocab)}
  word_to_idx = {word: i for i, word in enumerate(vocab)}
  vocab_size = len(vocab)
  
  def one_hot(idx, size):
    v = [0] * size
    v[idx] = 1
    return v
  
  input_sequence = [one_hot(word_to_idx[w], vocab_size) for w in vocab[:-1]]  # all but the last word
  target_sequence = [one_hot(word_to_idx[w], vocab_size) for w in vocab[1:]]  # all but the first word
  
  hidden_size = 10
  rnn = RNN(input_size=vocab_size, hidden_size=hidden_size, output_size=vocab_size, learning_rate=0.01)
  
  print(f"{'':=^50}")
  print("Initial state before training:")
  xs, hs, ps = rnn.forward(input_sequence)
  
  initial_loss = 0
  for t in range(len(input_sequence)):
    target_idx = target_sequence[t].index(1)
    initial_loss -= math.log(ps[t][target_idx])
  
  state = rnn.get_network_state()
  print(f"{'':=^50}")
  print(f"\tHidden Layer Outputs (first 2): {state['hidden']['outputs'][0]:.4f}, {state['hidden']['outputs'][1]:.4f}")
  print(f"\tOutput Layer Outputs: {[f'{p:.4f}' for p in state['outputs']['outputs']]}")
  print(f"\nInitial Loss: {initial_loss:.4f}")
  print(f"{'':=^50}")
  
  print("\nTraining network for 1000 epochs...\n")
  losses = rnn.train(input_sequence, target_sequence, epochs=1000)
  
  print(f"{'':=^50}") 
  print("Results after training:")
  xs, hs, ps = rnn.forward(input_sequence)
  
  state = rnn.get_network_state()
  print(f"{'':=^50}")
  print(f"\tHidden Layer Outputs (first 2): {state['hidden']['outputs'][0]:.4f}, {state['hidden']['outputs'][1]:.4f}")
  print(f"\tOutput Layer Outputs: {[f'{p:.4f}' for p in state['outputs']['outputs']]}")
  
  final_loss = 0
  for t in range(len(input_sequence)):
    target_idx = target_sequence[t].index(1)
    final_loss -= math.log(ps[t][target_idx])
  
  print(f"\nFinal Loss: {final_loss:.4f}")
  print(f"{'':=^50}")
  
  hidden = [0] * hidden_size
  
  print("\nTesting the model:")
  for word in vocab[:-1]:
    x = one_hot(word_to_idx[word], vocab_size)
    p, hidden = rnn.predict(x, hidden)
    
    max_idx = p.index(max(p))
    predicted_word = idx_to_word[max_idx]
    
    print(f"Input: {word}, Predicted next word: {predicted_word}")

