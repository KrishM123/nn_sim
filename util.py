import random


class Neuron:
    def __init__(self, pos, activation):
        self.pos = pos
        self.activation = activation
    
    def set_param(self, weights):
        self.weights = weights
    
    def eval(self, inputs):
        tot = 0
        for pos in range(len(inputs)):
            tot += inputs[pos] * self.weights[pos]
        return self.activation(tot + self.weights[-1])


class Layer:
    def __init__(self, pos, size, old_size, activation):
        self.pos = pos
        self.activation = activation
        self.neurons = []
        self.size = size
        self.old_size = old_size
        for num in range(size):
            self.neurons.append(Neuron((pos, num), activation))
    
    def random_params(self):
        for num in range(self.size):
            weights = []
            for pos in range(self.old_size + 1):
                weights.append(random.randrange(0, 100) / 100)
            self.neurons[num].set_param(weights)
    
    def update_params(self, weights):
        for pos in range(self.size):
            self.neurons[pos].set_param(weights[pos])

    def eval(self, inputs):
        outputs = []
        for pos in range(self.size):
            outputs.append(self.neurons[pos].eval(inputs))
        return outputs


class Network:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []
        self.size = 0
    
    def add_layer(self, size, activation):
        if self.size == 0:
            self.layers.append(Layer(self.size, size, self.input_size, activation))
        else:
            self.layers.append(Layer(self.size, size, self.layers[self.size].size, activation))
        self.size += 1
    
    def calculate(self, inputs):
        for pos in range(len(self.layers)):
            inputs = self.layers[pos].eval(inputs)
        return inputs

    def eval(self, inputs, y_real, loss):
        return loss(y_real, self.eval(inputs))
