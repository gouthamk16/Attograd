#1. Creating a perceptron for NAND logic function - DONE
#2. Create a two layer neuron to perform forward propagation (Linear)
#3. Create a two layer neuron to perform forward propagation Sigmoid)
#4. Create a two layer neuron to perform forward and backward propagation using gradient descent (sigmoid)

## Perceptron for NAND function
import numpy as np
import matplotlib.pyplot as plt
import random
from colorama import Fore
from loss_functions import mseLoss, maeLoss

# Defining the function
def activation(sum):
    if (sum>0):
        return 1
    else:
        return 0
# Creating a class for the mp neuron
class MPNeuron():
    def __init__(self, weights_range, bias, lr):
        self.weights = random.sample(weights_range, 2)
        print(f"Inniital Weights: {self.weights}")
        # self.bias = random.randint(-5, 5)
        # self.weights = [-2, -2]
        self.bias = bias
        print(f"Inniital Bias: {self.bias}")
        self.sum = 0
        self.output = 0
        self.lr = lr
    def activation(self):
        if (self.sum>0):
            return 1
        else:
            return 0
    def grad(self, target, x, optimizer):
        if optimizer == "SGD":
            for x1, x2 in x:
                self.weights[0] = self.weights[0] + self.lr * (target-self.output) * x1
                self.weights[1] = self.weights[1] + self.lr * (target-self.output) * x2
                self.bias = self.lr * (target-self.output)

# Defining the input data (NAND truth table)
x = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

target = [1, 1, 1, 0]
neuron = MPNeuron(weights_range=range(-4, 4), bias=3, lr=0.01) # Weights range : range within which the inital weights are initialized.
epochs = 100
loss_history = []
optimizer = "SGD"

# Training the perceptron (perceptron logic training)
for epoch in range(epochs):
    output_list = []
    i = 0
    for x1, x2 in x:
        neuron.sum = x1*neuron.weights[0] + x2*neuron.weights[1]
        neuron.sum += neuron.bias
        neuron.output = neuron.activation()
        output_list.append(neuron.output)
        neuron.grad(target[i], x, optimizer)
        i = i+1
    loss = maeLoss(output_list, target)
    loss_history.append(loss)
    print(f"Epoch: {epoch+1}/{epochs} | Loss: {loss}")
    print(output_list)
    if output_list == target:
        break

# Lets plot the pochs and the loss
def plotGrad(loss):
    plt.plot(loss)
    plt.show()

# plotGrad(loss_history)

# Making the neuron learn by itself - loss function and gradient descent 
