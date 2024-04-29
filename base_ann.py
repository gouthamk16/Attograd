#1. Creating a perceptron for NAND logic function - DONE
#2. Create a single layer neuron (i input, 1 output) to perform forward propagation (Linear)
#3. Create a aingle layer neuron (1 input, 1 oputput) to perform forward propagation Sigmoid)
#4. Create a aingle layer neuron (1 input, 1 oputput) to perform forward and backward propagation using gradient descent (sigmoid)

import numpy as np
import matplotlib.pyplot as plt
import random
from colorama import Fore

# Defining the function
def activation(sum):
    if (sum>0):
        return 1
    else:
        return 0
# Creating a class for the mp neuron
class MPNeuron():
    def __init__(self):
        self.weights = random.sample(range(-4, 4), 2)
        print(f"Inniital Weights: {self.weights}")
        # self.bias = random.randint(-5, 5)
        # self.weights = [-2, -2]
        self.bias = 3
        print(f"Inniital Bias: {self.bias}")
        self.sum = 0
        self.output = 0
        self.lr = 0.01
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
                # self.bias = self.lr * (target-self.output)

def mseLoss(z, target):
    sum = 0
    for i in range(len(target)):
        sum += abs(z[i] - target[i])
    return sum/len(target)

# Defining the data (NAND truth table)
x = [[0, 0],
     [0, 1],
     [1, 0],
     [1, 1]]

target = [1, 1, 1, 0]
neuron = MPNeuron()
epochs = 10
loss_history = []
optimizer = "SGD"

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
    loss = mseLoss(output_list, target)
    loss_history.append(loss)
    print(f"Epoch: {epoch+1}/{epochs} | Loss: {loss}")
    print(output_list)
    if output_list == target:
        break

# Lets plot the pochs and the loss
def plotGrad(loss):
    plt.plot(loss)
    # plt.show()

plotGrad(loss_history)

# Making the neuron learn by itself - loss function and gradient descent 
