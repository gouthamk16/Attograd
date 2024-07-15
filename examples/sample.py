import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from tensor import Tensor
from loss_functions import mseLoss as mse
from ann.attolayers import Linear, Sequential
from extras.netgraph import draw_dot
import numpy as np

# Set a seed
np.random.seed(0)

# Define a simple neural network to test the Linear layer implementation
network = Sequential([
    Linear(3, 4),
    Linear(4, 1)
])

x = Tensor([
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
])
y = Tensor([1.0, -1.0, -1.0, 1.0])

print(x, y)

# Creating a single training loop quivalent of the above steps
# Output before training
# print(network(x))
def train(network, x, y, lr=0.1, epochs=1000):
    for i in range(epochs):

        # output = network(x)
        output = [network(xi) for xi in x]
        # print(output)
        # print(y)
        loss = mse(output, y)
        # network.zero_grad()
        loss.backward()
        network.update(lr=lr)
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss.data}")

# Train the network
train(network, x, y, lr=0.1, epochs=1000)
# Final output after 1000 epochs
# print(network(x))

# Visualizing the network
# draw_dot(loss)