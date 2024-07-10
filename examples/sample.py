import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from tensor import Tensor
from loss_functions import mseLoss as mse
from ann.attolayers import Linear, Sequential
from extras.netgraph import draw_dot
import numpy as np

# x = [1, 2, 3, 4, 5]
# y = [5]


# model = Linear([4, 4], 5, 1)

# out = model(x)
# print(out)
# loss = sum((yout - ygt)**2 for ygt, yout in zip(y, out)) / len(y)
# print(loss)
# loss.backward()
# # out.backward()
# print(model.layers[0].neurons[0].w[0].grad)
# Set a seed
np.random.seed(0)

network = Sequential([
    Linear(3, 4, activation="tanh"),
    Linear(4, 1, activation="sigmoid")
])

# Input tensor
x = Tensor([1, 2, 3])
y = Tensor([5])

# Forward pass
output = network(x)
print(output)
loss = mse(output, y)
print(loss)
loss.backward()
# Check the gradient of one of the weights
print(network.layers[0].neurons[0].w[0].grad)
print(network.parameters())   
# draw_dot(loss)