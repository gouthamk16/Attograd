import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from tensor import Tensor
from loss_functions import mseLoss as mse
import numpy as np
from ann.attolayers import Linear, Sequential
from extras.netgraph import draw_dot

# np.random.seed(0)

x = Tensor([1.0, 3.0, 4.0, 2.0])
y = Tensor([5.0])

network = Sequential([
    Linear(4, 2),
    Linear(2, 1)
])

print(x, y)

epochs = 100
lr = 0.1

for epoch in range(epochs):
    output = network(x)
    # print(output)
    loss = mse(output, y)
    # print(loss)
    network.zero_grad()
    loss.backward()
    network.update(lr=lr)
    print(f"Epoch {epoch}, Loss: {loss.data}")

# draw_dot(loss)
