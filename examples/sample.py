from tensor import Tensor
from loss_functions import mseLoss as mse
import numpy as np
from ann.attolayers import Linear, Sequential
from extras.netgraph import draw_dot
from cuda.cuda_interface import needCuda

# np.random.seed(0)

## specify if you want to use cuda
needCuda.cuda = False

print(needCuda.cuda)

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
