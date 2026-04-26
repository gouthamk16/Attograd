# Attograd Framework Walkthrough

This guide will help you get started with the Attograd framework and show you how to build simple neural networks.

## Installation

First, make sure you have installed the Attograd framework:

```bash
pip install attograd
```

Or if you're working from source:

```bash
pip install -e .
```

## Basic Tensor Operations

The core of Attograd is the `Tensor` class, which represents a multi-dimensional array with automatic differentiation capabilities.

```python
from attograd import Tensor

# Create tensors
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

# Basic operations
c = a + b    # Addition: [5, 7, 9]
d = a * b    # Element-wise multiplication: [4, 10, 18]
e = a @ b    # Dot product: 32 (1*4 + 2*5 + 3*6)

# Scalar operations
f = a * 2    # [2, 4, 6]
g = a + 5    # [6, 7, 8]

# Functions
h = a.tanh() # Apply tanh to each element
```

## Automatic Differentiation

Attograd supports automatic differentiation, which is crucial for training neural networks:

```python
# Define a simple expression: y = x^2
x = Tensor(3.0)
y = x * x

# Compute gradient dy/dx at x=3
y.backward()

# The gradient should be 2x = 2*3 = 6
print(x.grad)  # Output: 6.0
```

## Building a Neural Network

Let's build a simple neural network for binary classification:

```python
from attograd import Tensor
from attograd.nn import Linear, TanhLayer

# Define the input data
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = Tensor([[0], [1], [1], [0]])  # XOR function

# Define a simple neural network
class XORNetwork:
    def __init__(self):
        self.linear1 = Linear(2, 3)
        self.tanh = TanhLayer()
        self.linear2 = Linear(3, 1)
    
    def __call__(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)
        return x
    
    def parameters(self):
        return self.linear1.parameters() + self.linear2.parameters()

# Create and train the network
from attograd.loss_functions import mseLoss

model = XORNetwork()
learning_rate = 0.1

# Training loop
for epoch in range(1000):
    # Forward pass
    predictions = model(X)
    loss = mseLoss(predictions, Y)
    
    # Reset gradients
    for p in model.parameters():
        p.grad = 0
    
    # Backward pass
    loss.backward()
    
    # Update parameters
    for p in model.parameters():
        p.data -= learning_rate * p.grad
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data.item()}")

# Test the model
print("\nPredictions after training:")
predictions = model(X)
for x, y_true, y_pred in zip(X.data, Y.data, predictions.data):
    print(f"Input: {x}, Target: {y_true}, Prediction: {y_pred}")
```

## Using CUDA Acceleration

If you have a CUDA-capable GPU, you can use it to accelerate computation:

```python
from attograd.cuda import is_cuda_available

if is_cuda_available.cuda:
    print("CUDA acceleration is available!")
    # Your code will automatically use GPU when available
else:
    print("CUDA is not available, using CPU instead.")
```

## Visualizing the Computation Graph

Attograd allows you to visualize the computation graph:

```python
from attograd import Tensor
from attograd.viz import draw_dot

# Create a computation graph
x = Tensor(2.0, label='x')
y = Tensor(3.0, label='y')
z = x * y + y.tanh()
z.label = 'z'

# Visualize the computation graph
draw_dot(z)
```

## What's Next?

- Check out the examples in the `examples/` directory for more complex models
- Read the API documentation in the `docs/` directory
- Contribute to the Attograd project by reporting issues or submitting pull requests

Happy building with Attograd!
