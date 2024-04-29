Attograd: A Lightweight Neural Network Framework (WIP)
------------------------------------------------------

Attograd is a lightweight framework for building and training neural networks in Python. It offers a simple and intuitive API for defining neural network architectures and training them on datasets.

**Current Stage (WIP):**

This repository currently implements a basic Perceptron model. More complex architectures and functionalities are actively under development.

**Installation (For Now):**

This is a work-in-progress project, and there is no formal installation method yet. To use the current code:

1.  Clone this repository.
2.  Install any required dependencies listed in `requirements.txt` (if it exists).

**Basic Usage (Perceptron Example):**

Python

```
import attograd as ag

# Define a simple Perceptron model
class Perceptron:
  def __init__(self, in_features, out_features):
    self.weights = ag.create_variable(shape=(in_features, out_features))
    self.bias = ag.create_variable(shape=(out_features,))

  def __call__(self, x):
    return ag.dot(x, self.weights) + self.bias

# Create an instance of the Perceptron
model = Perceptron(2, 1)

# Define some training data (replace with your actual data)
inputs = [[1.0, 2.0], [3.0, 4.0]]
targets = [1.0, 0.0]

# Train the model (implementation details will be added later)
# ... (This section will be filled with training functionality)

# Make a prediction
prediction = model(inputs[0])
print(prediction)

```

Use code [with caution.](/faq#coding)

content_copy

**Contributing:**

We welcome contributions to this project! Feel free to open pull requests with new features, bug fixes, or improvements to the documentation.

**License:**

This project is licensed under the MIT License (see LICENSE file for details).

**Disclaimer:**

This is an early-stage project. The API and functionalities are subject to change in the future.
