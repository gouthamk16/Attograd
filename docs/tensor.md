# Tensor Implementation

The `Tensor` class is the core component of the Attograd framework. It provides automatic differentiation capabilities which are essential for training neural networks.

## Basic Usage

```python
from attograd import Tensor

# Create tensors
x = Tensor([1, 2, 3])
y = Tensor([4, 5, 6])

# Operations
z = x + y  # Element-wise addition
w = x * y  # Element-wise multiplication

# Scalar operations
a = Tensor(2)
b = x * a  # Scales each element of x by 2

# Computing gradients
c = (x * x).sum()  # Sum of squares
c.backward()       # Compute gradients
print(x.grad)      # Gradient of c with respect to x
```

## Auto-differentiation

Attograd implements reverse-mode auto-differentiation, which is efficient for the types of computations used in neural networks. When you call `backward()` on a tensor, it computes the gradients of that tensor with respect to all tensors that were used to compute it.

### Example: Computing Gradients for a Simple Function

```python
from attograd import Tensor

# Function: f(x) = x^2
x = Tensor(3.0)
y = x * x

# Compute df/dx at x=3
y.backward()

# The gradient should be 2x = 2*3 = 6
print(x.grad)  # Output: 6.0
```

## Advanced Features

### Broadcasting

Attograd supports broadcasting, which allows operations between tensors of different shapes:

```python
x = Tensor([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
y = Tensor([10, 20, 30])            # Shape (3,)
z = x + y                           # Shape (2, 3)
```

### In-place Operations

For memory efficiency, some operations can be performed in-place:

```python
x = Tensor([1, 2, 3])
x.data += 1  # Increments all elements of x by 1
```

Note that in-place operations may break the computation graph for auto-differentiation.

## API Reference

See the [API Reference](api_reference.md#tensor) for a complete list of methods and attributes for the `Tensor` class.
