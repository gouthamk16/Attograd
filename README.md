# Attograd

A lightweight neural network framework with autograd built from scratch.

## Installation

```bash
git clone https://github.com/gouthamk16/attograd.git
cd attograd
pip install -e .
```

## Quick Start

```python
from attograd import Tensor, use_cuda
from attograd.nn import Linear, Sequential, TanhLayer
from attograd.loss_functions import mseLoss

x = [Tensor(2.0), Tensor(3.0), Tensor(-1.0)]
y = [Tensor(1.0)]

net = Sequential([
    Linear(3, 4, activation='tanh'),
    Linear(4, 1),
])

for epoch in range(100):
    out = net(x)
    loss = mseLoss(out, y)
    net.zero_grad()
    loss.backward()
    net.update(lr=0.01)
```

## CUDA Acceleration

CUDA support requires compiling the shared library first. You need `nvcc` (CUDA Toolkit) installed.

**Linux / WSL:**
```bash
make cuda
```

**Windows (from a terminal with nvcc on PATH):**
```bash
nvcc -shared -Xcompiler -fPIC -o attograd/cuda/shared_lib/vector_ops.so attograd/cuda/vector_ops.cu
```

Once compiled, enable GPU acceleration in your code:

```python
from attograd import use_cuda

use_cuda(True)   # use GPU
use_cuda(False)  # use CPU (default)
```

Calling `use_cuda(True)` on a machine without the compiled library raises a `RuntimeError` with instructions.

## Optimizers

```python
from attograd.nn import SGD, Adam

sgd = SGD(net.parameters(), lr=0.01)
adam = Adam(net.parameters(), lr=0.001)

loss.backward()
sgd.step()  # or adam.step()
net.zero_grad()
```

## Running Tests

```bash
pytest
```

## License

MIT
