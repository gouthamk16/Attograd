# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage
pytest --cov=attograd tests/

# Run a single test
pytest tests/test_tensor.py::test_matmul_grad

# Lint / format
make lint
make format

# Compile CUDA shared library (requires nvcc)
# Linux/WSL:
make cuda
# Windows (nvcc on PATH):
nvcc -shared -Xcompiler -fPIC -o attograd/cuda/shared_lib/vector_ops.so attograd/cuda/vector_ops.cu
```

## Architecture

Attograd is a from-scratch autograd engine + neural network framework backed by NumPy scalars.

**Core data flow:**

1. `attograd/tensor.py` — `Tensor` is the foundation. Every op stores `_prev` (child nodes) and a `_backward` closure that accumulates `.grad`. `backward()` topologically sorts the graph and calls each closure in reverse.

2. `attograd/nn/layers.py` — `Neuron` does the scalar dot product (iterating `w` and `x`). `Linear` is a list of `Neuron`s. `Sequential` chains layers. `TanhLayer`, `SigmoidLayer`, `ReLULayer`, `Flatten` are stateless wrappers.

3. `attograd/nn/optimizers.py` — `SGD`, `Adam`, `RMSProp`. All use `.step()` to update params. Call after `.backward()`.

4. `attograd/loss_functions.py` — `mseLoss`, `maeLoss`, `bceLoss`. Take lists of prediction/target `Tensor`s, return a scalar `Tensor`.

5. `attograd/cuda/` — optional CUDA path via `ctypes` + compiled `.so`. `is_cuda_available.cuda` is the internal flag. Use `attograd.use_cuda(True/False)` to toggle. The `.so` must be compiled with `make cuda` before use.

6. `attograd/viz/netgraph.py` — `draw_dot(root)` renders the computation graph. Requires the graphviz system binaries (not just the Python package).

**Scalar-only constraint:** `exp`, `log`, `tanh`, `sigmoid`, `relu` use `math.*` and only work on 0-d (scalar) tensors. They raise `ValueError` on arrays. `sum`, `flatten`, `reshape`, `matmul` work on arrays with full grad support.

**Known incomplete areas (genuine TODOs):**

- Broadcasting — not implemented; required before scalar ops can be vectorized
- `max`, `min`, `mean` on arrays — not yet on `Tensor`
- `concatenate`, one-hot encoding — not implemented
- `BatchNorm` layer — not implemented
- Dataloader — not implemented
- MNIST / integration tests — not written

**Package layout:**
```
attograd/
  tensor.py          # Tensor + autograd engine
  loss_functions.py  # mseLoss, maeLoss, bceLoss
  nn/
    layers.py        # Neuron, Linear, Sequential, activation layers, Flatten
    optimizers.py    # SGD, Adam, RMSProp
  cuda/
    cuda_interface.py  # ctypes bridge; silent fallback if .so missing
    vector_ops.cu      # CUDA matmul kernel
  viz/
    netgraph.py      # computation graph visualizer
tests/
  test_tensor.py     # 28 tests covering tensor ops, activations, losses, optimizers, layers
examples/
  simple_network.py
```
