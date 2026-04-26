"""
CUDA interface for GPU-accelerated operations.
Python bridge to the compiled CUDA shared library via ctypes.
"""

import ctypes
import numpy as np
import os

_lib = None
cuda_available = False

try:
    _lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared_lib', 'vector_ops.so')
    _lib = ctypes.CDLL(_lib_path)
    _lib.matMul.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    cuda_available = True
except (OSError, AttributeError):
    pass


class is_cuda_available:
    """Feature flag for CUDA acceleration. Set .cuda = False to force CPU."""
    cuda = cuda_available


def matrixMultiply(A, B):
    """
    Matrix multiply A @ B using CUDA if available, otherwise numpy.

    Args:
        A: 2D numpy float32 array
        B: 2D numpy float32 array

    Returns:
        Result matrix as numpy float32 array
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    if A.ndim == 1:
        A = A.reshape(1, -1)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    m, n = A.shape
    j, k = B.shape

    if n != j:
        if k == m:
            A, B = B, A
            m, n = A.shape
            j, k = B.shape
        else:
            raise ValueError(f"Incompatible shapes for matmul: {A.shape} @ {B.shape}")

    C = np.zeros((m, k), dtype=np.float32)

    if cuda_available and is_cuda_available.cuda:
        _lib.matMul(A, B, C, m, j, n, k)
    else:
        C = np.matmul(A, B).astype(np.float32)

    return C
