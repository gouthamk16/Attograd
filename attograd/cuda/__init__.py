"""
CUDA acceleration for Attograd framework.
"""

from .cuda_interface import matrixMultiply, is_cuda_available, cuda_available


def use_cuda(enabled: bool):
    """Enable or disable CUDA acceleration. No-op if CUDA is not available."""
    if enabled and not cuda_available:
        raise RuntimeError("CUDA is not available on this system. Run 'make cuda' to compile the shared library.")
    is_cuda_available.cuda = enabled
