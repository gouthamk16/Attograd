"""
Attograd: A Lightweight Neural Network Framework
===============================================

Attograd is a lightweight framework for building and training neural networks in Python.
It offers a simple and intuitive API for defining neural network architectures and training them on datasets.
"""

from .tensor import Tensor
from .loss_functions import mseLoss, maeLoss, bceLoss
from .cuda import use_cuda

from . import nn
from . import cuda
from . import viz

__version__ = "0.1.0"
__author__ = "Goutham"
