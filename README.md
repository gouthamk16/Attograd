Attograd: A Lightweight Neural Network Framework (WIP)
------------------------------------------------------

Attograd is a lightweight framework for building and training neural networks in Python. It offers a simple and intuitive API for defining neural network architectures and training them on datasets.

**Current Stage (WIP):**

This repository currently implements a basic Perceptron model. More complex architectures and functionalities are actively under development. CUDA support is being developed and can be tested on a gpu (Make sure you have cuda toolkit installed on your device).

**Installation (For Now):**

This is a work-in-progress project, and there is no formal installation method yet. To use the current code:

1.  Clone this repository.
2.  Install the required dependencies listed in `docs/requirements.txt` using:
    ```
    pip install -r requirements.txt
    ```
3. Run the `sample.py` in the `examples` folder to test the current implementation.

Use code with caution

**TO-DO** (1-Highest Priority)

1. Test both implementations of the weight update function in attolayers.py (line 67)
2. Linear layer gradcheck
3. Test the Linear layer implementation on MNIST dataset for digit classification. (Record the loss and accuracy)
4. Performance test and optimization for CUDA implementation of matmul in `cuda/vector_ops.cu`.
5. Test the paramter update for Sequential layer
6. Implement one-hot-encoding, max, min, avg, dot product, matmul, reverse division and concatenate methods in the tensor class. 
7. Implement the cuda support for the tensor class (matmul and data to deviceMemory). 
8. Implement Flatten and BatchNorm layers in attolayers.py
9. Implement broadcasting
10. Implement a dataloader (load the one in pytorch).

**Note Regarding CUDA implementation:**
Matrix Multiplication has been implemented in CUDA - available under `cuda/vector_ops.cu`.
Python interface has been provided to use the CUDA implementation using ctypes. 
Shared library has been created under `cuda/shared_lib/vector_ops.so`.

**Contributing:**

We welcome contributions to this project! Feel free to open pull requests with new features, bug fixes, or improvements to the documentation.

**License:**

This project is unlicensed.

**Disclaimer:**

This is an early-stage project. The functionalities are subject to change in the future.
