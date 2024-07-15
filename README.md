Attograd: A Lightweight Neural Network Framework (WIP)
------------------------------------------------------

Attograd is a lightweight framework for building and training neural networks in Python. It offers a simple and intuitive API for defining neural network architectures and training them on datasets.

**Current Stage (WIP):**

This repository currently implements a basic Perceptron model. More complex architectures and functionalities are actively under development.

**Installation (For Now):**

This is a work-in-progress project, and there is no formal installation method yet. To use the current code:

1.  Clone this repository.
2.  Install the required dependencies listed in `docs/requirements.txt` using:
    ```
    pip install -r requirements.txt
    ```

Use code with caution

**TO-DO** (1-Highest Priority)

1. Gradcheck for the tensor functions (for all functions with grad)
2. Test both implementations of the weight update function in attolayers.py
3. Implement one-hot-encoding, max, min, avg and dot product methods in the tensor class.
4. Implement matmul in Tensor
5. Test backprop for Linear layer (gradcheck)
6. Debug the linear layer backpropagation - currently not working
7. Test the tensor-gpu class functionality (on a GPU - preferably on both RoCM and Cuda)
8. Implement reverse division
9. Implement broadcasting

**Contributing:**

We welcome contributions to this project! Feel free to open pull requests with new features, bug fixes, or improvements to the documentation.

**License:**

This project is unlicensed.

**Disclaimer:**

This is an early-stage project. The API and functionalities are subject to change in the future.
