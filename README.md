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

1. Test both implementations of the weight update function in attolayers.py (line 67)
2. Linear layer gradcheck
3. Test the paramter update for Sequential layer
4. Test the Linear layer implementation on MNIST dataset for digit classification. (Record the loss and accuracy)
5. Implement one-hot-encoding, max, min, avg, dot product, matmul, reverse division and concatenate methods in the tensor class. 
6. Test the tensor-gpu class functionality (on a GPU - preferably on both RoCM and Cuda)
7. Implement Flatten and BatchNorm layers in attolayers.py
8. Implement broadcasting
9. Implement a dataloader (load the one in pytorch).

**Contributing:**

We welcome contributions to this project! Feel free to open pull requests with new features, bug fixes, or improvements to the documentation.

**License:**

This project is unlicensed.

**Disclaimer:**

This is an early-stage project. The API and functionalities are subject to change in the future.
