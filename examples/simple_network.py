"""
Basic neural network example using Attograd.
This example creates a simple feed-forward neural network and trains it.
"""

from attograd import Tensor, use_cuda
from attograd.loss_functions import mseLoss as mse
from attograd.nn.layers import Linear, Sequential

def main():
    use_cuda(False)  # set to True to use GPU

    # Create input and target tensors
    x = Tensor([1.0, 3.0, 4.0, 2.0])
    y = Tensor([5.0])

    # Create a simple neural network
    network = Sequential([
        Linear(4, 2),
        Linear(2, 1)
    ])

    print(f"Input: {x}")
    print(f"Target: {y}")

    # Training parameters
    epochs = 100
    lr = 0.1

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        output = network(x)
        loss = mse(output, y)
        
        # Backward pass
        network.zero_grad()
        loss.backward()
        
        # Update parameters
        network.update(lr=lr)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.data}")

    print(f"Final prediction: {network(x)}")
    print(f"Final loss: {mse(network(x), y).data}")

if __name__ == "__main__":
    main()
