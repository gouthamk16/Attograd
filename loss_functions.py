# Implement loss functions
# Mean squared error
# Mean absolute error
# Binary crossentropy
# Categorical crossentropy
# Sparse categorical crossentropy
# KL Divergence loss
# Logits of BCE, MSE and MAE

from tensor import Tensor

# Mean Squared Error
def mseLoss(z, target):
    return sum([(yout - ygt)**2 for ygt, yout in zip(target, z)])

# Mean Absolute Error
def maeLoss(z, target):
    return sum([abs(yout - ygt) for ygt, yout in zip(target, z)])

# Binary Cross Entropy
def bceLoss(z, target):
    return -sum([ygt * Tensor.log(yout) + (1 - ygt) * Tensor.log(1 - yout) for ygt, yout in zip(target, z)])

