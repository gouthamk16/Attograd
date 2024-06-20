# Implement loss functions
# Mean squared error
# Mean absolute error
# Binary crossentropy
# Categorical crossentropy
# Sparse categorical crossentropy
# KL Divergence loss
# Logits of BCE, MSE and MAE

from tensor import Tensor

# Mean squared error
def mseLoss(z, target):
    z.data = z.data.flatten()
    target = target.data.flatten()
    sum = 0
    for i in range(target.length()):
        interim_sum = z[i] - target[i]
        sum = (interim_sum ** 2) + sum  
    return sum/target.length()

# Mean absolute error
def maeLoss(z, target):
    sum = 0
    for i in range(target.length()):
        sum += abs(z[i] - target[i])
    return sum/target.length()

