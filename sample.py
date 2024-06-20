from tensor import Tensor
from loss_functions import mseLoss as mse

target = Tensor.random((1, 10))
y_pred = Tensor.random((1, 10))

loss = mse(target, y_pred)

print(loss)

# a = Tensor.ones((32))
# b = Tensor.ones((32))

# print(a+b)