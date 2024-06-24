from tensor import Tensor
from loss_functions import mseLoss as mse
from ann.attolayers import Linear, Sequential

x = Tensor([1, 2, 3, 4, 5])
y = Tensor([5])

model = Sequential(
    [Linear(5, 1)]
)

out = model(x)
print(out)

## Training