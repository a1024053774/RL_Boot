import torch
from torch import nn
from torch.nn import L1Loss

#Input dtype must be either a floating point or complex dtype
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss_MAE = L1Loss()
result_MAE = loss_MAE(inputs, targets)


loss_MSE = nn.MSELoss()
result_MSE = loss_MSE(inputs, targets)

print("MAE Loss:", result_MAE)
print("MSE Loss:", result_MSE)


# CROSSENTROPY LOSS
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1]) # Class index for the target
x = torch.reshape(x, (1, 3))  # batch size of 1, 3 classes
loss_CE = nn.CrossEntropyLoss()
result_CE = loss_CE(x, y)
print("CE Loss:", result_CE)
