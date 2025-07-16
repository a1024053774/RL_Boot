import torch
from torch import nn


class nn_New(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


nn_new = nn_New()
x = torch.tensor(1.0)
output = nn_new(x)
print(output)
