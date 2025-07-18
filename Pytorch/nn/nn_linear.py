import torch
import torchvision
from torch import nn

dataset = torchvision.datasets.CIFAR10('data', train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(196608, 10)  # CIFAR-10 images are 32x32 with 3 channels

    def forward(self, input):
        output = self.linear(input)
        return output

net = Net()


for data in dataloader:
    imgs, targets = data
    output = torch.flatten(imgs) # 降低维度,展平为1行
    # print("Output shape:", output.shape)
    output = net(output)
    print(output.shape)