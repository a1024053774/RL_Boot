import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader


dataset = torchvision.datasets.CIFAR10(root='data', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset ,batch_size = 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = Sequential( # 使用Sequential简化模型定义
            Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        x = self.model(x)
        return x

loss = nn.CrossEntropyLoss()
net = Net()
for data in dataloader:
    imgs, targets = data
    outputs = net(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward() #


