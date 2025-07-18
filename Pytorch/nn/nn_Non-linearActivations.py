import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#tensorboard --logdir = Pytorch/nn/logs/nn_Non-linearActivations

input = torch.Tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size= 64)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # output = self.relu(input)
        output = self.sigmoid(input)
        return output


net = Net()
writer = SummaryWriter("logs/nn_Non-linearActivations")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = net(imgs)
    writer.add_images("output", output, global_step=step)
    step += 1

writer.close()
