import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10

# >tensorboard --logdir=Pytorch/nn/logs/nn_conv2d

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(), download= True)
dataloader = DataLoader(dataset, batch_size=64)

class NN_Conv2d(nn.Module):
    def __init__(self):
        super(NN_Conv2d, self).__init__()
        self.conv_1 = Conv2d(3, 6, 3, 1, 0)

    def forward(self, input):
        x = self.conv_1(input)
        return x

nnConv2d = NN_Conv2d()

writer = SummaryWriter("logs/nn_conv2d")
step = 0
for data in dataloader:
    imgs, targets = data
    output = nnConv2d(imgs)
    print("卷积前的图片形状:", imgs.shape) # torch.Size([64, 3, 32, 32])
    print("卷积后的图片形状:", output.shape) # torch.Size([64, 6, 30, 30])
    writer.add_images("input: {}".format(step), imgs, step)

    #output为6通道,先转为3通道, torch.Size([64, 6, 30, 30]) -> torch.Size([xxx, 3, 30, 30])
    # 通道数变少,batch size相应会变多
    output = torch.reshape(output, (-1, 3, 30, 30)) # -1表示自动计算batch size
    writer.add_images("output: {}".format(step), output, step)

    step += 1

writer.close()

'''
    卷积设置out_channels=6,则输出的图片通道数为6,
    卷积操作后图片的宽度和高度会减小
'''
