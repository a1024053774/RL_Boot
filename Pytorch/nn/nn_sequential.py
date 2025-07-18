import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        '''
        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2)  # 第一个卷积
        self.maxpool1 = MaxPool2d(kernel_size=2)   #池化
        self.conv2 = Conv2d(32,32,5,padding=2)  #维持尺寸不变，所以padding仍为2
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32,64,5,padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()  #展平为64x4x4=1024个数据
        # 经过两个线性层：第一个线性层（1024为in_features，64为out_features)、第二个线性层（64为in_features，10为out_features)
        self.linear1 = Linear(1024,64)
        self.linear2 = Linear(64,10)
        #10为10个类别，若预测的是概率，则取最大概率对应的类别，为该图片网络预测到的类别
        '''
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

net = Net()
print(net)

input = torch.ones(64, 3, 32, 32) # batch_size=64, channel=3, height=32, width=32
output = net(input)
print(output.shape)

writer = SummaryWriter("logs/nn_sequential")
writer.add_graph(net, input)
writer.close()