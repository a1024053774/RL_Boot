import torch
from torch import nn

# 搭建神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 ,32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64), # 64个输出,4*4的特征图经过展平后为4*4*64 = 2^10 = 1024 个输入特征
            nn.Linear(64, 10) # 10个输出
        )
    def forward(self, x):
        return self.model(x)

# 测试网络结构
if __name__ == '__main__':
    net = Net()
    print(net)

    input = torch.ones(64, 3, 32, 32) # batch_size=64, channel=3, height=32, width=32
    output = net(input)
    print(output.shape)  # 输出形状应为 (64, 10)