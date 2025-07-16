import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#最大池化的作用是对输入的特征图进行下采样，减少特征图的尺寸，同时保留重要的特征信息。

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                        transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size= 64)

# input = torch.tensor([[1,2,0,3,1],
#                       [0,1,2,3,1],
#                       [1,2,1,0,0],
#                       [5,2,3,1,1],
#                       [2,1,0,1,1]], dtype=torch.float32)

# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool_1 = MaxPool2d(kernel_size=3, ceil_mode=False)
        # self.maxpool_2 = MaxPool2d(kernel_size=3, ceil_mode=False)

    # def forward(self, input_1, input_2):
    #     output_1 = self.maxpool_1(input)
    #     output_2 = self.maxpool_2(input)
    #     return output_1, output_2
    def forward(self, input):
        output = self.maxpool_1(input)
        return output

net = Net()

writer = SummaryWriter("logs/maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = net(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()


# output = net(input, input)
# print(output[0])
'''
tensor([[[[2., 3.],
          [5., 1.]]]])
'''
# print(output[1])
# tensor([[[[2.]]]])

'''
ceil_mode=True时，若池化核大小不能满足,会向上取整，类似padding的效果
ceil_mode=False时，若池化核大小不能满足时,会跳过
'''