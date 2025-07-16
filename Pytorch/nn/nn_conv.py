import torch
import torch.nn.functional as F

'''
torch.nn.Conv2d
in_channels, # 输入通道数
out_channels, # 输出通道数
kernel_size, # 卷积核大小，可以是单个整数或一个元组
stride=1, # 步长，默认为1
padding=0, # 填充，默认为0
dilation=1, # 膨胀系数，默认为1
groups=1, # 分组卷积，默认为1
bias=True, # 是否使用偏置，默认为True
padding_mode='zeros',  # 填充模式，默认为'zeros'，可以是'reflect', 'replicate', 'circular'等
device=None, # 设备类型，默认为None
dtype=None # 数据类型，默认为None

'''

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])
print(input.shape, kernel.shape) # 输出结果 torch.Size([5, 5]) torch.Size([3, 3])
# 结果不是4D张量，Conv2d需要4D张量作为输入。

# Reshape to 4D tensor for Conv2d

# Parameters(batch_size, in_channels, height, width)
input = torch.reshape(input, (1, 1, 5, 5))

# Parameters(out_channels, in_channels, kernel_height, kernel_width)
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input, kernel, stride=1)
print(output)

# 步长为2
output_2 = F.conv2d(input, kernel, stride=2)
print(output_2)

#padding一般为kernel_size // 2 向下取整
output_3 = F.conv2d(input, kernel, stride=1, padding='same')
print(output_3)


