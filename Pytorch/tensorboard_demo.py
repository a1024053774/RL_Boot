from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
import torch

# TensorBoard是一个可视化工具，可以帮助我们监控和调试PyTorch模型的训练过程。

#终端中启动tensorboard
#tensorboard --logdir=logs --port=6006
'''
注意相对路径
'''
#--logdir=Pytorch/logs

writer = SummaryWriter("logs")
image_path = "dataset/hymenoptera_data/train/bees_images/16838648_415acd9e3f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL) # 将PIL图像转换为NumPy数组
print(type(img_array))
print(img_array.shape) #默认是HWC格式（高度、宽度、通道数）

writer.add_image("test", img_array, 2, dataformats='HWC') # 添加图像数据

# # y = 2x
# for i in range(100):
#     writer.add_scalar("y=2x", 2*i, i)  # 添加标量数据 第二个参数是y值，第三个参数是x值

writer.close()




