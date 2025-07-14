import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # 禁用OneDNN优化
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/images/TokyoSunrise_ZH-CN0091906710_UHD.jpg")

# ToTensor
trans_toTensor = transforms.ToTensor()
img_tensor = trans_toTensor(img)
writer.add_image("ToTensor", img_tensor)


# Normalize 只可以对Tensor进行归一化, 加快收敛速度，减少训练时间
'''
# 归一化的作用是将数据缩放到一个特定的范围内，通常是0到1之间，或者是-1到1之间。
# 图片RGB三个信道，将每个信道中的输入进行归一化
# output[channel] = (input[channel] - mean[channel]) / std[channel]
'''
print(img_tensor[0][0][0]) # 打印第一个像素点的第一个通道的值
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])   # 归一化的均值和标准差
img_norm = trans_norm(img_tensor)  # 归一化后仍然是Tensor
print(img_norm[0][0][0]) # 打印第一个像素点的第一个通道的值
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))  # 将图片缩放到512x512
# img PIL ->resize ->img_resize PIL
img_resize = trans_resize(img) # resize方法返回的是PIL图像
# img_resize PIL -> toTensor -> img_resize_tensor Tensor
img_resize_tensor = trans_toTensor(img_resize) # 将PIL图像转换为Tensor
writer.add_image("Resize", img_resize_tensor, 0)
print(img_resize_tensor)

# Compose 组合变换
trans_resize_2 = transforms.Resize(512) #等比缩放
# img PIL -> resize ->img_resize PIL -> toTensor -> img_resize_tensor Tensor
trans_compose = transforms.Compose([trans_resize_2, trans_toTensor])
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop 随机裁剪
trans_random = transforms.RandomCrop((512, 1000))
trans_compose_2 = transforms.Compose([trans_random, trans_toTensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop_500*1000", img_crop, i)

writer.close()