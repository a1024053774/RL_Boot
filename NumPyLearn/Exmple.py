import numpy as np
from PIL import Image

im_1 = Image.open("cat.png")
im_2 = Image.open("Notion.jpg")
# im.show()

im1 = np.array(im_1)
im2 = np.array(im_2)
print(im1.shape)  # 获取图像的形状  (512, 512, 3)
#获取某个像素点
print(im1[100, 100])  # 获取第100行第100列的像素点 RGB值
#提取所有像素点的红色分量
red_channel = im1[:, :, 0]  # 提取红色通道
# Image.fromarray(red_channel).show() # 显示红色通道图像

#混合两张图像
# im_mix = im1 * 0.4 + im2 * 0.6
# im_mix = im_mix.astype(np.uint8)  # 转换为无符号8位整数类型
# Image.fromarray(im_mix).show()  # 显示混合后的图像

#对图片进行降采样
im_downsampled = im1[::2, ::2]  # 每隔一行和一列取一个像素点

#对图片进行翻转
im_flipped = im1[::-1, :,:]  # # 垂直翻转
#对图片进行裁剪
im_cropped = im1[100:400, 100:400, :]  # 裁剪区域为从(100, 100)到(400, 400)
Image.fromarray(im_cropped).show()  # 显示降采样后的图像

