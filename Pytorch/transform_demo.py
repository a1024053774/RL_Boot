from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import cv2

image_path = "dataset/hymenoptera_data/train/bees_images/16838648_415acd9e3f.jpg"
img = Image.open(image_path)

writer = SummaryWriter("logs")

# 读取图片两种方法
# 方法一: 将PIL图像转换为Tensor,PIL使用的通道顺序是RGB
tensor_transform = transforms.ToTensor() #先对类型进行实例化,不能直接调用
tensor_img = tensor_transform(img)
# 方法二: 使用OpenCV读取图片为numpy.ndarray格式,OpenCV使用的通道顺序是BGR
cv_img = cv2.imread(image_path)

writer.add_image("Tensor_img", tensor_img)
writer.close()


