from torch.utils.data import Dataset
from PIL import Image  # 读取图片
import os  # 想要获得所有图片的地址，需要导入os（系统库）


# 创建一个class，继承Dataset类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):  # 创建初始化类，即根据这个类去创建一个实例时需要运行的函数
        # 通过索引获取图片的地址，需要先创建图片地址的list
        # self可以把其指定的变量给后面的函数使用，相当于为整个class提供全局变量
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)  # 获得图片下所有的地址

    def __getitem__(self, idx):  # idx为编号
        # 获取每一个图片
        img_name = self.img_path[idx]  # 名称
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 每张图片的相对路径
        img = Image.open(img_item_path)  # 读取图片
        label = self.label_dir
        return img, label

    def __len__(self):  # 数据集的长度
        return len(self.img_path)


# 用类创建实例
root_dir = "dataset/hymenoptera_data/train"
ants_label_dir = "ants_images"
bees_label_dir = "bees_images"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

img, label = ants_dataset[0]
img.show()  # 可视化第一张图片

# 将ants(124张)和bees(121张)两个数据集进行拼接
train_dataset = ants_dataset + bees_dataset