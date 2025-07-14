import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),         # 将PIL图像转换为Tensor
])

train_set = torchvision.datasets.CIFAR10(root='./dataset', train=True, transform=dataset_transform ,download=True)
test_set = torchvision.datasets.CIFAR10(root='./dataset', train=False, transform=dataset_transform ,download=True)


writer = SummaryWriter("p10")
for i in range(10):
    img, target = train_set[i]  # 获取
    writer.add_image("test_set", img, i)

writer.close()



'''
# CIFAR-10数据集包含60000张32x32的彩色图像，分为10个类别，每个类别6000张图像。
print(test_set[0]) # 打印第一个样本的信息
print(test_set.classes)

img ,target = test_set[0] # 获取第一个样本的图像和标签
print(img)
print(target)
print(test_set.classes[target]) # 打印第一个样本的标签对应的类别名称
img.show()
'''

