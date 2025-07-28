import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "../../../imgs/model_test_plane.png"
image = Image.open(image_path).convert('RGB')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])

image = transform(image)

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

model = torch.load("model.pth", weights_only= False)  # 加载模型权重
'''
weights_only的T与F的区别：
- True: 只加载权重，不加载模型结构。
- False: 加载完整的模型结构和权重。
'''
print(model)
image = torch.reshape(image, (1, 3, 32, 32))

# 用GPU训练的模型->需要将模型和验证的图形都移动到GPU上
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
image = image.to(device)

model.eval()
with torch.no_grad():
    output = model(image)
print(output)
print(output.argmax(1)) # 输出预测的类别索引

'''
如何输出索引对应的类别名称？
CIFAR-10数据集中是：
cifar10_classes = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
'''




