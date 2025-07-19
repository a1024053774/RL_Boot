import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


train_dataset = torchvision.datasets.CIFAR10(root='data', train=True, download=True,
                                             transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = Sequential( # 使用Sequential简化模型定义
            Conv2d(in_channels=3, out_channels=32, kernel_size=5,padding=2),
            MaxPool2d(kernel_size=2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )
    def forward(self,x):
        x = self.model(x)
        return x

loss = nn.CrossEntropyLoss()
net = Net()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)


loss_history = []
epochs = 10 # 比如训练10轮
for epoch in range(epochs):
    print(f"---------- Epoch {epoch+1} ----------")
    for i, data in enumerate(train_dataloader):
        imgs, targets = data
        outputs = net(imgs)
        result_loss = loss(outputs, targets)

        optimizer.zero_grad()
        result_loss.backward()
        optimizer.step()

        loss_history.append(result_loss.item())

        # 打印训练进度
        if (i + 1) % 1000 == 0:
            print(f'Iteration {i+1}, Loss: {result_loss.item():.4f}')

# 绘制loss曲线
plt.figure(figsize=(10, 6))
plt.plot(loss_history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()

# 显示最终loss值
print(f'Final Loss: {loss_history[-1]:.4f}')
