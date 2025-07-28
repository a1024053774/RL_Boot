import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# from model import * # 导入模型
import time

# 准备数据集
train_data = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

# 打印数据集信息
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"Training data size: {train_data_size}") # 50000
print(f"Test data size: {test_data_size}") # 10000

# 用Dataloader加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

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

net = Net()
net = net.cuda()

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.cuda()
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置参数
total_train_step = 0 # 记录总步数
total_test_step = 0
epoch = 10 # 训练轮数

# 添加用于记录loss的列表
train_loss_history = []
test_loss_history = []
test_accuracy_history = []  # 添加准确率记录列表

start_time = time.time()

for i in range(epoch):
    print(f"---------- Epoch {i+1} ----------")

    # 训练阶段
    epoch_train_loss = 0.0
    train_batches = 0
    for data in train_dataloader:
        imgs, targets = data

        imgs = imgs.cuda()
        targets = targets.cuda()

        outputs = net(imgs)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        epoch_train_loss += loss.item()
        train_batches += 1
        total_train_step += 1
        if total_train_step % 1000 == 0: # 打印每1000步的训练信息
            end_time = time.time()
            print("用时: ", end_time - start_time)
            print(f"训练次数: {total_train_step}, Loss: {loss.item():.4f}")

    # 计算当前epoch的平均训练loss
    avg_train_loss = epoch_train_loss / train_batches
    train_loss_history.append(avg_train_loss)

    # 测试阶段
    total_test_loss = 0.0
    total_accuracy = 0
    with torch.no_grad(): # 不计算梯度
        for data in test_dataloader:
            imgs, targets = data

            imgs = imgs.cuda()
            targets = targets.cuda()

            outputs = net(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    # 计算整体测试集上的loss
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_loss_history.append(avg_test_loss)

    # 计算准确率并记录
    test_accuracy = total_accuracy / test_data_size
    test_accuracy_history.append(test_accuracy.cpu().item())

    print(f"整体测试集上的loss: {total_test_loss:.4f}, 平均测试集上的Loss: {avg_test_loss:.4f}")
    print(f"平均训练集上的Loss: {avg_train_loss:.4f}")
    print(f"整体测试集上的正确率: {test_accuracy:.4f}", total_test_step)

# 绘制loss和准确率曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 绘制loss曲线
epochs_range = range(1, epoch + 1)
ax1.plot(epochs_range, train_loss_history, 'b-', label='train_Loss', linewidth=2)
ax1.plot(epochs_range, test_loss_history, 'r-', label='test_Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Curve')
ax1.legend()
ax1.grid(True)

# 绘制准确率曲线
ax2.plot(epochs_range, test_accuracy_history, 'g-', label='test_Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Test Accuracy Curve')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

torch.save(net, "model.pth")
print(f"训练完成！最终训练Loss: {train_loss_history[-1]:.4f}, 最终测试Loss: {test_loss_history[-1]:.4f}")
print(f"最终测试准确率: {test_accuracy_history[-1]:.4f}")
