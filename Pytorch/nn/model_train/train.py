import torchvision
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model import * # 导入模型

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

net = Net()

# 定义损失函数和优化器
loss_function = nn.CrossEntropyLoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置参数
total_train_step = 0 # 记录总步数
total_test_step = 0
epoch = 10 # 训练轮数

# 添加用于记录loss的列表
train_loss_history = []
test_loss_history = []

for i in range(epoch):
    print(f"---------- Epoch {i+1} ----------")

    # 训练阶段
    epoch_train_loss = 0.0
    train_batches = 0
    for data in train_dataloader:
        imgs, targets = data
        outputs = net(imgs)
        loss = loss_function(outputs, targets)

        optimizer.zero_grad() # 清空梯度
        loss.backward() # 反向传播
        optimizer.step() # 更新参数

        epoch_train_loss += loss.item()
        train_batches += 1
        total_train_step += 1
        if total_train_step % 1000 == 0: # 打印每1000步的训练信息
            print(f"训练次数: {total_train_step}, Loss: {loss.item():.4f}")

    # 计算当前epoch的平均训练loss
    avg_train_loss = epoch_train_loss / train_batches
    train_loss_history.append(avg_train_loss)

    # 测试阶段
    total_test_loss = 0.0
    with torch.no_grad(): # 不计算梯度
        for data in test_dataloader:
            imgs, targets = data
            outputs = net(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_dataloader)
    test_loss_history.append(avg_test_loss)
    print(f"整体测试集上的loss: {total_test_loss:.4f}, 平均Loss: {avg_test_loss:.4f}")
    print(f"训练平均Loss: {avg_train_loss:.4f}")

# 绘制loss曲线
plt.figure(figsize=(10, 6))
epochs_range = range(1, epoch + 1)
plt.plot(epochs_range, train_loss_history, 'b-', label='训练Loss', linewidth=2)
plt.plot(epochs_range, test_loss_history, 'r-', label='测试Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练和测试Loss曲线')
plt.legend()
plt.grid(True)
plt.show()

print(f"训练完成！最终训练Loss: {train_loss_history[-1]:.4f}, 最终测试Loss: {test_loss_history[-1]:.4f}")
