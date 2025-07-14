import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform= torchvision.transforms.ToTensor() )

'''
dataset：只有dataset没有默认值，只需要将之前自定义的dataset实例化，再放到dataloader中即可
batch_size：每个batch的大小，默认值为1
shuffle：打乱与否，值为True的话两次顺序不一样。默认为False
num_workers：加载数据的线程数，默认值为0，表示在主线程中加载数据。设置为大于0的值可以加快数据加载速度，但需要注意线程安全问题。
drop_last：如果数据集的大小不能被batch_size整除，最后一个batch是否丢弃。默认为False，即保留最后一个batch。
'''
test_loader = DataLoader(dataset = test_data, batch_size= 4, shuffle= True, num_workers= 0, drop_last= False)

# 测试数据集中第一张图片和标签
img, target = test_data[0]
print(img.shape)
print(target)

# 写入tensorboard是四个图片一组
writer = SummaryWriter("dataloader")
for epoch in range(2): # 两轮
    step = 0
    for data in test_loader:
        imgs,targets = data
        # print(imgs.shape) #torch.Size([4, 3, 32, 32]) 4张图片，每张图片3个通道，32x32像素
        # print(targets) # tensor([5, 4, 7, 6]) 对应的标签
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1

writer.close()



