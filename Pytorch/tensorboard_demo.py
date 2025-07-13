from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

# writer.add_image() # 添加图像数据
# y = 2x
for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)  # 添加标量数据 第二个参数是y值，第三个参数是x值



writer.close()

#终端中启动tensorboard
#tensorboard --logdir=logs --port=6006
'''
注意相对路径
'''
#--logdir=Pytorch/logs
