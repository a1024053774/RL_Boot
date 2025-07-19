import torch
import torchvision



# vgg16 = torchvision.models.vgg16(pretrained=False)
# # 保存方式1：模型结构+模型参数
# # torch.save(vgg16,"vgg16_method1.pth")

# # 保存方式2：模型参数（官方推荐）
# torch.save(vgg16.state_dict(),"vgg16_method2.pth")   # 把vgg16的状态保存为字典形式（Python中的一种数据格式）


# # 方式1 对应 保存方式1，加载模型
# model = torch.load("vgg16_method1.pth", weights_only= False)
# print(model)  # 打印出的只是模型的结构，其实它的参数也被保存下来了

# 方式2 加载模型
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))  # vgg16通过字典形式，加载状态即参数
print(vgg16)
