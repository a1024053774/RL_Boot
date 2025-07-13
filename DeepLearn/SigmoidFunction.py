import numpy as np

# 1、使用np.exp()实现sigmoid function
#np.exp()是NumPy库中的一个函数，用于计算e的指数幂。它可以接受一个数组作为输入，并对数组中的每个元素计算e的指数幂。

def basic_sigmoid(x):
    """
    计算sigmoid函数
    """
    ### 开始
    s = 1 / (1 + np.exp(-x))
    ### 结束

    return s



# 2、实现Sigmoid gradient（梯度）
#完成sigmoid的梯度函数，用它去计算sigmoid相对于其输入x的梯度

def sigmoid_derivative(x):
    """
    计算sigmoid function函数相对于其输入x的梯度（也称为斜率或者导数）.
    """

    ### 开始
    ds = basic_sigmoid(x) * (1 - basic_sigmoid(x))
    ### 结束

    return ds

# 测试基本的sigmoid函数
s = basic_sigmoid(np.array([1, 2, 3]))
print(s)

# 测试sigmoid梯度函数
ds = sigmoid_derivative(np.array([1, 2, 3]))
print(ds)