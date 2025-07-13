import numpy as np
import time
a = np.random.rand(100000)  # 生成一个包含100万个随机数的数组
b = np.random.rand(100000)  # 生成另一个包含100万个随机数的数组

#方法一,for循环
c = 0
start_ = time.time()
for i in range(100000):
    c += a[i] * b[i]
end_ = time.time()
print("for循环计算所用的时间%s " % str(1000*(end_ - start_)) + "ms")



#方法二,使用NumPy的向量化操作
start = time.time()
d = np.dot(a, b)
end = time.time()
print("计算所用的时间%s " % str(1000*(end - start)) + "ms")
