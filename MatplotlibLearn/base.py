import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# print(matplotlib.__version__)

#从(0,0)到(6,250)的线段
xpoints = np.array([0, 6])
ypoints = np.array([0, 250])
plt.plot(xpoints, ypoints)
plt.show()

#无线绘图
plt.plot(xpoints, ypoints, 'o')
plt.show()

#多点,从(1,3) 到(2,8) 到(6,1) 到(8,10)
xpoints = np.array([1, 2, 6, 8])
ypoints = np.array([3, 8, 1, 10])
plt.plot(xpoints, ypoints)
plt.show()

#默认X点
ypoints = np.array([3, 8, 1, 10, 5, 7])
plt.plot(ypoints)
plt.show()

#使用标记
ypoints = np.array([3, 8, 1, 10, 5, 7])
plt.plot(ypoints, marker='o', markerfacecolor='red', markersize=10, color='skyblue', linewidth=4)
plt.title('Marker Example')
plt.show()

plt.plot(ypoints, 'o:r') # 'o'表示圆形标记，':r'表示红色虚线
plt.show()
