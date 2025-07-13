import numpy as np  # 导入NumPy库，用于数值计算
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图


# 条形图示例
x = np.array(['A', 'B', 'C', 'D', 'E'])  # 定义x轴标签
y = np.array([3, 8, 1, 10, 5])  # 定义y轴数据
plt.bar(x, y, color='skyblue', width=0.4)  # 绘制竖直条形图，设置颜色和宽度
plt.show()  # 显示图形

# 水平条形图示例
plt.barh(x, y, color='skyblue', height=0.4)  # 绘制水平条形图，设置颜色和高度
plt.show()  # 显示图形

# 直方图示例
#一共250个数据点，平均值170，标准差10
x = np.random.normal(170, 10, 250)  # 生成250个均值为170、标准差为10的正态分布数据
print(x)  # 打印生成的数据
plt.hist(x, color='skyblue')  # 绘制直方图，设置颜色
plt.show()  # 显示图形

# 散点图示例
x = np.random.normal(170, 10, 250)  # 生成x轴数据
y = np.random.normal(170, 10, 250)  # 生成y轴数据
plt.scatter(x, y, color='skyblue')  # 绘制散点图，设置颜色
plt.title('Scatter Plot Example')  # 设置图表标题
plt.xlabel('X-axis')  # 设置x轴标签
plt.ylabel('Y-axis')  # 设置y轴标签
plt.show()  # 显示图形

# 饼图示例
labels = ['Apple', 'Banana', 'Cherry', 'Date']  # 定义饼图标签
sizes = [15, 30, 45, 10]  # 定义每个扇区的大小
explode = (0.1, 0, 0, 0)  # 突出显示第一个扇区
plt.pie(sizes, explode=explode, labels=labels, shadow=True, autopct='%1.1f%%')  # 绘制饼图，设置突出、标签、阴影和百分比显示
plt.axis('equal')  # 确保饼图为圆形
plt.legend(title = "Four") # 添加图例并设置标题
plt.show()  # 显示图形
