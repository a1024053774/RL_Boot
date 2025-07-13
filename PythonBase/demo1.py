import sys
print("hello world!")
print('Let\'s go') #\转义字符
#input("\n请输入内容\n按下enter后退出") #获取输入，按下enter后退出

x="hello"
y="world"

#换行输出
print(x)
print(y)
print('-------')
#在一行输出
print(x,y)

print ('命令行参数为:')
for i in sys.argv:
    print (i)
print ('\n python 路径为',sys.path)

#数据类型
a = 100  # 整数类型
b = 100.0  # 浮点数类型
c = "Hello, World!" # 字符串类型
d= True # 布尔类型
print(a);print(b);print(c);print(d)

#多变量赋值
#a = b = c = 100
aa, bb, cc = 1, 2.2, "Hello" # 多变量赋值
print(aa,bb,cc)
print(type(aa),type(bb),type(cc)) # 输出变量类型

#---------------不可变数据类型-------------------
#Number（数字类型）
e, f, g, h = 20, 5.5, True, 4+3j
print(e, f, g, h)
print(type(e), type(f), type(g), type(h))  # 输出变量类型
#String（字符串类型）
str = 'LuckyEeEe'
print(str) #输出LuckyEeEe
print(str[-1]) #输出e
print(str[0:-1]) #输出LuckyEeE
print(str[2:]) #输出ckyEeEe
print(str * 2) #输出LuckyEeEeLuckyEeEe
print(str + " is a good") #输出LuckyEeEe is a good
#bool(布尔类型)
print(True) # 输出True
print(False > 1) # 输出False
print(float(True))  # 输出1
print(bool(0)) # 输出False
print(bool(1)) # 输出True

#---------------可变数据类型-------------------
#List（列表类型）
list1 = [1, 2.2, 'hello', True]
list2 = [3, 4.4, 'world', False]
print(list1[1:-1]) # 输出 [2.2, 'hello']
print(list2[0]) # 输出 3
print(list2[1:]) # 输出 [4.4, 'world', False]
print(list1 + list2) # 输出 [1, 2.2, 'hello', True, 3, 4.4, 'world', False]
#list值可变
list1[0] = 100 # 修改第一个元素为100
list1[1:3] = [200, 300, 400]  #在第二和第三个位置上插入了3个元素,则第四个位置后移一位,列表变为 [100, 200, 300, 400, True]
print(list1)
print(list2[0:4:2]) # 输出 [3, 'world']，步长为2

#Set（集合类型）
sites = {'Google', 'Taobao', 'Runoob', 'Facebook', 'Zhihu', 'Baidu'}
print(sites)   # 输出集合，重复的元素被自动去掉

# 成员测试
if 'Runoob' in sites :
    print('Runoob 在集合中')
else :
    print('Runoob 不在集合中')

# set可以进行集合运算
a = set('abracadabra')
b = set('alacazam')
print(a)
print(a - b)     # a 和 b 的差集
print(a | b)     # a 和 b 的并集
print(a & b)     # a 和 b 的交集
print(a ^ b)     # a 和 b 中不同时存在的元素

#Dictionary（字典类型）
dict1 = {'name': 'Lucky', 'age': 18}
dict2 = {'name': 'EeEe', 'age': 20}
print(dict1)  # 输出字典
print(dict1.keys())
print(dict1.values())  # 输出字典的值
print(dict1['name'])  # 输出字典中键为'name'的值
dict1['age'] = 19  # 修改字典中键为'age'的值

# Bytes（字节类型）
bytes1 = b'Hello, World!'  # 创建字节类型
bytes2 = bytes("hello", encoding="utf-8")  # 创建字节类型
extract_bytes1 = bytes1[0:5]  # 提取字节类型的前5个字节
print(extract_bytes1) # 输出 b'Hello'


#海象运算符
strr="hello world"
if ( n := len(strr)) > 10:
    print(f"字符串长度大于10，长度为: {n}")