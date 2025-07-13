class CuteCat:
    #定义类
    def __init__(self, cat_name, cat_age, cat_color): #(对象自身,属性1,属性2,...)
        # 初始化方法
        # 对象的属性
        self.name = cat_name
        self.age = cat_age
        self.color = cat_color

        # 对象的方法
    def meow(self):
        print(f"{self.name} says meow!")
    def think(self, content):
        print(f"{self.name} is thinking about {content}.")

# 创建 CuteCat 类的对象
cat1 = CuteCat('cat1', 2, 'red')
print(cat1.name, cat1.age, cat1.color)
# 调用对象的方法
cat1.meow()
cat1.think('fish')  




