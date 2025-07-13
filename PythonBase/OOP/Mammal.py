class Mammal:
    def __init__(self, name, sex):
        self.name = name
        self.sex = sex
        self.num_eyes = 2

    def breath(self):
        print(f"{self.name} is breathing.")
    def eat(self):
        print(f"{self.name} is eating.")


class Human(Mammal):
    def __init__(self, name, sex):
        super().__init__(name, sex)  # 调用父类的初始化方法
        self.has_tail = False # 子类特有属性
    def read(self): # 子类特有方法
        print(f"{self.name} is reading a book.")

class Cat(Mammal):
    def __init__(self, name, sex):
        super().__init__(name, sex)
        self.has_tail = True
    def meow(self):
        print(f"{self.name} says meow.")


cat1 = Cat("Jojo", "公")
print(cat1.num_eyes)
cat1.eat()