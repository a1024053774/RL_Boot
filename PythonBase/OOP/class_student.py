class Student:
    def __init__(self, name, no):
        self.name = name
        self.no = no
        self.grades = {"语文": 0, "数学": 0, "英语": 0}

    def set_grade(self, course, grade):
        if course in self.grades:
            self.grades[course] = grade
        else:
            print(f"课程 {course} 不存在。")

    def print_score(self):
        print(f"姓名: {self.name}, 学号: {self.no}, 成绩: {self.grades}")


# 创建学生对象
zhang = Student("张三", "2023001")
zhang.print_score()
zhang.set_grade("数学", 95)  # 修改数学成绩
zhang.print_score()  # 打印修改后的成绩