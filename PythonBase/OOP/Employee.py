class Employee:
    def __init__(self, name, id):
        self.name = name
        self.id = id

    def print_info(self):
        print(f"Employee Name: {self.name}, ID: {self.id}")

class FullTimeEmployee(Employee):
    def __init__(self, name, id, monthly_salary):
        super().__init__(name, id)
        self.monthly_salary = monthly_salary

    def calculate_monthly_salary(self):
        print("月薪是:", self.monthly_salary)

class PartTimeEmployee(Employee):
    def __init__(self, name, id, work_days, daily_salary):
        super().__init__(name, id)
        self.work_days = work_days
        self.daily_salary = daily_salary

    def calculate_monthly_salary(self):
        print("月薪是:", self.work_days * self.daily_salary)

# 测试代码
full_time_emp = FullTimeEmployee("Alice", "FT001", 5000)
full_time_emp.calculate_monthly_salary()

part_time_emp = PartTimeEmployee("Bob", "PT001", 20, 200)
part_time_emp.calculate_monthly_salary()