def calculate_sector(central_angle, radius):
    sector_area = central_angle / 360 * 3.14 * radius ** 2
    print(f"此扇形的面积为{sector_area:.2f}平方厘米")
    return sector_area #想要在函数之外使用局部变量,使用return语句将其返回

#return之后,可以获取面积的值进行后续操作
sector_1 = calculate_sector(160,30)
sector_2 = calculate_sector(160,40)

def calculateBMI(weight, height):
    bmi = weight / (height ** 2)
    if bmi < 18.5:
        print("体重过轻")
    elif 18.5 <= bmi < 25:
        print("体重正常")
    elif 25 <= bmi < 30:
        print("体重过重")
    else:
        print("肥胖")
    return bmi

person_1 = calculateBMI(50, 1.75)
person_2 = calculateBMI(200, 1.80)
print(person_1,person_2)




