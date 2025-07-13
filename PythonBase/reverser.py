def reverser(input):
    inputWords = input.split(" ") # 将输入字符串按空格分割成一个列表
    inputWords = inputWords[-1::-1] # 将列表中的元素反转
    outputWords = ' '.join(inputWords) # 将列表中的元素连接成一个字符串

    return outputWords

if __name__ == '__main__':
    input = 'I am working here'
    rw = reverser(input)
    print(rw)