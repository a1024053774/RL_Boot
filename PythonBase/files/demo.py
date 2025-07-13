
# open("demo.txt", "w").write("Hello, World!")
# f = open(".\demo.txt", "r")
# print(f.read())
# f.close()

# #附加模式
# with open("demo.txt", "a") as f: #append模式
#     f.write("\nThis is an appended line.")

#r+,支持读和写,并且写是以追加的方式
with open(".\demo.txt", "r+") as f:
    content = f.read()
    print("Original content:")
    print(content)

    f.write("\nThis is an appended line in r+ mode.")

    # Move the cursor back to the beginning of the file
    f.seek(0)

    # Read the updated content
    updated_content = f.read()
    print("Updated content:")
    print(updated_content)