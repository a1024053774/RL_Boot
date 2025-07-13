import os

'''
为指定目录下的图片创建标签文件。
每张图片对应一个文本文件，文本文件中包含了图片的标签信息。
'''

root_dir = "dataset/hymenoptera_data/train"
target_dir = "ants_images"
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split('_')[0]  # 将ants_image分为{ants, image}，取第一个元素作为标签
out_dir = "ants_label"

for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_dir, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)