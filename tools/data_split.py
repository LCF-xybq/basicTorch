import os
import random
from shutil import copy

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    garbage_class = ['glass', 'cardboard', 'metal', 'bag', 'plastic', 'trash']
    # 排序，保证顺序一致
    garbage_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(garbage_class))

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in garbage_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    print(train_images_path)
    return train_images_path, train_images_label, val_images_path, val_images_label

def copy_train_test(srclst, srclabel, trg, is_train=None):
    train_pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\train'
    test_pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\test'
    ann_train = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\ann_train.txt'
    ann_test = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\ann_test.txt'
    for img, label in zip(srclst, srclabel):
        img_name = img.split('\\')[-1]

        if is_train is None:
            raise ValueError('is_train is None')
        elif is_train:
            with open(ann_train, 'a') as f:
                f.write(f'{img_name} {label}\n')

            copy(img, train_pth)
        else:
            with open(ann_test, 'a') as f:
                f.write(f'{img_name} {label}\n')

            copy(img, test_pth)



if __name__ == '__main__':
    pth = r'D:\Program_self\basicTorch\inputs\garbage-v2\raw'
    train = r'D:\Program_self\basicTorch\inputs\garbage-v2\data\train'
    t, t_l, v, v_l = read_split_data(pth, 0.2)
    copy_train_test(t, t_l, train, is_train=True)
    copy_train_test(v, v_l, train, is_train=False)
