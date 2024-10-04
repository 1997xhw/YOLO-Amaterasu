import os
import shutil
import random


def split_dataset(raw_dir, train_ratio=0.8):
    # 设置输入目录路径
    det_dir = os.path.join(raw_dir, 'det')
    seg_dir = os.path.join(raw_dir, 'seg')
    images_dir = os.path.join(raw_dir, 'images')

    # 设置输出目录路径
    train_det_dir = os.path.join(det_dir, 'train')
    val_det_dir = os.path.join(det_dir, 'val')

    train_seg_dir = os.path.join(seg_dir, 'train')
    val_seg_dir = os.path.join(seg_dir, 'val')

    train_images_dir = os.path.join(images_dir, 'train')
    val_images_dir = os.path.join(images_dir, 'val')

    # 确保目标文件夹存在
    for folder in [train_det_dir, val_det_dir, train_seg_dir, val_seg_dir, train_images_dir, val_images_dir]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # 获取所有det文件的基本文件名（不包含扩展名）
    file_list = [os.path.splitext(file)[0] for file in os.listdir(det_dir) if file.endswith('.txt')]

    # 随机打乱文件列表
    random.shuffle(file_list)

    # 计算训练集的数量
    train_size = int(len(file_list) * train_ratio)

    # 将文件列表分为训练集和验证集
    train_files = file_list[:train_size]
    val_files = file_list[train_size:]

    # 移动文件的函数
    def move_files(file_list, src_det_dir, src_seg_dir, src_images_dir, dst_det_dir, dst_seg_dir, dst_images_dir):
        for file_base in file_list:
            # 处理det文件
            det_src_path = os.path.join(src_det_dir, file_base + '.txt')
            det_dst_path = os.path.join(dst_det_dir, file_base + '.txt')
            if os.path.exists(det_src_path):
                shutil.move(det_src_path, det_dst_path)

            # 处理seg文件
            seg_src_path = os.path.join(src_seg_dir, file_base + '.txt')
            seg_dst_path = os.path.join(dst_seg_dir, file_base + '.txt')
            if os.path.exists(seg_src_path):
                shutil.move(seg_src_path, seg_dst_path)

            # 处理images文件
            image_src_path = os.path.join(src_images_dir, file_base + '.jpg')  # 假设图像格式为.jpg
            image_dst_path = os.path.join(dst_images_dir, file_base + '.jpg')
            if os.path.exists(image_src_path):
                shutil.move(image_src_path, image_dst_path)

    # 移动训练集文件
    move_files(train_files, det_dir, seg_dir, images_dir, train_det_dir, train_seg_dir, train_images_dir)

    # 移动验证集文件
    move_files(val_files, det_dir, seg_dir, images_dir, val_det_dir, val_seg_dir, val_images_dir)

    print(f"数据集划分完成：{len(train_files)} 个文件到训练集，{len(val_files)} 个文件到验证集。")


# 使用示例
raw_dir = r"D:\yolopm-all\all\1004\raw"
split_dataset(raw_dir)

