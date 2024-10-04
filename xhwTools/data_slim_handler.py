import os
import shutil
import random

def sample_subset(root_dir, set_type, num_samples):
    # image_dir = os.path.join(root_dir, 'images', set_type)
    label_detection_dir = os.path.join(root_dir, 'detection-object', 'labels', set_type)
    label_drivable_dir = os.path.join(root_dir, 'seg-drivable-10', 'labels', set_type)
    label_lane_dir = os.path.join(root_dir, 'seg-lane-11', 'labels', set_type)

    # 创建目标目录，如果不存在
    os.makedirs(os.path.join(root_dir, 'sampled', 'images', set_type), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'sampled', 'detection-object', 'labels', set_type), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'sampled', 'seg-drivable-10', 'labels', set_type), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'sampled', 'seg-lane-11', 'labels', set_type), exist_ok=True)

    # 获取所有图像文件名
    # images = os.listdir(image_dir)

    # sampled_images = random.sample(images, num_samples)
    image_dir = os.path.join(root_dir, 'sampled\\images', set_type)
    sampled_images = os.listdir(image_dir)

    missing_labels = []

    # 复制图像和标签
    for img in sampled_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(root_dir, 'sampled', 'images', set_type, img))
        img_base, _ = os.path.splitext(img)

# 用法示例
root_dir = os.path.join('yolom_data')
sample_subset(root_dir, 'train2017', 10000)
sample_subset(root_dir, 'val2017', 3000)
