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


    image_dir = os.path.join(root_dir, 'sampled\\images', set_type)
    sampled_images = os.listdir(image_dir)

    missing_labels = []



    # 复制标签
    for img in sampled_images:
        img_base, _ = os.path.splitext(img)

        # 复制检测标签
        label_det_file = img_base + '.txt'
        if os.path.exists(os.path.join(label_detection_dir, label_det_file)):
            shutil.copy(os.path.join(label_detection_dir, label_det_file), os.path.join(root_dir, 'sampled', 'detection-object', 'labels', set_type, label_det_file))
        else:
            missing_labels.append((img, 'detection'))

        # 复制可驾驶区域分割标签
        label_drv_file = img_base + '.txt'
        if os.path.exists(os.path.join(label_drivable_dir, label_drv_file)):
            shutil.copy(os.path.join(label_drivable_dir, label_drv_file), os.path.join(root_dir, 'sampled', 'seg-drivable-10', 'labels', set_type, label_drv_file))
        else:
            missing_labels.append((img, 'drivable'))

        # 复制车道分割标签
        if os.path.exists(os.path.join(label_lane_dir, label_drv_file)):
            shutil.copy(os.path.join(label_lane_dir, label_drv_file), os.path.join(root_dir, 'sampled', 'seg-lane-11', 'labels', set_type, label_drv_file))
        else:
            missing_labels.append((img, 'lane'))

    # 输出缺失的标签文件
    for missing in missing_labels:
        print(f"Missing {missing[1]} label for image {missing[0]}")

# 用法示例
root_dir = os.path.join('yolom_data')
sample_subset(root_dir, 'train2017', 10000)
sample_subset(root_dir, 'val2017', 3000)
