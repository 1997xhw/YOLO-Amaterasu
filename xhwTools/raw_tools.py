import os
import shutil


def process_yolo_dataset(raw_dir):
    det_dir = os.path.join(raw_dir, 'det')
    seg_dir = os.path.join(raw_dir, 'seg')
    images_dir = os.path.join(raw_dir, 'images')

    deleted_files_log = []  # 用于存储被删除文件的名字

    # 遍历det文件夹中的每一个标签文件
    for det_file in os.listdir(det_dir):
        if det_file.endswith('.txt'):
            det_file_path = os.path.join(det_dir, det_file)

            # 读取标签文件内容
            with open(det_file_path, 'r') as f:
                lines = f.readlines()

            # 保留不属于类别4和5的行
            new_lines = [line for line in lines if not line.startswith('4') and not line.startswith('5')]

            # 如果新文件为空，则删除文件
            if not new_lines:
                # 删除det文件
                os.remove(det_file_path)

                # 删除对应的seg和images文件
                base_filename = os.path.splitext(det_file)[0]
                seg_file_path = os.path.join(seg_dir, base_filename + '.txt')
                image_file_path = os.path.join(images_dir, base_filename + '.jpg')  # 假设图像格式为.jpg

                # 删除seg文件
                if os.path.exists(seg_file_path):
                    os.remove(seg_file_path)

                # 删除image文件
                if os.path.exists(image_file_path):
                    os.remove(image_file_path)

                # 记录被删除的文件
                deleted_files_log.append(base_filename)

            else:
                # 如果新文件不为空，更新det文件内容
                with open(det_file_path, 'w') as f:
                    f.writelines(new_lines)

    # 写入被删除文件的日志
    if deleted_files_log:
        log_file_path = os.path.join(raw_dir, 'deleted_files_log.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write("Deleted the following files (det, seg, images):\n")
            for filename in deleted_files_log:
                log_file.write(f"{filename}\n")

    print(f"处理完成，删除了 {len(deleted_files_log)} 个文件。日志已存储在 {log_file_path}。")


# 使用示例
raw_dir = r"D:\yolopm-all\all\1004\raw"
process_yolo_dataset(raw_dir)
