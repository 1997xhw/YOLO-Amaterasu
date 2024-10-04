import json
import os
import shutil

# 标签的映射表，将 cat 和 dog 合并为 animal，将 car 和 truck 合并为 vehicle，将 bicycle 和 motorcycle 合并为 cycle
label_map = {
    "paving": 0,
    "person": 1,
    "cycle": 2,  # 合并 bicycle 和 motorcycle
    "vehicle": 3,  # 合并 car 和 truck
    # "animal": 4  # 合并 cat 和 dog
    # "obstacle" 类别被跳过，不包含在映射表中
}

# 需要合并的标签
merge_labels = ["cat", "dog"]
merge_labels_cycle = ["bicycle", "motorcycle"]
merge_labels_vehicle = ["car", "truck"]
merge_labels_obstacle = ["fire hydrant", "parking meter", "bench"]


def convert_to_yolo_format(json_dir, output_dir, image_dir):
    empty_files = []  # 用于存储空文件的名字

    # 确保目标文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取所有JSON文件
    for json_file in os.listdir(json_dir):
        if json_file.endswith(".json"):
            json_path = os.path.join(json_dir, json_file)

            # 打开并解析JSON文件
            with open(json_path, 'r') as f:
                data = json.load(f)

            image_width = data['imageWidth']
            image_height = data['imageHeight']
            shapes = data.get('shapes', [])

            output_txt = ""

            # 处理每个shape
            for shape in shapes:
                label = shape['label']

                # 合并 cat 和 dog 为 animal
                if label in merge_labels:
                    continue

                # 合并 bicycle 和 motorcycle 为 cycle
                elif label in merge_labels_cycle:
                    label = "cycle"

                # 合并 car 和 truck 为 vehicle
                elif label in merge_labels_vehicle:
                    label = "vehicle"

                # 合并 fire hydrant、parking meter、bench 为 obstacle
                elif label in merge_labels_obstacle:
                    label = "obstacle"

                # 跳过 obstacle 类别
                if label == "obstacle":
                    continue

                if label not in label_map:
                    continue  # 忽略不需要的标签

                label_index = label_map[label]

                # 只支持矩形
                if shape['shape_type'] == 'rectangle':
                    points = shape['points']

                    # 获取矩形的边界点
                    x_min = min(points[0][0], points[2][0])
                    x_max = max(points[0][0], points[2][0])
                    y_min = min(points[0][1], points[2][1])
                    y_max = max(points[0][1], points[2][1])

                    # 计算YOLO格式的中心坐标和宽高
                    x_center = (x_min + x_max) / 2 / image_width
                    y_center = (y_min + y_max) / 2 / image_height
                    width = (x_max - x_min) / image_width
                    height = (y_max - y_min) / image_height

                    # 按照YOLO格式添加行
                    output_txt += f"{label_index} {x_center} {y_center} {width} {height}\n"

            # 如果有有效的标签，才写入txt文件；否则记录空文件
            if output_txt.strip():
                txt_filename = os.path.splitext(json_file)[0] + ".txt"
                txt_path = os.path.join(output_dir, txt_filename)

                with open(txt_path, 'w') as f:
                    f.write(output_txt)

                # 复制同名的图像文件
                image_filename = os.path.splitext(json_file)[0] + ".jpg"  # 假设图片格式是jpg
                image_src_path = os.path.join(image_dir, image_filename)
                image_dest_path = os.path.join(output_dir+"/images", image_filename)

                if os.path.exists(image_src_path):
                    shutil.copy(image_src_path, image_dest_path)
                else:
                    print(f"图片文件 {image_filename} 不存在，跳过复制。")
            else:
                empty_files.append(json_file)  # 记录空文件

    # 输出空文件列表
    if empty_files:
        print("以下JSON文件没有有效标签，被跳过：")
        for file in empty_files:
            print(file)
    else:
        print("所有文件都包含有效标签。")


# 使用示例
json_dir = r"D:\yolopm-all\TP-paving\det\json"
output_dir = r"D:\yolopm-all\TP-paving\det"
image_dir = r"D:\yolopm-all\TP-paving\images-slim"  # 图像文件夹路径
convert_to_yolo_format(json_dir, output_dir, image_dir)
