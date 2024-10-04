import sys

sys.path.insert(0, "D:\\yolopm-all\\YOLOv8-multi-task\\ultralytics")

from ultralytics import YOLO

number = 2  # input how many tasks in your work
# model = YOLO('D:\\yolopm-all\\YOLOv8-multi-task\\runs\\v4.pt')  # Validate the model
# model = YOLO('D:\\yolopm-all\\YOLOv8-multi-task\\runs\\multi\\yolopm-xhw19\\weights\\best.pt')  # Validate the model
model = YOLO(r"D:\yolopm-all\YOLOv8-multi-task-paving\runs\multi\1003-SCSA-AKConv-concatV2\weights\best.pt")  # Validate the model
# model = YOLO('D:\\yolopm-all\\YOLOv8-multi-task\\runs\multi\\yolopm-xhw12\\weights\\best.pt')  # Validate the model
model.predict(source='D:\\yolopm-all\\YOLOv8-multi-task-paving\\test_data_yolopm', imgsz=(384, 672), device=[0],
              name='xhw_test', save=True, conf=0.25, show_labels=True)
