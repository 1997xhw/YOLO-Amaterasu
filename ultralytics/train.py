import sys

from ultralytics import YOLO

sys.path.insert(0, "D:\\yolopm-all\\LOv8-multi-task\\ultralytics")
# 现在就可以导入Yolo类了
# from ultralytics import YOLO

# Load a model
# model = YOLO('/home/jiayuan/yolom/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml', task='multi')  # build a new model from YAML


# model = YOLO('D:\\yolopm-all\\YOLOv8-multi-task\\ultralytics\\models\\v8\\yolov8-bdd-v4-one-dropout-individual-n.yaml', task='multi')  # build a new model from YAML


###############################
# xhw
model = YOLO(r'D:\\yolopm-all\\YOLO-Amaterasu\\ultralytics\\models\\v8\\yolov8-m-xhw-SCSA-mbnv4.yaml',
             task='multi')  # build a new model from YAML
###############################


# model = YOLO('D:\\yolopm-all\\YOLOv8-multi-task\\runs\\20240903last.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
# model.train(data='D:\\yolopm-all\\YOLOv8-multi-task\\ultralytics\\datasets\\bdd-multi.yaml', batch=12, epochs=300, imgsz=(640,640), device=[0], name='yolopm', val=True, task='multi',classes=[2,3,4,9,10,11],combine_class=[2,3,4,9],single_cls=True)


model.train(data=r'D:\\yolopm-all\\YOLO-Amaterasu\\ultralytics\\datasets\\multi-paving-xhw.yaml',
            batch=16,
            epochs=2,
            imgsz=(640, 640),
            device=[0],
            name='MultiPavingEye-mbnv4',
            val=True,
            task='multi',
            classes=[0, 1, 2, 3],
            # combine_class=[2, 3, 4, 9],
            save_conf=True,
            save_txt=True,
            single_cls=False,
            )
