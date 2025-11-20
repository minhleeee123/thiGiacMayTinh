from ultralytics import YOLO
import os
if __name__ == '__main__':
    data_path = os.path.abspath('data.yaml')
    model = YOLO('yolov8n.pt')
    model.train(data=data_path, epochs=30, imgsz=640, batch=4)