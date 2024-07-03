import numpy as np
from ultralytics import YOLO

class yolo():
    def __init__(self):
        self.model = YOLO("model/yolov8m.pt")  # initializing the model

    def pred(self, img:np.ndarray) -> list:
        return self.model(img)