import numpy as np
import cv2
from services.ObjectDetection.yolov8 import yolo

class ObjectDetector:
    def __init__(self):
        pass

    def predictor(self, img: np.ndarray) -> list:
        detect = yolo()
        results = detect.pred(img)
        return results

    def confidence(self, results: list) -> list:
        confidences = []
        for result in results:
            for box in result.boxes:
                confidences.append(box.conf)
        return confidences

    def coords(self, results: list) -> list:
        coordinates = []
        for result in results:
            for box in result.boxes:
                coordinates.append(box.xyxy[0])
        return coordinates

    def class_names(self, results: list) -> list:
        class_names = []
        for result in results:
            for box in result.boxes:
                class_names.append(result.names[int(box.cls[0])])
        return class_names

    def detect_objects(self, img: np.ndarray) -> dict:
        results = self.predictor(img)
        return {
            'bounding_boxes': self.coords(results),
            'class_names': self.class_names(results),
            'confidences': self.confidence(results)
        }

    def box(self, results: dict, frame: np.ndarray) -> None:
        bounding_boxes = results['bounding_boxes']
        class_names = results['class_names']
        confidences = results['confidences']

        for i in range(len(bounding_boxes)):
            x1, y1, x2, y2 = map(int, bounding_boxes[i])
            if confidences[i] > 0.5:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_names[i]} {confidences[i].item():.2f}', (x1, y1),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)