from ultralytics import YOLO
import cv2


class ObjectDetection:
    def __init__(self, model_path="model\yolov8m.pt"):
        self.model = YOLO(model_path)


    def img_display(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        results = self.model(img)
        self.box(results, img)
        cv2.imshow('Object Detection',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def vid_display(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model.track(frame, stream=True)
            self.box(results, frame)
            cv2.imshow('Object Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


    def box(self, results, frame):
        for result in results:
            classes_names = result.names
            for box in result.boxes:
                if box.conf[0] > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f'{classes_names[int(box.cls[0])]} {box.conf[0]:.2f}', (x1, y1),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)