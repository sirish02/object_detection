from services.ObjectDetection.object_detector import ObjectDetector
import cv2
import sys
#sys.path.append("D:\Treeleaf\pythonProject\services\ObjectDetection")
#from object_detector import ObjectDetector

def main():
    video_path = "test4.mp4"
    cap = cv2.VideoCapture(video_path)
    detector = ObjectDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect_objects(frame)
        detector.box(results, frame)

        cv2.imshow('Object Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
