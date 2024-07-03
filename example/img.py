from services.ObjectDetection.object_detector import ObjectDetector
import cv2

def main():
    img_path = "test2.jpg"
    detector = ObjectDetector()
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Error: Unable to read image at '{img_path}'.")
        return

    results = detector.detect_objects(img)
    detector.box(results, img)

    cv2.imshow('Object Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()