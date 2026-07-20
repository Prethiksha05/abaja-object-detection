from ultralytics import YOLO
import cv2

# Use a larger pretrained model
model = YOLO("yolo11x.pt")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, conf=0.3)

    annotated = results[0].plot()

    cv2.imshow("Live Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
