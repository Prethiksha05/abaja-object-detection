from ultralytics import YOLO
import cv2

model = YOLO("yolo11x.pt")   # much stronger than yolo11n

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    results = model(frame, conf=0.3)

    cv2.imshow("Detection", results[0].plot())

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

