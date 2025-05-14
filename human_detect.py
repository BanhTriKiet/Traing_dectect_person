import cv2
from ultralytics import YOLO
import time

MODEL_PATH = "models/v9_MOT17_320.pt"
VIDEO_PATH = "assets/test.mp4"

model = YOLO(MODEL_PATH, task='detect')
cap = cv2.VideoCapture(VIDEO_PATH)
count = 0

while True:
    start_time = time.perf_counter()
    ret, frame = cap.read()
    if frame is None:
        print("Empty Frame")
        time.sleep(0.1)
        count += 1
        if count < 3:
            continue
        else:
            break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=320)
    detections = results[0].boxes

    for box in detections:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box (x1, y1, x2, y2)
        track_id = int(box.id) if box.id is not None else -1  # Track ID
        confidence = float(box.conf[0]) if box.conf is not None else 0  # Confidence score

        # Vẽ bounding box và hiển thị ID
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(frame, f"ID: {track_id} Conf: {confidence:.2f}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    end_time = time.perf_counter()
    fps = 1 / (end_time - start_time)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (10, 30),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

    cv2.imshow("YOLO Tracking", cv2.resize(frame, (640, 480)))

    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()