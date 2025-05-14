import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# ==== CONFIG ====
MODEL_PATH = "models/v12_MOT17_320_150.pt"
IMG_FOLDER = "test/MOT20-01/img1"
OUTPUT_TXT = "results/MOT20-01.txt"
OUTPUT_IMG_DIR = "results/images"
IMG_SUFFIX = ".jpg"

# ==== INIT ====
model = YOLO(MODEL_PATH, task='detect')

img_files = sorted([f for f in os.listdir(IMG_FOLDER) if f.endswith(IMG_SUFFIX)])
frame_id = 0

os.makedirs(os.path.dirname(OUTPUT_TXT), exist_ok=True)
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

with open(OUTPUT_TXT, "w") as f:
    for img_name in tqdm(img_files):
        frame_id += 1
        img_path = os.path.join(IMG_FOLDER, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"⚠️ Không thể đọc ảnh: {img_path}")
            continue

        # Chạy tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", imgsz=320)
        detections = results[0].boxes

        # Dùng set để kiểm tra trùng ID trong cùng frame
        used_ids = set()

        for box in detections:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id) if box.id is not None else -1
            conf = float(box.conf[0]) if box.conf is not None else 0.0

            # Bỏ qua nếu ID không xác định hoặc đã xuất hiện trong frame này
            if track_id <= 0 or track_id in used_ids:
                continue

            used_ids.add(track_id)

            w = x2 - x1
            h = y2 - y1

            # Ghi kết quả theo định dạng MOT
            line = f"{frame_id},{track_id},{x1},{y1},{w},{h},{conf:.4f},-1,-1\n"
            f.write(line)

            # Vẽ bounding box
            label = f"ID {track_id} ({conf:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Lưu ảnh nếu cần
        output_img_path = os.path.join(OUTPUT_IMG_DIR, img_name)
        #cv2.imwrite(output_img_path, frame)

print(f"✅ Đã lưu kết quả MOT tại: {OUTPUT_TXT}")
print(f"🖼️ Ảnh có vẽ kết quả lưu tại: {OUTPUT_IMG_DIR}")
