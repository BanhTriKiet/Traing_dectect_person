from ultralytics import YOLO

# Load mô hình đã huấn luyện
model = YOLO("models/v9_MOT17_320.pt")  # Đường dẫn đến trọng số của bạn

# Chạy đánh giá trên tập test được chỉ định trong data.yaml
metrics = model.val(
    data="./assets/MOT17_NEW/data.yaml",    # File YAML chứa thông tin dataset (phải có dòng 'test:')
    split="test",        # Chạy trên tập test (không phải train/val)
    imgsz=320          # Kích thước ảnh input
)

# In kết quả đánh giá
print(metrics)