from ultralytics import YOLO

model = YOLO("best.pt")
metrics = model.val(data="BTXRD/BTXRD.yaml", imgsz=320)

# Hiển thị kết quả gọn gàng
print("\n--- Kết quả đánh giá ---")
for k, v in metrics.results_dict.items():
    print(f"{k}: {v:.4f}")