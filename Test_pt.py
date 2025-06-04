from ultralytics import YOLO

model = YOLO("best.pt")
results = model("BTXRD/images/val/IMG000071.jpeg")  # hoặc một ảnh bất kỳ

for r in results:
    r.show()         # Hiển thị ảnh với mask
    r.save("output/")  # Lưu ảnh kết quả
    print(r.masks)   # mask numpy array
    print(r.boxes)   # bounding box
    print(r.probs)   # class probabilities