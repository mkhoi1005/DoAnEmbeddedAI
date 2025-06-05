from ultralytics import YOLO
import json
import numpy as np
import cv2

# Đường dẫn
img_path = "BTXRD/images/val/IMG000071.jpeg"
anno_path = "BTXRD/images/annotations/IMG000071.json"
class_path = "BTXRD/images/class_names.txt"

# 1. Đọc class_names.txt để ánh xạ label -> index
class_map = {}
with open(class_path, encoding="utf-8") as f:
    for line in f:
        if ':' in line:
            idx, name = line.strip().split(':', 1)
            class_map[name.strip()] = int(idx.strip())

# 2. Đọc annotation và chuẩn hóa về 320x320, lấy cả label và index
with open(anno_path, encoding="utf-8") as f:
    anno = json.load(f)
img_h, img_w = anno["imageHeight"], anno["imageWidth"]
anno_instances = []
for shape in anno["shapes"]:
    if shape["shape_type"] == "polygon":
        poly = []
        for x, y in shape["points"]:
            x320 = x * 320 / img_w
            y320 = y * 320 / img_h
            poly.append([x320, y320])
        label = shape.get("label", "").strip()
        class_index = class_map.get(label, -1)
        anno_instances.append({
            "class_index": class_index,
            "label": label,
            "polygon": np.array(poly, dtype=np.float32)
        })

# 3. Predict với YOLOv8
model = YOLO("best.pt")
results = model(img_path, imgsz=320)

# 4. Đọc và resize ảnh để vẽ
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (320, 320))
img_show = img_resized.copy()

# 5. In kết quả predict theo mẫu yêu cầu và vẽ polygon predict (xanh lá, đã scale)
for r in results:
    if r.masks is not None:
        for i, (mask, box, cls) in enumerate(zip(r.masks.xy, r.boxes.xyxy, r.boxes.cls)):
            score = float(r.boxes.conf[i])
            class_index = int(cls)
            polygon_flat = mask.flatten().tolist()
            x1, y1, x2, y2 = box.tolist()
            arr = [class_index, score, x1, y1, x2, y2] + polygon_flat
            print(f"Predict instance {i}:")
            print(arr)
            # Scale polygon predict về 320x320
            mask_scaled = []
            for x, y in mask:
                x320 = x * 320 / img.shape[1]
                y320 = y * 320 / img.shape[0]
                mask_scaled.append([x320, y320])
            pts = np.array(mask_scaled, dtype=np.int32)
            cv2.polylines(img_show, [pts], isClosed=True, color=(0,255,0), thickness=2)

# 6. In annotation đã chuẩn hóa (có class_index) và vẽ polygon annotation (xanh dương)
for i, ins in enumerate(anno_instances):
    poly_flat = ins["polygon"].flatten().tolist()
    print(f"Annotation instance {i}:")
    print([ins["class_index"]] + poly_flat)
    pts = ins["polygon"].astype(np.int32)
    cv2.polylines(img_show, [pts], isClosed=True, color=(255,0,0), thickness=2)

# 7. So sánh số lượng instance
num_pred = len(results[0].masks.xy) if results[0].masks is not None else 0
num_anno = len(anno_instances)
print(f"\nSố instance predict: {num_pred}")
print(f"Số instance annotation: {num_anno}")

# 8. Hiển thị ảnh kết quả
cv2.imshow("Predict (green) & Annotation (blue)", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()