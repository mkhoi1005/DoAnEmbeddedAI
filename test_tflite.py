import numpy as np
import cv2
import tensorflow.lite as tflite
import json

# Đường dẫn model, ảnh, annotation, class
model_path = "best_float32.tflite"
image_path = "BTXRD/images/val/IMG000071.jpeg"
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

# 3. Load model và ảnh
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread(image_path)
img_resized = cv2.resize(img, (320, 320))
img_show = img_resized.copy()
input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
input_data = (input_data - 0.0) / 255.0

# 4. Chạy model
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

feature_dim = output_details[0]['shape'][2]
class_count = len(class_map)
polygon_points = (feature_dim - 4 - class_count) // 2

# 5. In kết quả predict theo mẫu yêu cầu và vẽ polygon predict (xanh lá)
for idx, row in enumerate(output_data[0]):
    score = np.max(row[4:4+class_count])
    class_index = int(np.argmax(row[4:4+class_count]))
    if score < 0.1:
        continue
    x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
    polygon = row[4+class_count:]
    if len(polygon) % 2 != 0:
        polygon = polygon[:-1]
    if len(polygon) != polygon_points * 2:
        print(f"Row {idx}: Polygon length không đúng ({len(polygon)})")
        continue
    polygon_pixel = []
    for i, v in enumerate(polygon):
        if i % 2 == 0:
            polygon_pixel.append(float(v) * 320)
        else:
            polygon_pixel.append(float(v) * 320)
    arr = [class_index, score, x1, y1, x2, y2] + polygon_pixel
    print(f"Predict instance {idx}:")
    print(arr)
    # Vẽ polygon predict (xanh lá)
    pts = np.array(polygon_pixel, dtype=np.int32).reshape(-1, 2)
    cv2.polylines(img_show, [pts], isClosed=True, color=(0,255,0), thickness=2)

# 6. In annotation đã chuẩn hóa (có class_index) và vẽ polygon annotation (xanh dương)
for i, ins in enumerate(anno_instances):
    poly_flat = ins["polygon"].flatten().tolist()
    print(f"Annotation instance {i}:")
    print([ins["class_index"]] + poly_flat)
    pts = ins["polygon"].astype(np.int32)
    cv2.polylines(img_show, [pts], isClosed=True, color=(255,0,0), thickness=2)

# 7. So sánh số lượng instance
num_pred = sum(1 for row in output_data[0] if np.max(row[4:4+class_count]) >= 0.1)
num_anno = len(anno_instances)
print(f"\nSố instance predict: {num_pred}")
print(f"Số instance annotation: {num_anno}")

# 8. Hiển thị ảnh kết quả
cv2.imshow("Predict (green) & Annotation (blue)", img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()