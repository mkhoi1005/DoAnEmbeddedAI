import os
import cv2
import json
import numpy as np

# Đường dẫn thư mục
base_dir = "D:/EmbededAI/datasets/BTXRD"
image_dir = os.path.join(base_dir, "images", "all")
json_dir = os.path.join(base_dir, "Annotations")
output_dir = "output_annotated_images"

# Tạo thư mục đầu ra nếu chưa có
os.makedirs(output_dir, exist_ok=True)

def draw_annotation(image_path, json_path, save_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[LỖI] Không đọc được ảnh: {image_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        points = shape["points"]
        label = shape["label"]
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        x, y = pts[0][0]
        cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite(save_path, image)
    print(f"[OK] Đã xử lý: {save_path}")

# Duyệt toàn bộ ảnh trong thư mục
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.jpeg')]

for image_file in image_files:
    name = os.path.splitext(image_file)[0]
    image_path = os.path.join(image_dir, image_file)
    json_path = os.path.join(json_dir, name + ".json")
    save_path = os.path.join(output_dir, image_file)

    if not os.path.exists(json_path):
        print(f"[BỎ QUA] Không tìm thấy annotation: {json_path}")
        continue

    draw_annotation(image_path, json_path, save_path)

print("✅ Hoàn tất! Tất cả ảnh đã lưu tại:", output_dir)
