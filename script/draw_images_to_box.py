import os
import cv2

# Đường dẫn
image_dir = r"D:\EmbededAI\datasets\BTXRD\images\all"
label_dir = r"D:\EmbededAI\datasets\BTXRD\labels\all"
output_dir = r"D:\EmbededAI\output\images_with_boxes"
os.makedirs(output_dir, exist_ok=True)

# Danh sách lớp (tùy vào dataset của bạn, ví dụ 0 = osteosarcoma)
class_names = ["osteosarcoma"]  # Cập nhật nếu có nhiều lớp

# Lặp qua từng file ảnh
for img_name in os.listdir(image_dir):
    if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(image_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    if not os.path.exists(label_path):
        print(f"[!] Không tìm thấy label cho {img_name}")
        continue

    image = cv2.imread(img_path)
    h, w, _ = image.shape

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id, x_center, y_center, bw, bh = map(float, parts)

        # Chuyển đổi tọa độ từ YOLO về pixel
        x_center *= w
        y_center *= h
        bw *= w
        bh *= h

        x1 = int(x_center - bw / 2)
        y1 = int(y_center - bh / 2)
        x2 = int(x_center + bw / 2)
        y2 = int(y_center + bh / 2)

        color = (0, 255, 0)
        label = class_names[int(class_id)] if int(class_id) < len(class_names) else f"id_{class_id}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, image)
    print(f"[✓] Đã vẽ {img_name} → {out_path}")
