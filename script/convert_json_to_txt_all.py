import os
import json

# Thư mục đầu vào và đầu ra
json_folder = r"D:\EmbededAI\datasets\BTXRD\Annotations"
output_folder = r"D:\EmbededAI\datasets\BTXRD\labels\all"

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Bảng ánh xạ tên lớp → class_id
label2id = {
    "osteochondroma": 0,
    "synovial osteochondroma": 1,
    "osteofibroma": 2,
    "simple bone cyst": 3,
    "other bt": 4,
    "other mt": 5,
    "multiple osteochondromas": 6,
    "osteosarcoma": 7,
    "giant cell tumor": 8,
}

# Lặp qua từng file JSON
for filename in os.listdir(json_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(json_folder, filename)
    txt_path = os.path.join(output_folder, filename.replace(".json", ".txt"))

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_w = data.get("imageWidth")
    image_h = data.get("imageHeight")

    if not image_w or not image_h:
        print(f"Bỏ qua {filename}: thiếu imageWidth/imageHeight")
        continue

    with open(txt_path, "w") as out:
        for shape in data.get("shapes", []):
            if shape.get("shape_type") != "polygon":
                continue  # Bỏ qua bbox và các shape khác

            label = shape.get("label", "").strip()
            if label not in label2id:
                print(f"[!] Nhãn '{label}' không có trong bảng label2id. Bỏ qua trong {filename}")
                continue

            class_id = label2id[label]
            points = shape.get("points", [])

            if len(points) < 3:
                continue  # polygon phải có ít nhất 3 điểm

            line = [str(class_id)]
            for x, y in points:
                x_norm = x / image_w
                y_norm = y / image_h
                line.append(f"{x_norm:.6f}")
                line.append(f"{y_norm:.6f}")

            out.write(" ".join(line) + "\n")

print("✅ Đã chuyển xong tất cả file JSON sang TXT (polygon-only, có class mapping).")
