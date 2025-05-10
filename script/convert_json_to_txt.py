import os
import json

# Đường dẫn dữ liệu
json_folder = "new_datasets/BTXRD/annotations"
output_folder = "datasets/BTXRD/labels/val"
os.makedirs(output_folder, exist_ok=True)

# Nhãn theo class_names.txt
class_names = [
    "osteochondroma",
    "synovial osteochondroma",
    "osteofibroma",
    "simple bone cyst",
    "other bt",
    "other mt",
    "multiple osteochondromas",
    "osteosarcoma",
    "giant cell tumor"
]
label_map = {name.lower(): idx for idx, name in enumerate(class_names)}

# Hàm xử lý rectangle thành 4 điểm polygon
def rectangle_to_polygon(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ]

# Xử lý từng JSON
for filename in os.listdir(json_folder):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(json_folder, filename)
    with open(json_path, "r") as f:
        data = json.load(f)

    w = data.get("imageWidth", 1)
    h = data.get("imageHeight", 1)

    yolo_lines = []
    for shape in data.get("shapes", []):
        label = shape["label"].lower().strip()
        shape_type = shape.get("shape_type", "polygon")
        if label not in label_map:
            print(f"[!] Không có nhãn: {label} trong class_names.txt")
            continue
        class_id = label_map[label]

        if shape_type == "rectangle":
            points = rectangle_to_polygon(shape["points"][0], shape["points"][1])
        elif shape_type == "polygon":
            points = shape["points"]
        else:
            continue  # Bỏ qua loại shape không rõ

        # Chuẩn hóa
        x_coords = [p[0] / w for p in points]
        y_coords = [p[1] / h for p in points]
        coords = [coord for xy in zip(x_coords, y_coords) for coord in xy]

        line = f"{class_id} " + " ".join(f"{c:.6f}" for c in coords)
        yolo_lines.append(line)

    # Lưu file .txt
    txt_filename = filename.replace(".json", ".txt")
    txt_path = os.path.join(output_folder, txt_filename)
    with open(txt_path, "w") as out_f:
        out_f.write("\n".join(yolo_lines))
