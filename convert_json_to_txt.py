import os
import json
from tqdm import tqdm

# Cấu hình
json_folder = "datasets/BTXRD/Annotations"
output_folder = "datasets/BTXRD/labels"
os.makedirs(output_folder, exist_ok=True)

label_map = {
    "giant cell tumor": 0,
    "multiple osteochondromas": 1,
    "osteochondroma": 2,
    "osteofibroma": 3,
    "osteosarcoma": 4,
    "other bt": 5,
    "other mt": 6,
    "simple bone cyst": 7,
    "synovial osteochondroma": 8
}

# Duyệt tất cả file JSON
for filename in tqdm(os.listdir(json_folder)):
    if not filename.endswith(".json"):
        continue

    json_path = os.path.join(json_folder, filename)
    with open(json_path, 'r') as f:
        data = json.load(f)

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]
    label_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        class_id = label_map.get(label)
        if class_id is None:
            continue

        shape_type = shape.get("shape_type", "polygon")
        points = shape["points"]

        # Chuyển rectangle thành polygon
        if shape_type == "rectangle" and len(points) == 2:
            (x1, y1), (x2, y2) = points
            points = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]

        # Chuẩn hóa điểm
        normalized_points = []
        for x, y in points:
            nx = x / image_width
            ny = y / image_height
            normalized_points.extend([nx, ny])

        line = f"{class_id} " + " ".join(f"{p:.6f}" for p in normalized_points)
        label_lines.append(line)

    # Ghi file txt
    txt_filename = filename.replace(".json", ".txt")
    txt_path = os.path.join(output_folder, txt_filename)
    with open(txt_path, "w") as f:
        f.write("\n".join(label_lines))
