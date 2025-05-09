import os
import json

input_dir = 'datasets/BTXRD/Annotations'
output_dir = 'datasets/BTXRD/labels'

os.makedirs(output_dir, exist_ok=True)

label_map = {
    "benign": 0,
    "malignant": 1,
    "other mt": 2  # hoặc bạn có thể chọn chỉ dùng benign và malignant
}

for filename in os.listdir(input_dir):
    if not filename.endswith('.json'):
        continue

    filepath = os.path.join(input_dir, filename)
    with open(filepath, 'r') as f:
        data = json.load(f)

    shapes = data.get("shapes", [])
    if not shapes:
        print(f"⚠️ Bỏ qua: {filename} không có shapes.")
        continue

    image_width = data["imageWidth"]
    image_height = data["imageHeight"]

    label_lines = []
    for shape in shapes:
        label = shape["label"].lower().strip()
        if label not in label_map:
            continue  # bỏ qua label không xác định

        cls_id = label_map[label]
        if shape["shape_type"] == "rectangle":
            (x1, y1), (x2, y2) = shape["points"]
            cx = (x1 + x2) / 2 / image_width
            cy = (y1 + y2) / 2 / image_height
            w = abs(x2 - x1) / image_width
            h = abs(y2 - y1) / image_height
            label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        elif shape["shape_type"] == "polygon":
            points = shape["points"]
            norm_points = [(x / image_width, y / image_height) for x, y in points]
            flat_coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in norm_points])
            label_lines.append(f"{cls_id} {flat_coords}")

    if label_lines:
        out_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(output_dir, out_filename), 'w') as out_f:
            out_f.write("\n".join(label_lines))
    else:
        print(f"⚠️ Bỏ qua: {filename} không có label hợp lệ.")
