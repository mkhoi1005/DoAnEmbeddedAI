import os
import json
import numpy as np

class_map = {
    "osteochondroma": 0,
    "synovial osteochondroma": 1,
    "osteofibroma": 2,
    "simple bone cyst": 3,
    "other bt": 4,
    "other mt": 5,
    "multiple osteochondromas": 6,
    "osteosarcoma": 7,
    "giant cell tumor": 8
}

ann_dir = "BTXRD/annotations"

for fname in os.listdir(ann_dir):
    if fname.endswith(".json"):
        json_path = os.path.join(ann_dir, fname)
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        labels = []
        for shape in data['shapes']:
            class_idx = class_map[shape['label']]
            points = [coord for point in shape['points'] for coord in point]
            label = [class_idx] + points  # Lưu là list, không phải np.array
            labels.append(label)
        labels = np.array(labels, dtype=object)  # Mỗi phần tử là list
        npy_path = os.path.join(ann_dir, fname.replace(".json", ".npy"))
        np.save(npy_path, labels)
        print(f"Converted {fname} -> {os.path.basename(npy_path)}")