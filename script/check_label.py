import os
import numpy as np
from collections import Counter

label_dir = r""D:\EmbededAI\btxrd_yolov8n_seg_new1\content\DoAnEmbeddedAI\runs\segment\btxrd_yolov8n_seg\weights\best.pt""
classes = ['osteochondroma', 'synovial osteochondroma', 'osteofibroma', 
           'simple bone cyst', 'other bt', 'other mt', 
           'multiple osteochondromas', 'osteosarcoma', 'giant cell tumor']
small_objects_by_class = Counter()

for label_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, label_file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            points = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            area = (x_max - x_min) * (y_max - y_min)
            if area < 0.01:  # Diện tích <1%
                small_objects_by_class[class_id] += 1

print("Small objects by class:")
for class_id, count in small_objects_by_class.items():
    print(f"{classes[class_id]}: {count}")