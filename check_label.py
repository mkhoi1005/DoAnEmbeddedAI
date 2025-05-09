import os
import json

annotations_dir = 'datasets/BTXRD/Annotations'
unique_labels = set()

for file in os.listdir(annotations_dir):
    if not file.endswith('.json'):
        continue

    json_path = os.path.join(annotations_dir, file)
    with open(json_path, 'r') as f:
        data = json.load(f)

    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label', '').strip()
        if label:
            unique_labels.add(label)

# In ra tất cả các nhãn khác nhau
print("Các nhãn (label) khác nhau trong tập dữ liệu:")
for label in sorted(unique_labels):
    print(f"- {label}")
