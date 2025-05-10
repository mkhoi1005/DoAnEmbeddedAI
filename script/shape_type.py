import os
import json
from collections import Counter

# Đường dẫn thư mục chứa annotation JSON
json_folder = "datasets/BTXRD/Annotations"

# Khởi tạo bộ đếm
shape_counter = Counter()

# Duyệt tất cả các file JSON
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(json_folder, filename)
        with open(json_path, 'r') as f:
            data = json.load(f)
            for shape in data.get("shapes", []):
                shape_type = shape.get("shape_type", "unknown")
                shape_counter[shape_type] += 1

# In kết quả
print("Thống kê các loại shape_type trong dataset:")
for shape_type, count in shape_counter.items():
    print(f"- {shape_type}: {count} instances")
