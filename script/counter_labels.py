import os
from collections import defaultdict

labels_dir = "datasets/BTXRD/labels/train"
class_counts = defaultdict(int)

for file in os.listdir(labels_dir):
    if file.endswith(".txt"):
        class_set = set()
        with open(os.path.join(labels_dir, file), "r") as f:
            for line in f:
                if line.strip():
                    class_id = int(line.split()[0])
                    class_set.add(class_id)
        for cls in class_set:
            class_counts[cls] += 1

# In kết quả
for cls in sorted(class_counts.keys()):
    print(f"Lớp {cls}: {class_counts[cls]} ảnh chứa lớp này")
