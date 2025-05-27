import os
import json
import shutil
from collections import defaultdict
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import pandas as pd

# --- Cấu hình đường dẫn ---
images_all_dir = '../datasets/BTXRD/images/all1'
annotations_dir = '../datasets/BTXRD/annotations'
images_train_dir = '../datasets/BTXRD/images/train'
images_val_dir = '../datasets/BTXRD/images/val'

# Tạo folder train/val nếu chưa có
os.makedirs(images_train_dir, exist_ok=True)
os.makedirs(images_val_dir, exist_ok=True)

# --- Map label sang class id ---
label_to_class_id = {
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

# --- Bước 1: Tạo danh sách ảnh và nhãn ---

image_files = sorted(os.listdir(images_all_dir))
image_info = []

def get_classes_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    classes = set()
    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label')
        if label is not None:
            class_id = label_to_class_id.get(label)
            if class_id is not None:
                classes.add(class_id)
    return list(classes)

for img_file in image_files:
    name, ext = os.path.splitext(img_file)
    json_file = os.path.join(annotations_dir, name + '.json')
    if os.path.exists(json_file):
        classes = get_classes_from_json(json_file)
    else:
        classes = []  # Ảnh không có khối u
    image_info.append({
        'image_file': img_file,
        'classes': classes
    })

# --- Bước 2: Chuẩn bị DataFrame đa nhãn ---

all_classes = range(9)  # 9 class

def create_multi_label_vector(label_list):
    vector = [0]*len(all_classes)
    for c in label_list:
        vector[c] = 1
    return vector

df = pd.DataFrame(image_info)
df['multi_label'] = df['classes'].apply(create_multi_label_vector)

X = df['image_file'].values
y = list(df['multi_label'].values)

# --- Bước 3: Stratified split đa nhãn ---
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(msss.split(X, y))

train_files = df.iloc[train_idx]['image_file'].tolist()
val_files = df.iloc[val_idx]['image_file'].tolist()

# --- Bước 4: Copy ảnh sang folder train/val ---
print("Copy images to train folder...")
for f in train_files:
    src = os.path.join(images_all_dir, f)
    dst = os.path.join(images_train_dir, f)
    shutil.copy2(src, dst)

print("Copy images to val folder...")
for f in val_files:
    src = os.path.join(images_all_dir, f)
    dst = os.path.join(images_val_dir, f)
    shutil.copy2(src, dst)

# --- Bước 5: Thống kê ảnh có/không có khối u ---
def count_has_tumor(file_list):
    count_yes = 0
    count_no = 0
    for f in file_list:
        name = os.path.splitext(f)[0]
        json_file = os.path.join(annotations_dir, name + '.json')
        if os.path.exists(json_file):
            count_yes += 1
        else:
            count_no += 1
    return count_yes, count_no

train_yes, train_no = count_has_tumor(train_files)
val_yes, val_no = count_has_tumor(val_files)

# --- Bước 6: Thống kê phân phối class ---
def count_classes(file_list):
    class_counts = defaultdict(int)
    for f in file_list:
        name = os.path.splitext(f)[0]
        json_file = os.path.join(annotations_dir, name + '.json')
        if os.path.exists(json_file):
            classes = get_classes_from_json(json_file)
            for c in classes:
                class_counts[c] += 1
    return class_counts

train_class_counts = count_classes(train_files)
val_class_counts = count_classes(val_files)

# --- Bước 7: In thông tin chi tiết sau khi chia ---

total_train = len(train_files)
total_val = len(val_files)

print("\n===== Thông tin phân chia dữ liệu =====")
print(f"Tổng ảnh train: {total_train}")
print(f"  - Ảnh có khối u: {train_yes} ({train_yes / total_train * 100:.2f}%)")
print(f"  - Ảnh không khối u: {train_no} ({train_no / total_train * 100:.2f}%)")

print(f"\nTổng ảnh val: {total_val}")
print(f"  - Ảnh có khối u: {val_yes} ({val_yes / total_val * 100:.2f}%)")
print(f"  - Ảnh không khối u: {val_no} ({val_no / total_val * 100:.2f}%)")

print("\nPhân phối số lượng từng class trong tập train:")
for c in sorted(all_classes):
    count = train_class_counts.get(c, 0)
    print(f"  Class {c}: {count}")

print("\nPhân phối số lượng từng class trong tập val:")
for c in sorted(all_classes):
    count = val_class_counts.get(c, 0)
    print(f"  Class {c}: {count}")

print("\nTỉ lệ ảnh có khối u trên tổng ảnh trong train: "
      f"{train_yes / total_train * 100:.2f}%")
print("Tỉ lệ ảnh không khối u trên tổng ảnh trong train: "
      f"{train_no / total_train * 100:.2f}%")

print("\nTỉ lệ ảnh có khối u trên tổng ảnh trong val: "
      f"{val_yes / total_val * 100:.2f}%")
print("Tỉ lệ ảnh không khối u trên tổng ảnh trong val: "
      f"{val_no / total_val * 100:.2f}%")

print("\n=======================================")
