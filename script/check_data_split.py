import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# --- Cấu hình ---
images_train_dir = '../datasets/BTXRD/images/train'
images_val_dir = '../datasets/BTXRD/images/val'
annotations_dir = '../datasets/BTXRD/annotations'

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
NUM_CLASSES = len(label_to_class_id)

def get_classes_from_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    classes = set()
    shapes = data.get('shapes', [])
    for shape in shapes:
        label = shape.get('label')
        if label in label_to_class_id:
            classes.add(label_to_class_id[label])
    return list(classes)

def create_df(image_dir):
    image_files = sorted(os.listdir(image_dir))
    records = []
    for f in image_files:
        name, _ = os.path.splitext(f)
        json_file = os.path.join(annotations_dir, name + '.json')
        if os.path.exists(json_file):
            classes = get_classes_from_json(json_file)
        else:
            classes = []
        label_vector = [0] * NUM_CLASSES
        for c in classes:
            label_vector[c] = 1
        records.append({
            'image_file': f,
            'has_tumor': int(bool(classes)),
            'multi_label': label_vector
        })
    return pd.DataFrame(records)

# Load dữ liệu
df_train = create_df(images_train_dir)
df_val = create_df(images_val_dir)

# --- Kiểm tra số lượng ---
print("\n📊 Tổng số ảnh")
print(f"  Train: {len(df_train)}")
print(f"  Val  : {len(df_val)}")
print(f"  Tổng : {len(df_train) + len(df_val)}")

# --- Kiểm tra không trùng ảnh ---
train_set = set(df_train['image_file'])
val_set = set(df_val['image_file'])
intersection = train_set & val_set
print("\n🔍 Kiểm tra trùng ảnh giữa train và val:")
print(f"  Số ảnh trùng: {len(intersection)}")
assert len(intersection) == 0, "❌ Có ảnh bị trùng giữa train và val!"

# --- Kiểm tra tỉ lệ ảnh có/không có khối u ---
def print_tumor_ratio(df, name):
    yes = df['has_tumor'].sum()
    no = len(df) - yes
    print(f"\n🦠 {name}")
    print(f"  Ảnh có khối u    : {yes} ({yes / len(df) * 100:.2f}%)")
    print(f"  Ảnh không khối u : {no} ({no / len(df) * 100:.2f}%)")

print_tumor_ratio(df_train, "Train")
print_tumor_ratio(df_val, "Val")

# --- Kiểm tra phân phối nhãn ---
def compute_distribution(df):
    labels = np.array(df['multi_label'].tolist())
    return labels.sum(axis=0) / len(df)

train_dist = compute_distribution(df_train)
val_dist = compute_distribution(df_val)

print("\n📈 Phân phối nhãn theo class:")
for i, (t, v) in enumerate(zip(train_dist, val_dist)):
    diff = abs(t - v) * 100
    print(f"  Class {i}: Train {t:.2%} | Val {v:.2%} | Chênh lệch: {diff:.2f}%")

# --- Trung bình số nhãn mỗi ảnh ---
train_avg = np.array(df_train['multi_label'].tolist()).sum(axis=1).mean()
val_avg = np.array(df_val['multi_label'].tolist()).sum(axis=1).mean()
print(f"\n📌 Trung bình số nhãn/ảnh:")
print(f"  Train: {train_avg:.2f}")
print(f"  Val  : {val_avg:.2f}")

# --- Vẽ biểu đồ ---
plt.figure(figsize=(10, 4))
x = np.arange(NUM_CLASSES)
bar_width = 0.35
plt.bar(x - bar_width/2, train_dist, width=bar_width, label='Train')
plt.bar(x + bar_width/2, val_dist, width=bar_width, label='Val')
plt.xlabel('Class ID')
plt.ylabel('Tỉ lệ xuất hiện')
plt.title('Phân phối nhãn giữa Train và Val')
plt.xticks(x)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
