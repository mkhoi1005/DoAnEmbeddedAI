import os
import shutil

all_dir = "datasets/BTXRD/labels/all"
val_dir = "datasets/BTXRD/labels/val"
train_dir = "datasets/BTXRD/labels/train"
os.makedirs(train_dir, exist_ok=True)

val_files = {os.path.splitext(f)[0] for f in os.listdir(val_dir) if f.endswith(".txt")}

for file in os.listdir(all_dir):
    if file.endswith(".txt"):
        file_id = os.path.splitext(file)[0]
        if file_id not in val_files:
            shutil.copy2(os.path.join(all_dir, file), os.path.join(train_dir, file))

print("✅ Đã tạo tập train từ all, bỏ qua các ảnh đã có trong val.")
