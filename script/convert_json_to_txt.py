import os
import shutil
# chuyển json sang txt cho tập train/val

# Thư mục chứa ảnh
train_images_dir = r"D:\EmbededAI\datasets\BTXRD\images\train"
val_images_dir = r"D:\EmbededAI\datasets\BTXRD\images\val"

# Thư mục nhãn gốc (đã tạo)
all_labels_dir = r"D:\EmbededAI\datasets\BTXRD\labels\all"

# Thư mục đích để copy nhãn tương ứng
train_labels_dir = r"D:\EmbededAI\datasets\BTXRD\labels\train"
val_labels_dir = r"D:\EmbededAI\datasets\BTXRD\labels\val"

# Tạo thư mục nếu chưa tồn tại
os.makedirs(train_labels_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

def copy_labels(images_dir, labels_dir):
    for filename in os.listdir(images_dir):
        if filename.endswith(".jpeg"):
            base_name = os.path.splitext(filename)[0]
            label_file = base_name + ".txt"
            label_path = os.path.join(all_labels_dir, label_file)
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(labels_dir, label_file))
            else:
                print(f"[!] Không tìm thấy nhãn cho ảnh: {filename}")

# Copy nhãn tương ứng
copy_labels(train_images_dir, train_labels_dir)
copy_labels(val_images_dir, val_labels_dir)

print("✅ Đã copy các file nhãn tương ứng theo train/val.")
