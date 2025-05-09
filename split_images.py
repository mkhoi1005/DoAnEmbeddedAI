import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn gốc
image_dir = "datasets/BTXRD/images"
label_dir = "datasets/BTXRD/labels"

# Tạo thư mục đích
for split in ['train', 'val']:
    os.makedirs(f"{image_dir}/{split}", exist_ok=True)
    os.makedirs(f"{label_dir}/{split}", exist_ok=True)

# Lấy danh sách ảnh (không đuôi)
image_files = sorted([f[:-5] for f in os.listdir(image_dir) if f.endswith(".jpeg")])

# Tách tumor và normal
tumor_images = image_files[:1867]
normal_images = image_files[1867:]

# Chia train/val theo từng loại
tumor_train, tumor_val = train_test_split(tumor_images, test_size=0.2, random_state=42)
normal_train, normal_val = train_test_split(normal_images, test_size=0.2, random_state=42)

# Gộp lại
train_files = tumor_train + normal_train
val_files = tumor_val + normal_val

# Hàm copy ảnh và label
def copy_files(file_list, split):
    for name in file_list:
        img_src = os.path.join(image_dir, f"{name}.jpeg")
        lbl_src = os.path.join(label_dir, f"{name}.txt")

        img_dst = os.path.join(image_dir, split, f"{name}.jpeg")
        lbl_dst = os.path.join(label_dir, split, f"{name}.txt")

        if os.path.exists(img_src):
            shutil.copy(img_src, img_dst)
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, lbl_dst)

# Thực hiện sao chép
copy_files(train_files, 'train')
copy_files(val_files, 'val')

print(f"✔ Đã tách {len(train_files)} ảnh vào 'train' và {len(val_files)} ảnh vào 'val'")
