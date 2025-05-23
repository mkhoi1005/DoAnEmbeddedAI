import os
import shutil
import random

# Đường dẫn thư mục
all_images_dir = 'datasets/BTXRD/images/all'
all_labels_dir = 'datasets/BTXRD/labels/all'
train_images_dir = 'datasets/BTXRD/images/train'
train_labels_dir = 'datasets/BTXRD/labels/train'

# Lấy danh sách ảnh normal từ ID 1868 trở đi
normal_ids = []
for i in range(1868, 4000):  # Dự phòng tới ID lớn hơn nếu có
    base = f"IMG{i:06d}"
    img_path = os.path.join(all_images_dir, base + '.jpeg')
    if os.path.exists(img_path):
        normal_ids.append(base)

# Chọn ngẫu nhiên 30% ảnh normal còn lại
selected_ids = random.sample(normal_ids, int(0.3 * len(normal_ids)))

# Copy ảnh và nhãn tương ứng
count_copied = 0
for base in selected_ids:
    src_img = os.path.join(all_images_dir, base + '.jpeg')
    dst_img = os.path.join(train_images_dir, base + '.jpeg')

    src_label = os.path.join(all_labels_dir, base + '.txt')
    dst_label = os.path.join(train_labels_dir, base + '.txt')

    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)
    
    if os.path.exists(src_label):
        shutil.copy2(src_label, dst_label)

    count_copied += 1

print(f"Đã sao chép {count_copied} ảnh normal (40%) từ all sang train.")
