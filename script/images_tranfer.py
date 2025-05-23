import os
import shutil

labels_dir = 'datasets/BTXRD/labels/train'
all_images_dir = 'datasets/BTXRD/images/all'
train_images_dir = 'datasets/BTXRD/images/train'

# Đếm số ảnh được copy
copied_count = 0
missing_count = 0

for label_file in os.listdir(labels_dir):
    if not label_file.endswith('.txt'):
        continue

    base_name = os.path.splitext(label_file)[0]
    image_name = base_name + '.jpeg'
    src_image_path = os.path.join(all_images_dir, image_name)
    dst_image_path = os.path.join(train_images_dir, image_name)

    if not os.path.exists(dst_image_path):
        if os.path.exists(src_image_path):
            shutil.copy2(src_image_path, dst_image_path)
            copied_count += 1
        else:
            print(f"Thiếu ảnh: {image_name}")
            missing_count += 1

print(f"Đã sao chép {copied_count} ảnh còn thiếu vào images/train.")
if missing_count > 0:
    print(f"Có {missing_count} ảnh bị thiếu trong images/all.")
