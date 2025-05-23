import os
import random
import shutil

random.seed(42)

all_img_dir = '../datasets/BTXRD/images/all'
train_img_dir = '../datasets/BTXRD/images/train'
val_img_dir = '../datasets/BTXRD/images/val'

os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)

all_images = [f for f in os.listdir(all_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

total = len(all_images)
train_count = int(0.8 * total)

train_images = all_images[:train_count]
val_images = all_images[train_count:]

for img_name in train_images:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(train_img_dir, img_name))

for img_name in val_images:
    shutil.copy(os.path.join(all_img_dir, img_name), os.path.join(val_img_dir, img_name))

print(f"Chia {train_count} ảnh train và {total - train_count} ảnh val thành công.")
