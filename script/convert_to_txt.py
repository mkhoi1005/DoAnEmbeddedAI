import os
import shutil
from tqdm import tqdm

def move_labels(img_dir, label_all_dir, label_target_dir):
    os.makedirs(label_target_dir, exist_ok=True)
    image_list = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in tqdm(image_list, desc=f"Copying labels for {img_dir}"):
        name = os.path.splitext(img_file)[0]
        src_label = os.path.join(label_all_dir, f"{name}.txt")
        dst_label = os.path.join(label_target_dir, f"{name}.txt")

        if os.path.exists(src_label):
            shutil.copyfile(src_label, dst_label)
        else:
            # Nếu không có file .txt trong all, vẫn tạo file rỗng
            open(dst_label, 'w').close()

# Đường dẫn
label_all_dir = '../datasets/BTXRD/labels/all'
move_labels('../datasets/BTXRD/images/train', label_all_dir, '../datasets/BTXRD/labels/train')
move_labels('../datasets/BTXRD/images/val', label_all_dir, '../datasets/BTXRD/labels/val')
