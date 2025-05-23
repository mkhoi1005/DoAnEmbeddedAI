import os
import shutil
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# Thư mục nguồn và đích
src_folder = "new_datasets/BTXRD/val"
dst_folder = "datasets/BTXRD/images/val"
os.makedirs(dst_folder, exist_ok=True)

# Copy tất cả file .jpeg
for filename in os.listdir(src_folder):
    if filename.lower().endswith(".jpeg"):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        shutil.copy2(src_path, dst_path)

print("Đã copy xong tất cả ảnh .jpeg vào thư mục đích.")
