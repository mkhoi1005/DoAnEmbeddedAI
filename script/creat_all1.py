import os
import shutil

# Đường dẫn tới các thư mục
all_dir = '../datasets/BTXRD/images/all'
test_dir = '../datasets/BTXRD/images/test'
all1_dir = '../datasets/BTXRD/images/all1'

# Tạo thư mục all1 nếu chưa tồn tại
os.makedirs(all1_dir, exist_ok=True)

# Lấy tên file trong thư mục test
test_filenames = set(os.listdir(test_dir))

# Lặp qua tất cả file trong thư mục all
for filename in os.listdir(all_dir):
    if filename not in test_filenames:
        src_path = os.path.join(all_dir, filename)
        dst_path = os.path.join(all1_dir, filename)
        shutil.copy2(src_path, dst_path)

print("Hoàn tất sao chép các ảnh không nằm trong thư mục test.")
