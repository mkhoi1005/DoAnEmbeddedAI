import os

# Đường dẫn tới thư mục
image_dir = 'datasets/BTXRD/images/train'
label_dir = 'datasets/BTXRD/labels/train'

# Bắt đầu từ ID 1868 trở đi
start_index = 1868
deleted = 0

for i in range(start_index, 4000):  # Dư phòng ID lớn hơn 3746
    base_name = f"IMG{i:06d}"
    image_path = os.path.join(image_dir, base_name + ".jpeg")
    label_path = os.path.join(label_dir, base_name + ".txt")

    removed = False

    if os.path.exists(image_path):
        os.remove(image_path)
        removed = True

    if os.path.exists(label_path):
        os.remove(label_path)
        removed = True

    if removed:
        deleted += 1

print(f"Đã xóa {deleted} ảnh và label bắt đầu từ ID {start_index}.")
