# kiểm tra xem các file ảnh và nhãn có khớp 1-1 không
import os

image_dir = "datasets/BTXRD/images/train"
label_dir = "datasets/BTXRD/labels/train"

# Lấy danh sách tên file (không có đuôi mở rộng)
image_files = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith(".jpeg")}
label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith(".txt")}

# Tìm sự khác biệt
missing_labels = image_files - label_files
missing_images = label_files - image_files

# In kết quả
if not missing_labels and not missing_images:
    print("✅ Ảnh và nhãn đã khớp 1-1 hoàn toàn.")
else:
    if missing_labels:
        print("⚠️ Các ảnh bị thiếu nhãn:")
        for f in sorted(missing_labels):
            print(f"{f}.jpeg")
    if missing_images:
        print("⚠️ Các nhãn bị thiếu ảnh:")
        for f in sorted(missing_images):
            print(f"{f}.txt")
