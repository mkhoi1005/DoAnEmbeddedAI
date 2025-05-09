import os
from tqdm import tqdm

# Cấu hình
image_folder = "datasets/BTXRD/images"
annotation_folder = "datasets/BTXRD/Annotations"
label_folder = "datasets/BTXRD/labels"
os.makedirs(label_folder, exist_ok=True)

# Lấy danh sách ảnh
all_images = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))])

# Tạo danh sách annotation có nhãn
annotated_images = set(f.replace('.json', '') for f in os.listdir(annotation_folder) if f.endswith('.json'))

# Duyệt qua tất cả ảnh
for img_file in tqdm(all_images):
    img_name = os.path.splitext(img_file)[0]
    label_path = os.path.join(label_folder, img_name + ".txt")

    # Nếu ảnh không có annotation → tạo file txt rỗng
    if img_name not in annotated_images:
        open(label_path, "w").close()  # Tạo file rỗng
