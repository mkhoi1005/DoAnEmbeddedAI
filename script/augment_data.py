import os
import cv2
from tqdm import tqdm
import albumentations as A

# Đường dẫn
image_dir = "../datasets/BTXRD/images/train"
label_dir = "../datasets/BTXRD/labels/train"  # polygon labels

# Tăng cường lưu đè vào image_dir
num_augments = 5

# Tạo pipeline tăng cường
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=15, p=0.3),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.Blur(blur_limit=3, p=0.2)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# Hàm chuyển polygon -> YOLO bbox
def polygon_to_yolo_bbox(polygon):
    xs = polygon[::2]
    ys = polygon[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return [x_center, y_center, width, height]

# Lặp qua ảnh
image_files = [f for f in os.listdir(image_dir) if f.endswith((".jpg", ".jpeg", ".png"))]

for img_file in tqdm(image_files):
    base_name = os.path.splitext(img_file)[0]
    img_path = os.path.join(image_dir, img_file)
    label_path = os.path.join(label_dir, base_name + ".txt")

    if not os.path.exists(label_path):
        print(f"Bỏ qua ảnh {img_file} vì không có file label.")
        continue

    image = cv2.imread(img_path)
    if image is None:
        print(f"Không thể đọc ảnh {img_path}")
        continue

    height, width, _ = image.shape

    bboxes = []
    class_labels = []
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue  # Không hợp lệ

        # polygon → bbox
        bbox = polygon_to_yolo_bbox(coords)
        bboxes.append(bbox)
        class_labels.append(cls_id)

    if len(bboxes) == 0:
        print(f"Bỏ qua ảnh {img_file} vì không có bbox.")
        continue

    for i in range(num_augments):
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_labels = augmented['class_labels']

            aug_img_name = f"{base_name}_aug{i}.jpg"
            aug_img_path = os.path.join(image_dir, aug_img_name)
            cv2.imwrite(aug_img_path, aug_image)

            aug_label_path = os.path.join(label_dir, f"{base_name}_aug{i}.txt")
            with open(aug_label_path, 'w') as f:
                for cls, bbox in zip(aug_labels, aug_bboxes):
                    f.write(f"{cls} {' '.join([f'{x:.6f}' for x in bbox])}\n")

        except Exception as e:
            print(f"Lỗi khi augment ảnh {img_file}: {e}")
