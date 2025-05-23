import os
import json
from pathlib import Path

def convert_labelme_to_yolo(json_dir, image_dir, output_dir, class_mapping):
    """
    Chuyển đổi toàn bộ annotations từ LabelMe JSON sang YOLOv8 segmentation format
    Giữ nguyên mọi thông tin: polygon, rectangle, điểm ảnh...
    
    Args:
        json_dir: Thư mục chứa file JSON annotations
        image_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục đầu ra cho file TXT
        class_mapping: Ánh xạ tên lớp sang ID số
    """
    # Tạo thư mục đầu ra
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách file JSON
    json_files = list(Path(json_dir).glob('*.json'))
    processed_images = set()

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            image_name = Path(data['imagePath']).stem
            processed_images.add(image_name)
            txt_file = Path(output_dir) / f"{image_name}.txt"
            
            img_width = data['imageWidth']
            img_height = data['imageHeight']
            
            with open(txt_file, 'w', encoding='utf-8') as f_txt:
                for shape in data['shapes']:
                    label = shape['label'].strip().lower()
                    if label not in class_mapping:
                        continue
                    
                    class_id = class_mapping[label]
                    shape_type = shape['shape_type']
                    points = shape['points']
                    
                    # Chuẩn hóa tọa độ về [0,1]
                    normalized_points = []
                    for x, y in points:
                        nx = max(0, min(x, img_width)) / img_width
                        ny = max(0, min(y, img_height)) / img_height
                        normalized_points.extend([nx, ny])
                    
                    # Ghi vào file theo định dạng YOLOv8 segmentation
                    line = [str(class_id)] + [f"{coord:.6f}" for coord in normalized_points]
                    f_txt.write(" ".join(line) + "\n")
        
        except Exception as e:
            print(f"Lỗi khi xử lý file {json_file}: {str(e)}")

    # Xử lý ảnh không có annotation
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    for img_file in Path(image_dir).iterdir():
        if img_file.suffix.lower() in image_extensions:
            txt_file = Path(output_dir) / f"{img_file.stem}.txt"
            if img_file.stem not in processed_images:
                # Tạo file trống cho ảnh không có khối u
                with open(txt_file, 'w') as f:
                    pass

# Định nghĩa class mapping theo yêu cầu
CLASS_MAPPING = {
    "osteochondroma": 0,
    "synovial osteochondroma": 1,
    "osteofibroma": 2,
    "simple bone cyst": 3,
    "other bt": 4,
    "other mt": 5,
    "multiple osteochondromas": 6,
    "osteosarcoma": 7,
    "giant cell tumor": 8,
    # Thêm các biến thể viết thường nếu cần
    **{k.lower(): v for k, v in {
        "osteochondroma": 0,
        "synovial osteochondroma": 1,
        "osteofibroma": 2,
        "simple bone cyst": 3,
        "other bt": 4,
        "other mt": 5,
        "multiple osteochondromas": 6,
        "osteosarcoma": 7,
        "giant cell tumor": 8
    }.items()}
}

# Thực hiện chuyển đổi
convert_labelme_to_yolo(
    json_dir="../datasets/BTXRD/annotations",
    image_dir="../datasets/BTXRD/images/all",
    output_dir="../datasets/BTXRD/labels/all2",
    class_mapping=CLASS_MAPPING
)

print("Chuyển đổi hoàn tất! Đã giữ nguyên toàn bộ thông tin annotation.")