import os
import yaml
from collections import Counter
from tqdm import tqdm

def count_labels(label_dir):
    if not os.path.exists(label_dir):
        print(f"❌ Không tìm thấy thư mục: {label_dir}")
        return Counter()

    label_count = Counter()
    for label_file in tqdm(os.listdir(label_dir), desc=f"Đếm file trong {label_dir}"):
        if not label_file.endswith('.txt'):
            continue
        path = os.path.join(label_dir, label_file)
        with open(path, 'r') as f:
            for line in f:
                if line.strip():  # tránh dòng trống
                    cls_id = int(line.strip().split()[0])
                    label_count[cls_id] += 1
    return label_count

def load_yaml(yaml_path):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file YAML: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    yaml_path = os.path.join(os.path.dirname(__file__), '..', 'BTXRD.yaml')
    data_yaml = load_yaml(yaml_path)

    # Chuẩn hóa đường dẫn từ YAML: thay \ thành /
    data_yaml['train'] = data_yaml['train'].replace('\\', '/')
    data_yaml['val'] = data_yaml['val'].replace('\\', '/')

    base_path = data_yaml['path']
    train_labels = os.path.normpath(os.path.join(base_path, data_yaml['train'].replace('images', 'labels')))
    val_labels = os.path.normpath(os.path.join(base_path, data_yaml['val'].replace('images', 'labels')))
    
    print(f"Train labels path: {train_labels}")
    print(f"Val labels path: {val_labels}")

    class_names = data_yaml['names']

    print("🔍 Counting labels in training set...")
    train_count = count_labels(train_labels)

    print("\n🔍 Counting labels in validation set...")
    val_count = count_labels(val_labels)

    print("\n📊 Class Distribution:")
    for cls_id, cls_name in class_names.items():
        train = train_count.get(cls_id, 0)
        val = val_count.get(cls_id, 0)
        total = train + val
        print(f"Class {cls_id} ({cls_name}): Train={train}, Val={val}, Total={total}")


if __name__ == '__main__':
    main()
