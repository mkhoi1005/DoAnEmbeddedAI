import os

def check_classes(label_dir, expected_classes):
    found_classes = set()

    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(label_dir, filename), "r") as f:
                for line in f:
                    if line.strip():  # Bỏ dòng trống
                        class_id = int(line.split()[0])
                        found_classes.add(class_id)

    missing_classes = expected_classes - found_classes
    print(f"Found {len(found_classes)} classes in {label_dir}: {sorted(found_classes)}")
    if missing_classes:
        print(f"Missing classes: {sorted(missing_classes)}")
    else:
        print("✅ All classes present!")

# Cấu hình thư mục
expected = set(range(9))  # 0 đến 8
check_classes("datasets/BTXRD/labels/train", expected)
check_classes("datasets/BTXRD/labels/val", expected)
