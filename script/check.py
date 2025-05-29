import os
import numpy as np

label_dir = r"D:\EmbededAI\datasets\BTXRD\labels\train"
small_objects = []
try:
    for label_file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, label_file)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    try:
                        parts = line.strip().split()
                        if len(parts) < 3:  # Ensure enough values for class_id and points
                            print(f"Skipping invalid line in {label_file}: {line.strip()}")
                            continue
                        class_id = int(parts[0])
                        points = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                        x_min, y_min = points.min(axis=0)
                        x_max, y_max = points.max(axis=0)
                        area = (x_max - x_min) * (y_max - y_min)
                        if area < 0.01:
                            small_objects.append((label_file, class_id, area))
                    except Exception as e:
                        print(f"Error processing line in {label_file}: {e}")
        except Exception as e:
            print(f"Error reading file {label_file}: {e}")
except FileNotFoundError:
    print(f"Directory not found: {label_dir}")
except Exception as e:
    print(f"Error accessing directory: {e}")

print(f"Found {len(small_objects)} small objects.")