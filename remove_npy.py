import os

ann_dir = "BTXRD/annotations"

for fname in os.listdir(ann_dir):
    if fname.endswith(".npy"):
        file_path = os.path.join(ann_dir, fname)
        os.remove(file_path)
        print(f"Deleted {fname}")