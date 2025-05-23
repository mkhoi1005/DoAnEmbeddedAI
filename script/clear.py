import os

image_dir = "../datasets/BTXRD/images/train"
label_dir = "../datasets/BTXRD/labels/train"

for f in os.listdir(image_dir):
    if "_aug" in f:
        os.remove(os.path.join(image_dir, f))

for f in os.listdir(label_dir):
    if "_aug" in f:
        os.remove(os.path.join(label_dir, f))
