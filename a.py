from ultralytics import YOLO
from model import Model
from PIL import Image
import numpy as np

# Đường dẫn model
pt_path = "best.pt"
tflite_path = "best_float32.tflite"
img_path = "BTXRD/images/val/IMG000527.jpeg"  # Thay bằng ảnh bạn muốn test
input_size = 320

# Load models
model_pt = YOLO(pt_path)
model_tflite = Model(tflite_path)

# Load ảnh và resize
img = Image.open(img_path).convert("RGB").resize((input_size, input_size))

# Predict với PyTorch (.pt)
results_pt = model_pt(img, imgsz=input_size, verbose=False)[0]
pt_instances = []
if results_pt.masks is not None:
    for i, (mask, box, cls) in enumerate(zip(results_pt.masks.xy, results_pt.boxes.xyxy, results_pt.boxes.cls)):
        flat_poly = mask.flatten().tolist()
        score = float(results_pt.boxes.conf[i])
        pt_instances.append([int(cls), score] + flat_poly)

# Predict với TFLite
results_tflite = model_tflite.predict(img)

# In kết quả
print("=== PyTorch (.pt) predict ===")
for i, ins in enumerate(pt_instances):
    print(f"Instance {i}: {ins[:8]} ...")  # In 8 giá trị đầu

print("\n=== TFLite predict ===")
for i, ins in enumerate(results_tflite):
    print(f"Instance {i}: {ins[:8]} ...")  # In 8 giá trị đầu

print(f"\nSố instance .pt: {len(pt_instances)} | Số instance tflite: {len(results_tflite)}")