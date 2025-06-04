from PIL import Image
import numpy as np
import cv2
import os
from model import Model

# Đường dẫn model và ảnh
tflite_model_path = "best_float32.tflite"
img_path = "BTXRD/images/val/IMG000071.jpeg"
output_dir = "output_tflite"
os.makedirs(output_dir, exist_ok=True)

# Load model
model = Model(model_path=tflite_model_path)
model.prepare()

# Load và resize ảnh
img = Image.open(img_path).convert("RGB").resize((320, 320))
img_np = np.array(img).copy()

# Predict
preds = model.predict(img)
print("TFLite predict polygons:", preds)

# Lấy polygons, mask_coeffs_list, mask_proto
if isinstance(preds, tuple):
    polygons, mask_coeffs_list, mask_proto = preds
else:
    polygons = preds
    mask_coeffs_list = []
    mask_proto = None

# Vẽ kết quả lên ảnh
for i, poly in enumerate(polygons):
    if len(poly) < 6:
        continue  # [class_id, score, x, y, w, h]
    class_id = int(poly[0])
    score = poly[1]
    x, y, w, h = poly[2:6]
    # Chuyển box về pixel
    x1 = int((x - w / 2) * img_np.shape[1])
    y1 = int((y - h / 2) * img_np.shape[0])
    x2 = int((x + w / 2) * img_np.shape[1])
    y2 = int((y + h / 2) * img_np.shape[0])
    cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_np, f"cls:{class_id} {score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # Nếu muốn vẽ mask instance, cần postprocess thêm từ mask_coeffs_list và mask_proto

# Hiển thị và lưu ảnh kết quả
cv2.imshow("TFLite Predict", img_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, "IMG000071_tflite.png"), img_np)