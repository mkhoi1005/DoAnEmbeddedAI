from model import Model
from PIL import Image
import numpy as np
import cv2
import os

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

# Predict
preds = model.predict(img)
print("TFLite predict polygons:", preds)

# Hiển thị/lưu ảnh với mask (nếu có mask)
img_np = np.array(img)
for i, poly in enumerate(preds):
    class_id = poly[0]
    pts = np.array(poly[1:], dtype=np.int32).reshape(-1, 2)
    cv2.polylines(img_np, [pts], isClosed=True, color=(0,255,0), thickness=2)
    cv2.putText(img_np, str(class_id), tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

cv2.imshow("TFLite Predict", img_np)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, "IMG000071_tflite.png"), img_np)