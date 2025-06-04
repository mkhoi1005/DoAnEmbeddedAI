import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2
import os

onnx_model_path = "best.onnx"
img_path = "BTXRD/images/val/IMG000071.jpeg"
output_dir = "output_onnx"
os.makedirs(output_dir, exist_ok=True)

NUM_CLASSES = 9

# Load ảnh và preprocess giống YOLOv8
img = Image.open(img_path).convert("RGB").resize((320, 320))
img_np = np.array(img).astype(np.float32) / 255.0  # chuẩn hóa
img_np = np.transpose(img_np, (2, 0, 1))  # CHW
img_np = np.expand_dims(img_np, axis=0)   # batch

# Load ONNX model
session = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name

# Predict
outputs = session.run(None, {input_name: img_np})
print("ONNX raw outputs:", [o.shape for o in outputs])

# Debug: kiểm tra max obj và max class score
boxes_raw = outputs[0][0]  # (45, 2100)
print("Max obj:", np.max(boxes_raw[4, :]))
print("Max class score:", np.max(boxes_raw[5:5+NUM_CLASSES, :]))

img_vis = np.array(img)
num_box = 0

# Mỗi cột là 1 detection: [x, y, w, h, obj, class1,...,class9, mask...]
for i in range(boxes_raw.shape[1]):
    det = boxes_raw[:, i]
    x, y, w, h = det[:4]
    obj = det[4]
    class_scores = det[5:5+NUM_CLASSES]
    class_id = np.argmax(class_scores)
    score = obj * class_scores[class_id]
    if score < 0.05:  # giảm ngưỡng để kiểm tra
        continue
    x1 = int(x - w/2)
    y1 = int(y - h/2)
    x2 = int(x + w/2)
    y2 = int(y + h/2)
    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_vis, f"{int(class_id)} {score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    num_box += 1

print(f"Num boxes detected: {num_box}")

cv2.imshow("ONNX Predict", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(output_dir, "IMG000071_onnx.png"), img_vis)