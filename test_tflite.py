import numpy as np
import cv2
import tensorflow.lite as tflite
import json

# Đường dẫn model, ảnh, annotation, class
model_path = "calibrate.tflite"
image_path = "BTXRD/images/val/IMG000071.jpeg"
anno_path = "BTXRD/images/annotations/IMG000071.json"
class_path = "BTXRD/images/class_names.txt"

# 1. Đọc class_names.txt để ánh xạ label -> index
class_map = {}
with open(class_path, encoding="utf-8") as f:
    for line in f:
        if ':' in line:
            idx, name = line.strip().split(':', 1)
            class_map[name.strip()] = int(idx.strip())

# 2. Đọc annotation và chuẩn hóa về 320x320, lấy cả label và index
with open(anno_path, encoding="utf-8") as f:
    anno = json.load(f)
img_h, img_w = anno["imageHeight"], anno["imageWidth"]
anno_instances = []
for shape in anno["shapes"]:
    if shape["shape_type"] == "polygon":
        poly = []
        for x, y in shape["points"]:
            x320 = x * 320 / img_w
            y320 = y * 320 / img_h
            poly.append([x320, y320])
        label = shape.get("label", "").strip()
        class_index = class_map.get(label, -1)
        anno_instances.append({
            "class_index": class_index,
            "label": label,
            "polygon": np.array(poly, dtype=np.float32)
        })

def resample_polygon(poly, num_points):
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    if len(pts) == num_points:
        return pts.flatten().tolist()
    dists = np.sqrt(np.sum(np.diff(np.vstack([pts, pts[0]]), axis=0)**2, axis=1))
    total = np.sum(dists)
    if total == 0:
        return np.tile(pts[0], (num_points, 1)).flatten().tolist()
    step = total / num_points
    new_pts = [pts[0]]
    acc = 0
    i = 0
    for _ in range(1, num_points):
        acc += step
        while acc > dists[i]:
            acc -= dists[i]
            i += 1
            if i >= len(pts):
                i = 0
        ratio = acc / dists[i]
        new_pt = pts[i] + ratio * (pts[(i+1)%len(pts)] - pts[i])
        new_pts.append(new_pt)
    return np.array(new_pts, dtype=np.float32).flatten().tolist()

def procrustes_align_polygon(predict_poly, anno_poly):
    pts_pred = np.array(predict_poly, dtype=np.float32).reshape(-1, 2)
    pts_anno = np.array(anno_poly, dtype=np.float32).reshape(-1, 2)
    c_pred = np.mean(pts_pred, axis=0)
    c_anno = np.mean(pts_anno, axis=0)
    pts_pred_centered = pts_pred - c_pred
    pts_anno_centered = pts_anno - c_anno
    norm_pred = np.sqrt(np.sum(pts_pred_centered**2))
    norm_anno = np.sqrt(np.sum(pts_anno_centered**2))
    pts_pred_scaled = pts_pred_centered / (norm_pred + 1e-8)
    pts_anno_scaled = pts_anno_centered / (norm_anno + 1e-8)
    U, _, Vt = np.linalg.svd(np.dot(pts_anno_scaled.T, pts_pred_scaled))
    R = np.dot(U, Vt)
    pts_pred_rotated = np.dot(pts_pred_scaled, R.T)
    pts_pred_final = pts_pred_rotated * norm_anno + c_anno
    pts_pred_final = np.clip(pts_pred_final, 0, 319)
    return pts_pred_final.flatten().tolist()

def ensure_same_direction(predict_poly, anno_poly):
    def polygon_area(pts):
        x = pts[:,0]
        y = pts[:,1]
        return 0.5 * np.sum(x[:-1]*y[1:] - x[1:]*y[:-1])
    pts_pred = np.array(predict_poly, dtype=np.float32).reshape(-1, 2)
    pts_anno = np.array(anno_poly, dtype=np.float32).reshape(-1, 2)
    pts_pred_closed = np.vstack([pts_pred, pts_pred[0]])
    pts_anno_closed = np.vstack([pts_anno, pts_anno[0]])
    area_pred = polygon_area(pts_pred_closed)
    area_anno = polygon_area(pts_anno_closed)
    if np.sign(area_pred) != np.sign(area_anno):
        pts_pred = pts_pred[::-1]
    return pts_pred.flatten().tolist()

def optimal_reorder_polygon_start(predict_poly, anno_poly):
    pts_pred = np.array(predict_poly, dtype=np.float32).reshape(-1, 2)
    pts_anno = np.array(anno_poly, dtype=np.float32).reshape(-1, 2)
    n = len(pts_pred)
    min_sum = None
    best_pts = pts_pred
    for shift in range(n):
        pts_shifted = np.roll(pts_pred, -shift, axis=0)
        dist_sum = np.sum(np.linalg.norm(pts_shifted - pts_anno, axis=1))
        if (min_sum is None) or (dist_sum < min_sum):
            min_sum = dist_sum
            best_pts = pts_shifted
    return best_pts.flatten().tolist()

# 3. Load model và ảnh
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img = cv2.imread(image_path)
img_resized = cv2.resize(img, (320, 320))
input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
input_data = input_data / 255.0

# 4. Chạy model và lấy output thô
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("=== Output thô của TFLite ===")
print("Shape:", output_data.shape)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
print(output_data)

# 5. Lấy mask từ output segmentation (giả sử foreground là kênh cuối)
if output_data.ndim == 4:
    mask = output_data[0, :, :, -1]
elif output_data.ndim == 3:
    mask = output_data[:, :, -1]
else:
    raise RuntimeError("Không xác định được shape output_data!")

mask = (mask > 0).astype(np.uint8)  # nhị phân hóa

# 6. Tìm contour lớn nhất và chuẩn hóa polygon predict
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
polygon_aligned = None
if len(contours) > 0:
    cnt = max(contours, key=cv2.contourArea)
    cnt = cnt.squeeze()
    if cnt.ndim == 1:
        cnt = cnt[np.newaxis, :]
    # Nội suy polygon predict về cùng số điểm với annotation
    anno_poly = anno_instances[0]["polygon"].flatten().tolist()
    polygon_resampled = resample_polygon(cnt, len(anno_poly)//2)
    # Không cần scale vì mask và annotation đều ở 320x320
    polygon_aligned = procrustes_align_polygon(polygon_resampled, anno_poly)
    polygon_aligned = ensure_same_direction(polygon_aligned, anno_poly)
    polygon_aligned = optimal_reorder_polygon_start(polygon_aligned, anno_poly)
    print("\nBest predict instance (chuẩn hóa):")
    print(polygon_aligned)
    pred_pts = np.array(polygon_aligned).reshape(-1, 2)
    anno_pts = np.array(anno_poly).reshape(-1, 2)
    mean_dist = np.mean(np.linalg.norm(pred_pts - anno_pts, axis=1))
    print(f"Mean distance giữa predict và annotation: {mean_dist:.2f} pixels")
else:
    print("\nKhông có contour nào được phát hiện từ mask predict!")

# 7. In annotation để đối chiếu
for i, ins in enumerate(anno_instances):
    poly_flat = ins["polygon"].flatten().tolist()
    print(f"\nAnnotation instance {i}:")
    print([ins["class_index"]] + poly_flat)

# 8. Vẽ mask predict (đỏ trong suốt) và annotation (xanh lá) lên ảnh
img_vis = img_resized.copy()
# Vẽ annotation (màu xanh lá)
for ins in anno_instances:
    pts = ins["polygon"].astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
# Vẽ mask segment predict (đỏ trong suốt)
mask_vis = (mask * 255).astype(np.uint8)
mask_vis = cv2.resize(mask_vis, (img_vis.shape[1], img_vis.shape[0]), interpolation=cv2.INTER_NEAREST)
color_mask = np.zeros_like(img_vis)
color_mask[:, :, 0] = mask_vis  # tô kênh đỏ
alpha = 0.4
img_vis = cv2.addWeighted(color_mask, alpha, img_vis, 1 - alpha, 0)
# Vẽ polygon predict (đường viền đỏ)
if polygon_aligned is not None:
    pts_pred = np.array(polygon_aligned, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(img_vis, [pts_pred], isClosed=True, color=(0, 0, 255), thickness=2)
# Hiển thị hoặc lưu ảnh
cv2.imshow("Predict mask (red) vs Annotation (green)", img_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()