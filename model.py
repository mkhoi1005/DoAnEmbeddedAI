import numpy as np
import torch
import cv2

class Model(object):
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.input_width = 320
        self.input_height = 320

        # In tổng số tham số mô hình
        total_params = sum(p.numel() for p in self.model.model.parameters())
        print(f"Total model parameters: {total_params:,}")

    def prepare(self):
        return None

    def resample_polygon(self, poly, num_points):
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        if len(pts) == num_points:
            return pts.flatten().tolist()
        if len(pts) < 2 or num_points < 2:
            return []
        dists = np.sqrt(np.sum(np.diff(np.vstack([pts, pts[0]]), axis=0)**2, axis=1))
        total = np.sum(dists)
        if total == 0 or num_points == 0:
            return []
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
            ratio = acc / dists[i] if dists[i] > 0 else 0
            new_pt = pts[i] + ratio * (pts[(i+1)%len(pts)] - pts[i])
            new_pts.append(new_pt)
        return np.array(new_pts, dtype=np.float32).flatten().tolist()

    def predict(self, image, labels=None):
        if hasattr(image, "convert"):
            image = image.convert("RGB")
        img = np.array(image)
        img_resized = cv2.resize(img, (self.input_width, self.input_height))
        results = self.model.predict(img_resized, imgsz=320, conf=0.3, device=self.device, verbose=False)
        results = results[0]
        output = []
        if hasattr(results, "masks") and results.masks is not None:
            for i, mask in enumerate(results.masks.data):
                mask_np = mask.cpu().numpy().astype(np.uint8)
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if len(contours) == 0:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                cnt = cnt.squeeze()
                if cnt.ndim == 1:
                    cnt = cnt[np.newaxis, :]
                if len(cnt) < 3:
                    continue
                num_points = 40
                polygon_resampled = self.resample_polygon(cnt, num_points)
                class_index = int(results.boxes.cls[i].cpu().numpy())
                score = float(results.boxes.conf[i].cpu().numpy())
                instance = [class_index, score, 0, 0, 0, 0] + polygon_resampled
                output.append(instance)
        return output