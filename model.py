import numpy as np
import tflite_runtime.interpreter as tflite

class Model(object):
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]
        self.score_threshold = 0.1  # Đặt thấp để không bỏ sót instance

        # Tự động lấy số class và số điểm polygon từ output shape
        output_shape = self.output_details[0]["shape"]  # (1, feature_dim, N)
        feature_dim = output_shape[1]
        self.n_mask = 32  # YOLOv8 mặc định
        found = False
        for n_class in range(1, 100):
            remain = feature_dim - 4 - 1 - n_class - self.n_mask
            if remain >= 6 and remain % 2 == 0:
                self.class_count = n_class
                self.polygon_points = remain // 2
                found = True
                break
        if not found:
            raise RuntimeError("Không thể tự động xác định số class và số điểm polygon từ output shape!")

        # Tìm index output detection
        self.detection_output_index = None
        for od in self.output_details:
            shape = od["shape"]
            if len(shape) == 3 and shape[1] == feature_dim:
                self.detection_output_index = od["index"]
        if self.detection_output_index is None:
            raise RuntimeError("Không tìm thấy output detection phù hợp!")

    def prepare(self):
        return None

    def predict(self, image):
        # Nếu là PIL.Image thì chuyển sang numpy array
        if hasattr(image, "convert"):
            image = np.array(image.convert("RGB"))
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        input_data = np.expand_dims(image.astype(np.float32), axis=0) / 255.0

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.detection_output_index)  # (1, feature_dim, N)
        preds = output_data[0].transpose(1, 0)  # (N, feature_dim)

        results = []
        for det in preds:
            obj_score = det[4]
            class_scores = det[5:5+self.class_count]
            class_id = int(np.argmax(class_scores))
            score = obj_score * class_scores[class_id]
            if score < self.score_threshold:
                continue
            # bbox (xywh normalized)
            x, y, w, h = det[0], det[1], det[2], det[3]
            x1 = (x - w/2) * self.input_width
            y1 = (y - h/2) * self.input_height
            x2 = (x + w/2) * self.input_width
            y2 = (y + h/2) * self.input_height
            # polygon (chuẩn hóa [0,1]), nhân về pixel
            polygon = det[5+self.class_count+self.n_mask : 5+self.class_count+self.n_mask+self.polygon_points*2]
            polygon_pixel = []
            for i, v in enumerate(polygon):
                if abs(v) > 1.5:
                    polygon_pixel.append(float(v))
                else:
                    polygon_pixel.append(float(v) * (self.input_width if i % 2 == 0 else self.input_height))
            # Đúng format metrics cần: [class_index, score, x1, y1, x2, y2, x3, y3, ..., xn, yn]
            instance = [class_id, float(score)] + [x1, y1, x2, y2] + polygon_pixel
            results.append(instance)
        return results