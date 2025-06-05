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
        self.score_threshold = 0.3  # hoặc 0.1 tuỳ yêu cầu

        # Tự động lấy số class và số điểm polygon từ output shape
        output_shape = self.output_details[0]["shape"]  # (1, 45, 2100)
        feature_dim = output_shape[1]  # 45
        # YOLOv8-seg: [x, y, w, h, obj, cls1, ..., clsn, mask1, ..., maskN, poly1, ..., polyM]
        # Đoạn đầu: 4 bbox + 1 obj + n class + n_mask + n_polygon
        # Để xác định số class và số điểm polygon:
        # Giả sử số mask coeffs = 32 (YOLOv8 mặc định), số polygon points = (feature_dim - 4 - 1 - n_class - 32) // 2
        # Thử các giá trị hợp lý cho n_class (ví dụ từ 1 đến 100)
        found = False
        for n_class in range(1, 100):
            n_mask = 32  # YOLOv8 mặc định
            remain = feature_dim - 4 - 1 - n_class - n_mask
            if remain >= 6 and remain % 2 == 0:
                self.class_count = n_class
                self.polygon_points = remain // 2
                found = True
                break
        if not found:
            raise RuntimeError("Không thể tự động xác định số class và số điểm polygon từ output shape!")
        
        # Tìm index của output detection và prototype mask
        self.detection_output_index = None
        self.prototype_mask_index = None
        for od in self.output_details:
            shape = od["shape"]
            if len(shape) == 3 and shape[1] == feature_dim:  # (1, 45, 2100)
                self.detection_output_index = od["index"]
            elif len(shape) == 4 and shape[1] == 80 and shape[2] == 80 and shape[3] == 32:
                self.prototype_mask_index = od["index"]
        if self.detection_output_index is None or self.prototype_mask_index is None:
            raise RuntimeError("Không tìm thấy output detection hoặc prototype mask phù hợp!")

    def prepare(self):
        # Kiểm tra input/output details
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)

    def predict(self, image):
        # Nếu là PIL.Image thì chuyển sang numpy array
        if hasattr(image, "convert"):
            image = np.array(image.convert("RGB"))
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        input_data = np.expand_dims(image.astype(np.float32), axis=0) / 255.0
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        assert image.shape == (self.input_height, self.input_width, 3), "Ảnh đầu vào sai kích thước!"
        # Lấy raw outputs
        detections = self.interpreter.get_tensor(self.detection_output_index)[0]  # (45, 2100)
        prototype_mask = self.interpreter.get_tensor(self.prototype_mask_index)[0]  # (80, 80, 32)

        results = []
        for i in range(detections.shape[0]):
            det = detections[i]
            bbox = det[:4]
            x, y, w, h = bbox
            x1 = (x - w/2) * self.input_width
            y1 = (y - h/2) * self.input_height
            x2 = (x + w/2) * self.input_width
            y2 = (y + h/2) * self.input_height

            class_scores = det[4:4+self.class_count]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

            # Polygon points (giả sử sau mask_coeffs là polygon, chuẩn hóa [0,1])
            mask_coeff = det[4+self.class_count : 4+self.class_count+32]
            polygon = det[4+self.class_count+32 : 4+self.class_count+32+self.polygon_points*2]
            polygon_pixel = []
            for j, v in enumerate(polygon):
                if abs(v) > 1.5:
                    polygon_pixel.append(float(v))
                else:
                    polygon_pixel.append(float(v) * (self.input_width if j % 2 == 0 else self.input_height))

            # Đúng format metrics cần: [class_id, score, x1, y1, x2, y2, x3, y3, ..., xn, yn]
            instance = [class_id, score] + polygon_pixel
            results.append(instance)

        return results