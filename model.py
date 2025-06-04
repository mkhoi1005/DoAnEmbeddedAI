import numpy as np
import tensorflow as tf
import cv2

NUM_CLASSES = 9  # Số class theo class_names.txt

class Model(object):
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print("Input details:", self.input_details)
        print("Output details:", self.output_details)

        self.floating_model = self.input_details[0]["dtype"] == np.float32
        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]
        self.score_threshold = 0.3  # Có thể tăng lên 0.5 nếu còn nhiều mask rác

    def prepare(self):
        return None

    def predict(self, image):
        # Resize ảnh về đúng input size của model
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_resized = image.resize((self.input_width, self.input_height))
        img_np = np.array(image_resized).astype(np.float32)
        input_data = np.expand_dims(img_np, axis=0)

        if self.floating_model:
            input_data = input_data / 255.0

        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        output_boxes = self.interpreter.get_tensor(self.output_details[0]["index"])
        output_masks = self.interpreter.get_tensor(self.output_details[1]["index"])

        results = np.squeeze(output_boxes)
        proto = np.squeeze(output_masks)
        # Lấy shape thật của proto
        if proto.ndim == 3:
            h_proto, w_proto, c_proto = proto.shape
        elif proto.ndim == 2:
            h_proto, w_proto = proto.shape
            c_proto = 1
        else:
            raise ValueError("Unexpected proto shape: {}".format(proto.shape))

        instance_dim = 4 + 1 + NUM_CLASSES + 32  # box(4), score(1), class_scores, mask_coeff(32)
        if results.ndim == 1:
            results = results.reshape(-1, instance_dim)
        elif results.shape[1] != instance_dim and results.size % instance_dim == 0:
            results = results.reshape(-1, instance_dim)
        elif results.shape[1] > instance_dim:
            results = results[:, :instance_dim]

        polygons = []
        img_h, img_w = self.input_height, self.input_width

        for i in range(results.shape[0]):
            row = results[i]
            score = row[4]
            class_scores = row[5:5+NUM_CLASSES]
            class_id = int(np.argmax(class_scores))
            if score < self.score_threshold:
                continue
            x1, y1, x2, y2 = row[:4]

            # Nếu box nhỏ (0-1), scale lên ảnh
            if x2 <= 2 and y2 <= 2:
                x1 *= img_w
                x2 *= img_w
                y1 *= img_h
                y2 *= img_h

            mask_coeff = row[5+NUM_CLASSES:5+NUM_CLASSES+32]
            if mask_coeff.shape[0] != 32:
                continue

            mask = np.dot(proto.reshape(-1, 32), mask_coeff)
            mask = mask.reshape(h_proto, w_proto)
            mask = 1 / (1 + np.exp(-mask))

            box_w, box_h = max(int(x2 - x1), 1), max(int(y2 - y1), 1)
            mask_resized = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
            x1i, y1i = max(int(x1), 0), max(int(y1), 0)
            x2i, y2i = min(x1i + box_w, img_w), min(y1i + box_h, img_h)
            h_mask, w_mask = y2i - y1i, x2i - x1i
            # Chỉ gán khi vùng hợp lệ và mask_bin đủ lớn
            if h_mask > 0 and w_mask > 0 and \
               mask_bin.shape[0] >= h_mask and mask_bin.shape[1] >= w_mask and \
               y2i > y1i and x2i > x1i:
                mask_full[y1i:y2i, x1i:x2i] = mask_bin[:h_mask, :w_mask]

            contours, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = contour.squeeze()
                if contour.ndim != 2 or len(contour) < 3:
                    continue
                polygon = [int(class_id)]
                for point in contour:
                    polygon.extend([int(point[0]), int(point[1])])
                polygons.append(polygon)

        return polygons