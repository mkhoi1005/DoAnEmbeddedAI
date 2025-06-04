import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.BOX_COORD_NUM = 4

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(self.input_details)

        self.floating_model = self.input_details[0]["dtype"] == np.float32

        self.input_height = self.input_details[0]["shape"][1]
        self.input_width = self.input_details[0]["shape"][2]

        self.max_box_count = self.output_details[0]["shape"][2]
        self.output_dim = self.output_details[0]["shape"][1]

        # Số class = output_dim - 4 (box) - 32 (mask)
        self.mask_dim = 32
        self.class_count = self.output_dim - self.BOX_COORD_NUM - self.mask_dim

        self.input_mean = 0.0
        self.input_std = 1.0
        self.keypoint_count = 0
        self.score_threshold = 0.3

    def prepare(self):
        return None

    def predict(self, image):
        # Chuyển ảnh PIL sang numpy nếu cần
        if hasattr(image, 'convert'):
            image = np.array(image, dtype=np.float32)
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)
        if image.shape[-1] != 3:
            raise ValueError("Input image must have 3 channels (RGB)")
        input_data = np.expand_dims(image, axis=0)  # (1, H, W, C)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()

        # Lấy cả 2 output tensor
        det_out = self.interpreter.get_tensor(self.output_details[0]["index"])  # (1, 45, 2100)
        mask_proto = self.interpreter.get_tensor(self.output_details[1]["index"])  # (1, 32, 80, 80)

        results = np.squeeze(det_out).transpose()  # (2100, 45)

        polygons = []
        mask_coeffs_list = []
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        for det in results:
            box = det[:self.BOX_COORD_NUM]
            class_scores = det[self.BOX_COORD_NUM:self.BOX_COORD_NUM + self.class_count]
            mask_coeffs = det[-self.mask_dim:]
            class_probs = softmax(class_scores)
            class_id = int(np.argmax(class_probs))
            score = class_probs[class_id]
            if score < self.score_threshold:
                continue
            polygons.append([class_id, float(score)] + list(box))
            mask_coeffs_list.append(mask_coeffs)

        # Trả về polygons, mask_coeffs_list, mask_proto để post-process mask ngoài
        return polygons, mask_coeffs_list, mask_proto