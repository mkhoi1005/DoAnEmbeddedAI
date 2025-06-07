import sys
import types
import numpy as np
import pytest
from sample_evaluation_instance.model import Model

# File: tests/test_model.py

# Patch cv2 and tensorflow.lite
cv2_mock = types.SimpleNamespace()
cv2_mock.resize = lambda img, size: np.zeros((size[1], size[0], 3), dtype=np.uint8)
cv2_mock.cvtColor = lambda img, code: img
cv2_mock.COLOR_BGR2RGB = 0

sys.modules['cv2'] = cv2_mock

tf_mock = types.SimpleNamespace()
tflite_mock = types.SimpleNamespace()
class DummyInterpreter:
    def __init__(self, model_path=None):
        self.input = np.zeros((1, 320, 320, 3), dtype=np.float32)
        self.output = None
    def allocate_tensors(self): pass
    def get_input_details(self):
        return [{"shape": [1, 320, 320, 3], "index": 0}]
    def get_output_details(self):
        # output_details[0]: mask, output_details[1]: detection
        return [{"index": 0, "shape": [1, 320, 320, 1]}, {"index": 1, "shape": [1, 42, 2]}]
    def set_tensor(self, idx, val): self.input = val
    def invoke(self): pass
    def get_tensor(self, idx):
        return self.output

tflite_mock.Interpreter = DummyInterpreter
tf_mock.lite = tflite_mock
sys.modules['tensorflow'] = tf_mock
sys.modules['tensorflow.lite'] = tflite_mock

# Absolute import for Model

@pytest.fixture
def model(monkeypatch):
    # Patch interpreter output for each test
    m = Model(model_path="dummy.tflite")
    return m

def test_predict_valid_detection(model, monkeypatch):
    # Simulate 2 detections: one valid, one with out-of-bounds polygon
    # [class_prob, score, x, y, w, h, ...polygon...]
    det1 = np.array(
        [0.9, 0.8, 0.5, 0.5, 0.2, 0.2] + [0.1, 0.1, 0.2, 0.2, 0.3, 0.3] + [0.0]*30,
        dtype=np.float32
    )
    det2 = np.array(
        [0.6, 0.7, 0.6, 0.6, 0.1, 0.1] + [1.2, 1.2, 0.5, 0.5] + [0.0]*32,
        dtype=np.float32
    )
    # Shape: (N, feature_dim)
    output = np.stack([det1, det2], axis=0)
    # Model expects (1, N, feature_dim)
    model.transpose_output = False
    model.input_width = 320
    model.input_height = 320
    model.num_classes = 1
    model.num_bbox_coords = 4
    model.max_polygon_coords = 36
    model.score_threshold = 0.3

    def fake_get_tensor(idx):
        return output[None, :, :]  # (1, N, feature_dim)
    model.interpreter.get_tensor = fake_get_tensor

    results = model.predict(np.zeros((320, 320, 3), dtype=np.uint8))
    # Only det1 should be valid, det2 polygon stops at out-of-bounds
    assert len(results) == 2
    # Check class_index, score, bbox, polygon
    r = results[0]
    assert r[0] == 0  # class_index
    assert abs(r[1] - 0.8) < 1e-5
    # bbox: x=0.5, y=0.5, w=0.2, h=0.2, input=320
    # x1 = (0.5-0.1)*320 = 0.4*320 = 128
    # y1 = (0.5-0.1)*320 = 128
    # x2 = (0.5+0.1)*320 = 0.6*320 = 192
    # y2 = (0.5+0.1)*320 = 192
    assert abs(r[2] - 128) < 1
    assert abs(r[3] - 128) < 1
    assert abs(r[4] - 192) < 1
    assert abs(r[5] - 192) < 1
    # Polygon: [0.1,0.1,0.2,0.2,0.3,0.3] * 320
    assert r[6:12] == pytest.approx([32,32,64,64,96,96], abs=1)
    # det2 polygon should only include [0.5,0.5] (since 1.2 is out of bounds)
    r2 = results[1]
    assert r2[6:8] == pytest.approx([160,160], abs=1)
    assert len(r2) == 8  # class, score, bbox, 1 polygon point

def test_predict_score_below_threshold(model, monkeypatch):
    det = np.array([0.9, 0.1, 0.5, 0.5, 0.2, 0.2] + [0.1, 0.1] + [0.0]*34, dtype=np.float32)
    output = det[None, None, :]
    model.interpreter.get_tensor = lambda idx: output
    model.transpose_output = False
    model.input_width = 320
    model.input_height = 320
    model.num_classes = 1
    model.num_bbox_coords = 4
    model.max_polygon_coords = 36
    model.score_threshold = 0.3
    results = model.predict(np.zeros((320, 320, 3), dtype=np.uint8))
    assert results == []

def test_predict_empty_output(model, monkeypatch):
    output = np.zeros((1, 0, 42), dtype=np.float32)
    model.interpreter.get_tensor = lambda idx: output
    model.transpose_output = False
    results = model.predict(np.zeros((320, 320, 3), dtype=np.uint8))
    assert results == []