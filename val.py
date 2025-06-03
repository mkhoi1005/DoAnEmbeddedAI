import os
import cv2
import numpy as np
import argparse
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from mean_average_precision import MetricBuilder

# === Preprocess ===
def preprocess_image(img_path, imgsz):
    img = Image.open(img_path).convert('RGB').resize((imgsz, imgsz))
    img_np = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(img_np, axis=0)  # (1, imgsz, imgsz, 3)

# === Postprocess masks (if needed) ===
def postprocess(boxes, scores, classes, orig_shape, conf_thresh=0.25):
    boxes_out, scores_out, classes_out = [], [], []
    for i in range(len(scores)):
        if scores[i] < conf_thresh:
            continue
        y1, x1, y2, x2 = boxes[i]
        h, w = orig_shape
        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h
        boxes_out.append([x1, y1, x2, y2])
        scores_out.append(scores[i])
        classes_out.append(int(classes[i]))
    return boxes_out, scores_out, classes_out

# === Parse YOLO .txt label ===
def parse_label_yolo(txt_path, img_size):
    boxes, classes = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            cls, cx, cy, w, h = map(float, line.strip().split())
            cx, cy, w, h = cx * img_size[0], cy * img_size[1], w * img_size[0], h * img_size[1]
            x1, y1 = cx - w / 2, cy - h / 2
            x2, y2 = cx + w / 2, cy + h / 2
            boxes.append([x1, y1, x2, y2])
            classes.append(int(cls))
    return boxes, classes

# === Inference ===
def run_inference(interpreter, input_details, output_details, img_input):
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    scores = interpreter.get_tensor(output_details[1]['index'])[0]
    classes = interpreter.get_tensor(output_details[2]['index'])[0]
    return boxes, scores, classes

# === Main validate ===
def validate(args):
    interpreter = tf.lite.Interpreter(model_path=args.weights)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    imgsz = args.imgsz

    metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False)

    val_images = sorted(list(Path(args.val_images).glob("*.jpg")))
    for img_path in tqdm(val_images, desc="Validating"):
        img_np = preprocess_image(str(img_path), imgsz)
        orig = cv2.imread(str(img_path))
        h, w = orig.shape[:2]
        
        boxes, scores, classes = run_inference(interpreter, input_details, output_details, img_np)
        pred_boxes, pred_scores, pred_classes = postprocess(boxes, scores, classes, (h, w), conf_thresh=args.conf)

        # Format prediction for mAP library
        predictions = []
        for b, s, c in zip(pred_boxes, pred_scores, pred_classes):
            predictions.append({'bbox': b, 'score': s, 'class': c})

        # Parse ground-truth
        label_path = Path(args.val_labels) / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        gt_boxes, gt_classes = parse_label_yolo(label_path, (w, h))
        gts = []
        for b, c in zip(gt_boxes, gt_classes):
            gts.append({'bbox': b, 'class': c})

        metric_fn.add(predictions, gts)

    results = metric_fn.value(iou_thresholds=[0.5, 0.75, 0.95])
    print("\n==== Evaluation Result ====")
    print(f"mAP@0.5:      {results['map_0.50']:.3f}")
    print(f"mAP@0.5:0.95: {results['map']:.3f}")
    print(f"Recall:       {results['recall']:.3f}")
    print("===========================\n")

# === Argument ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to .tflite file')
    parser.add_argument('--val_images', type=str, required=True, help='Path to validation images folder')
    parser.add_argument('--val_labels', type=str, required=True, help='Path to labels folder (.txt in YOLO format)')
    parser.add_argument('--imgsz', type=int, default=320, help='Inference image size (default=320)')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()

    validate(args)
