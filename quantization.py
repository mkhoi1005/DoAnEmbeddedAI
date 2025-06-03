import argparse
import os
import time
import logging
from ultralytics import YOLO
import yaml

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_opt():
    parser = argparse.ArgumentParser(description="Quantize YOLOv8 model to TFLite INT8")
    parser.add_argument('--model', type=str, required=True, help='Path to the .pt model (e.g., best.pt or pruned.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--imgsz', type=int, default=320, help='Image size for export (default: 320 for Raspberry Pi)')
    parser.add_argument('--output_dir', type=str, default='quantized_model', help='Directory to save quantized model')
    return parser.parse_args()

def validate_yaml(data_path):
    """Kiểm tra file YAML và đảm bảo có tập validation."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset YAML file not found at {data_path}")
    
    with open(data_path, 'r') as f:
        data = yaml.safe_load(f)
    
    if 'val' not in data or not os.path.exists(data['val']):
        raise ValueError(f"Validation set not defined or not found in {data_path}. Ensure 'val' path exists.")
    return data

def evaluate_model(model, data_yaml):
    """Đánh giá mô hình và trả về mAP50(M), mAP50-95(M), Recall."""
    logger.info("Evaluating model before quantization...")
    metrics = model.val(data=data_yaml)
    return metrics.mask.map, metrics.mask.map95, metrics.mask.rec

def main(opt):
    # Kiểm tra file YAML
    logger.info(f"🔍 Validating dataset YAML at {opt.data}")
    validate_yaml(opt.data)

    # Load mô hình
    logger.info(f"🚀 Loading model from {opt.model}")
    if not os.path.exists(opt.model):
        raise FileNotFoundError(f"Model file not found at {opt.model}")
    model = YOLO(opt.model)

    # Đánh giá hiệu suất trước quantization
    initial_map50, initial_map95, initial_recall = evaluate_model(model, opt.data)
    logger.info(f"Initial Performance: mAP50(M)={initial_map50:.3f}, mAP50-95(M)={initial_map95:.3f}, Recall(M)={initial_recall:.3f}")

    # Export sang TFLite với INT8 quantization
    start_time = time.time()
    logger.info(f"🔁 Exporting model to TFLite INT8 (imgsz={opt.imgsz}, calibration using val set in {opt.data})")
    try:
        export_path = model.export(
            format='tflite',
            int8=True,
            imgsz=opt.imgsz,
            data=opt.data
        )
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise e

    # Di chuyển file đầu ra
    os.makedirs(opt.output_dir, exist_ok=True)
    output_file = os.path.join(opt.output_dir, os.path.basename(export_path))
    os.rename(export_path, output_file)

    # Kiểm tra kích thước file
    tflite_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
    export_time = time.time() - start_time
    logger.info(f"✅ Quantized model saved at: {output_file}")
    logger.info(f"📏 TFLite model size: {tflite_size:.2f} MB")
    logger.info(f"⏱️ Export time: {export_time:.2f} seconds")

if __name__ == '__main__':
    try:
        opt = parse_opt()
        main(opt)
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        raise e

