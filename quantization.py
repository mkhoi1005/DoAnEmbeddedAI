import argparse
from ultralytics import YOLO
import os

def parse_opt():
    parser = argparse.ArgumentParser(description="Quantize YOLOv8 model to TFLite INT8")
    parser.add_argument('--model', type=str, required=True, help='Path to the .pt model (e.g., best.pt or pruned.pt)')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for export (default: 640)')
    parser.add_argument('--output_dir', type=str, default='quantized_model', help='Directory to save quantized model')
    return parser.parse_args()

def main(opt):
    print(f"🚀 Loading model from {opt.model}")
    model = YOLO(opt.model)

    print(f"🔁 Exporting model to TFLite INT8 (with calibration using val set in {opt.data})")
    export_path = model.export(
        format='tflite',
        int8=True,
        imgsz=opt.imgsz,
        data=opt.data
    )

    # Di chuyển file đầu ra đến thư mục mong muốn
    os.makedirs(opt.output_dir, exist_ok=True)
    output_file = os.path.join(opt.output_dir, os.path.basename(export_path))
    os.rename(export_path, output_file)

    print(f"✅ Quantized model saved at: {output_file}")

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
