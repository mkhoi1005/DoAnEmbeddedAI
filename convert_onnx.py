from ultralytics import YOLO

# Load mô hình YOLOv8 từ file best.pt
model = YOLO('best.pt')

# Export sang ONNX
model.export(format='onnx', imgsz=320, dynamic=False, simplify=True, opset=12)