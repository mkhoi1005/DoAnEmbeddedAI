# Import necessary libraries
import ultralytics
ultralytics.checks()
from ultralytics import YOLO   

model= YOLO("yolov8s-seg.pt")  # Load a pretrained YOLOv8 model

def load_model(model_path):
    """
    Load a YOLOv8 model from the specified path.
    
    Args:
        model_path (str): Path to the YOLOv8 model file.
        
    Returns:
        model: Loaded YOLOv8 model.
    """
    return YOLO(model_path)