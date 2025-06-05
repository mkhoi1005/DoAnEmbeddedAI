import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="best_float32.tflite")
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()
input_details = interpreter.get_input_details()

print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# Tính số lớp từ output shape (YOLOv8-seg: 4 box + N class + 32 mask)
output_dim = output_details[0]['shape'][1]
mask_dim = 32  # Thường là 32 với YOLOv8-seg
class_count = output_dim - 4 - mask_dim
print("Number of classes in TFLite model:", class_count)