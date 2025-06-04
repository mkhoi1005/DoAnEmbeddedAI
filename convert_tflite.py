import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model("best_tf")
tflite_model = converter.convert()
with open("best_float32_inst.tflite", "wb") as f:
    f.write(tflite_model)