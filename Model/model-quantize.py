import tensorflow as tf

# Load float model
converter = tf.lite.TFLiteConverter.from_saved_model("best_float32_edgetpu.tflite")  # or from_keras_model()
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset (needed for proper quantization)
def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 640, 640, 3).astype(np.float32)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

quant_model = converter.convert()

with open("quant_model.tflite", "wb") as f:
    f.write(quant_model)
