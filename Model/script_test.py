import tflite_runtime.interpreter as tflite

# Load model with Edge TPU delegate
delegate = tflite.load_delegate("libedgetpu.so.1")
interpreter = tflite.Interpreter(
    model_path="best_float32_edgetpu.tflite",  # Replace with your model path
    experimental_delegates=[delegate]
)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)