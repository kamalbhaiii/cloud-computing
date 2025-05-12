import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from picamera2 import Picamera2
import time

# Debug: Print script start
print("[DEBUG] Starting prediction script...")

# Load labelmap
labels = {}
try:
    with open("labelmap.txt", "r") as f:
        for line in f:
            idx, label = line.strip().split(" ", 1)
            labels[int(idx)] = label
    print("[DEBUG] Labelmap loaded successfully:", labels)
except Exception as e:
    print("[ERROR] Failed to load labelmap:", e)
    exit(1)

# Initialize Edge TPU model
try:
    interpreter = edgetpu.make_interpreter("best_float32_edgetpu.tflite")
    interpreter.allocate_tensors()
    print("[DEBUG] Edge TPU model loaded and tensors allocated")
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    exit(1)

# Get model input details
input_details = interpreter.get_input_details()[0]
input_shape = input_details['shape']
height, width = input_shape[1], input_shape[2]
print(f"[DEBUG] Model expects input shape: {input_shape}")

# Initialize camera
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()
    print("[DEBUG] Camera initialized and started")
except Exception as e:
    print("[ERROR] Failed to initialize camera:", e)
    exit(1)

# Main loop
try:
    while True:
        # Capture frame
        start_time = time.time()
        frame = picam2.capture_array()
        print("[DEBUG] Frame captured")

        # Preprocess frame
        frame = frame[:height, :width, :3]  # Ensure RGB and correct size
        frame = frame.astype(np.float32)
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        print("[DEBUG] Frame preprocessed")

        # Run inference
        common.set_input(interpreter, frame)
        interpreter.invoke()
        output = common.output_tensor(interpreter, 0)
        print("[DEBUG] Inference completed")

        # Process output
        scores = output[0]
        max_score_idx = np.argmax(scores)
        max_score = scores[max_score_idx]
        predicted_label = labels.get(max_score_idx, "Unknown")
        print(f"Predicted: {predicted_label} with Confidence: {max_score:.4f}")

        # Debug: Frame rate
        elapsed = time.time() - start_time
        print(f"[DEBUG] Frame processed in {elapsed:.3f} seconds")

except KeyboardInterrupt:
    print("[DEBUG] Stopping script...")
finally:
    picam2.stop()
    picam2.close()
    print("[DEBUG] Camera stopped and resources released")