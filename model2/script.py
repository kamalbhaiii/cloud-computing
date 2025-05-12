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

# Get model input and output details
input_details = interpreter.get_input_details()[0]
input_shape = input_details['shape']
height, width = input_shape[1], input_shape[2]
print(f"[DEBUG] Model expects input shape: {input_shape}")

output_details = interpreter.get_output_details()
print(f"[DEBUG] Model output details: {output_details}")

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
        # Copy output to avoid holding reference to internal tensor
        output = np.copy(common.output_tensor(interpreter, 0))
        print(f"[DEBUG] Inference completed, output shape: {output.shape}")

        # Process object detection output
        # Assuming output shape: (1, 8, 8400)
        # 8 = [x, y, w, h, objectness, class_score_0, class_score_1, class_score_2]
        output = output[0]  # Remove batch dimension: (8, 8400)
        objectness_scores = output[4, :]  # Objectness scores
        class_scores = output[5:8, :]  # Class scores for 3 classes (adjust if 4)

        # Find detection with highest objectness score
        max_idx = np.argmax(objectness_scores)
        max_objectness = objectness_scores[max_idx]
        print(f"[DEBUG] Highest objectness score: {max_objectness:.4f} at index {max_idx}")

        # Get class scores for this detection
        detection_class_scores = class_scores[:, max_idx]
        max_class_idx = np.argmax(detection_class_scores)
        max_class_score = detection_class_scores[max_class_idx]
        predicted_label = labels.get(max_class_idx, "Unknown")
        print(f"[DEBUG] Class scores: {detection_class_scores}, Predicted class: {predicted_label}")

        # Print prediction
        confidence = max_objectness * max_class_score  # Combine objectness and class score
        print(f"Predicted: {predicted_label} with Confidence: {confidence:.4f}")

        # Debug: Frame rate
        elapsed = time.time() - start_time
        print(f"[DEBUG] Frame processed in {elapsed:.3f} seconds")

except KeyboardInterrupt:
    print("[DEBUG] Stopping script...")
finally:
    picam2.stop()
    picam2.close()
    print("[DEBUG] Camera stopped and resources released")