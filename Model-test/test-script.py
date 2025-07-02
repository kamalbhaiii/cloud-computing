import numpy as np
import time
import csv
from datetime import datetime
from picamera2 import Picamera2
from PIL import Image
import os
import tensorflow.lite as tflite

# Load the label map
label_map = {}
with open('labelmap.txt', 'r') as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# Load the TFLite model
model_path = 'best_float32.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("Model loaded using tflite-runtime.")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_h, input_w = input_shape[1], input_shape[2]

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (input_w, input_h)})
picam2.configure(config)
picam2.start()
print("Pi Camera initialized.")

# CSV file for automatic saving
csv_filename = "detections.csv"
write_header = not os.path.isfile(csv_filename)

with open(csv_filename, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["timestamp", "label", "score", "inference_time"])

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        rgb = frame[..., [2, 1, 0]]  # Convert BGR to RGB

        # Resize and preprocess
        pil_image = Image.fromarray(rgb)
        resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
        input_tensor = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)  # Normalize to 0-1

        # Set input and run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time

        # Get output tensors (assuming typical YOLO-like output structure)
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # Simple postprocessing (this depends on your model output format!)
        detections = []
        threshold = 0.1  # Confidence threshold
        for obj in output_data:
            if obj[4] > threshold:  # obj[4] is confidence
                class_id = int(np.argmax(obj[5:]))  # class probabilities start at index 5
                score = obj[4]
                detections.append((class_id, score))

        # Save to CSV
        timestamp = datetime.now().isoformat()
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if detections:
                for class_id, score in detections:
                    label = label_map.get(class_id, str(class_id))
                    writer.writerow([timestamp, label, f"{score:.2f}", f"{inference_time:.4f}"])
            else:
                writer.writerow([timestamp, "None", "0.00", f"{inference_time:.4f}"])

        # Debug print
        if detections:
            for class_id, score in detections:
                label = label_map.get(class_id, str(class_id))
                print(f"Detected: {label.capitalize()} | Confidence: {score:.2f}")
        else:
            print("No objects detected.")

        print(f"Inference time: {inference_time:.4f} seconds")
        print("-" * 50)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    picam2.stop()
    print("Camera closed.")
