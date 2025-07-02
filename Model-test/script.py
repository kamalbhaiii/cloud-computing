
import numpy as np
import time
import csv
from datetime import datetime
from picamera2 import Picamera2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.detect import get_objects
from PIL import Image
import os

# Load the label map
label_map = {}
with open('labelmap.txt', 'r') as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# Load the Edge TPU model using pycoral
model_path = 'best_float32_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
print("Model loaded and allocated using pycoral.")

# Get input details
input_w, input_h = input_size(interpreter)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (input_w, input_h)})
picam2.configure(config)
picam2.start()
print("Pi Camera initialized.")

# CSV file for automatic saving
csv_filename = "detections.csv"

# Write CSV header if file does not exist
write_header = not os.path.isfile(csv_filename)
with open(csv_filename, mode='a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    if write_header:
        writer.writerow(["timestamp", "label", "score", "inference_time"])

try:
    while True:
        # Capture frame as numpy array
        frame = picam2.capture_array()
        # Convert BGR to RGB using NumPy
        rgb = frame[..., [2, 1, 0]]  # Swap BGR to RGB by reordering channels

        # Resize using PIL
        pil_image = Image.fromarray(rgb)
        resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
        resized_array = np.array(resized)

        # Prepare input for the model
        set_input(interpreter, resized_array)

        # Run inference
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time

        # Get detected objects
        objs = get_objects(interpreter, score_threshold=0.1)

        # Automatic push: append detections to CSV
        with open(csv_filename, mode='a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            timestamp = datetime.now().isoformat()
            if objs:
                for obj in objs:
                    label = label_map.get(obj.id, obj.id)
                    score = obj.score
                    # Write detection to CSV
                    writer.writerow([timestamp, label, f"{score:.2f}", f"{inference_time:.4f}"])
            else:
                writer.writerow([timestamp, "None", "0.00", f"{inference_time:.4f}"])

        # (Optional) Still print for debug
        if objs:
            for obj in objs:
                label = label_map.get(obj.id, obj.id)
                score = obj.score
                print(f"Predicted: {label.capitalize()} with Confidence: {score:.2f}")
        else:
            print("No objects detected.")

        print(f"Inference time: {inference_time:.4f} seconds")
        print("-" * 50)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting gracefully.")
finally:
    picam2.stop()
    print("Camera closed.")
