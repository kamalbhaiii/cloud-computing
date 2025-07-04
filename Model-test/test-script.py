import numpy as np
import time
from datetime import datetime
from picamera2 import Picamera2
from PIL import Image
from threading import Thread
import os
import tflite_runtime.interpreter as tflite
from background_uploader import upload_image_to_db

def background_upload(image_path, category):
    print(image_path)
    print(category)
    upload_image_to_db(image_path, category)

# Create temp detection image folder
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

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

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        rgb = frame[..., [2, 1, 0]]  # Convert BGR to RGB

        # Resize and preprocess
        pil_image = Image.fromarray(rgb)
        resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
        input_tensor = np.expand_dims(np.array(resized, dtype=np.float32) / 255.0, axis=0)  # Normalize

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time

        # Postprocessing for YOLOv8-style TFLite output
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape: (7, 8400)
        print("Raw output shape:", output_data.shape)

        predictions = output_data.transpose()  # Shape becomes (8400, 7)
        threshold = 0.4  
        detections = []

        for pred in predictions:
            x, y, w, h = pred[:4]
            objectness = pred[4]
            class_id = int(pred[5])
            confidence = pred[6]

            if confidence > threshold:
                detections.append((class_id, confidence))

        # Save image only once if at least one object is detected
        if detections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = os.path.join(TEMP_DIR, f"{timestamp}.jpg")
            pil_image.save(file_path)

            for class_id, confidence in detections:
                label = label_map.get(class_id, f"class_{class_id}")
                print(f"Detected: {label.capitalize()} | Confidence: {confidence:.2f}")
                Thread(target=background_upload, args=(file_path, label), daemon=True).start()
        else:
            print("No objects detected.")

        print(f"Inference time: {inference_time:.4f} seconds")
        print("-" * 50)
        time.sleep(2.0)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    picam2.stop()
    print("Camera closed.")
