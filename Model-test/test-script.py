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
model_path = 'best_full_integer_quant.tflite'  # <-- Change to your model path
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("Model loaded using tflite-runtime.")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_quant = input_details[0]['quantization']  # (scale, zero_point)
output_quant = output_details[0]['quantization']
output_dtype = output_details[0]['dtype']
input_h, input_w = input_shape[1], input_shape[2]

print(f"Input dtype: {input_dtype}, quant: {input_quant}")
print(f"Output dtype: {output_dtype}, quant: {output_quant}")

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
        rgb = frame[..., [2, 1, 0]]  # BGR to RGB

        # Resize and preprocess
        pil_image = Image.fromarray(rgb)
        resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
        image_np = np.array(resized)

        # Preprocess based on input type
        if input_dtype == np.float32:
            input_tensor = np.expand_dims(image_np.astype(np.float32) / 255.0, axis=0)
        elif input_dtype in [np.uint8, np.int8]:
            scale, zero_point = input_quant
            input_tensor = ((image_np.astype(np.float32) / 255.0) / scale + zero_point).astype(input_dtype)
            input_tensor = np.expand_dims(input_tensor, axis=0)
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype}")

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time

        # Get and dequantize output if needed
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        if output_dtype in [np.uint8, np.int8]:
            scale, zero_point = output_quant
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        print("Raw output shape:", output_data.shape)

        predictions = output_data.transpose()  # (8400, 7)
        threshold = 0.4
        best_detection = None
        max_confidence = 0.0

        for pred in predictions:
            x, y, w, h = pred[:4]
            objectness = pred[4]
            class_id = int(pred[5])
            confidence = pred[6]

            if confidence > threshold and confidence > max_confidence:
                max_confidence = confidence
                best_detection = (class_id, confidence)

        # Save and upload if detection is valid
        if best_detection:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            file_path = os.path.join(TEMP_DIR, f"{timestamp}.jpg")
            pil_image.save(file_path)

            class_id, confidence = best_detection
            label = label_map.get(class_id, f"class_{class_id}")
            print(f"Detected: {label.capitalize()} | Confidence: {confidence:.2f}")
            Thread(target=background_upload, args=(file_path, label), daemon=True).start()
        else:
            print("No objects detected.")

        print(f"Inference time: {inference_time:.4f} seconds")
        print("-" * 50)
        print("Next frame capture after 5 seconds.")
        print("-" * 50)
        time.sleep(5.0)

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    picam2.stop()
    print("Camera closed.")
