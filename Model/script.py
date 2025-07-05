import numpy as np
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from threading import Thread
import os
import tflite_runtime.interpreter as tflite
from background_uploader import upload_image_to_db
from InquirerPy import inquirer
from picamera2 import Picamera2

def background_upload(image_path, category):
    upload_image_to_db(image_path, category)

# Directories
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# Load the label map
label_map = {}
with open('labelmap.txt', 'r') as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# Automatically list all .tflite files in the current directory
tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]

if not tflite_files:
    raise FileNotFoundError("No .tflite model files found in the current directory.")

# Prompt the user to select a model interactively
selected_model = inquirer.select(
    message="Select the TFLite model to use:",
    choices=tflite_files,
    default=tflite_files[0]
).execute()

model_path = selected_model
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
print("Model loaded using tflite-runtime.")

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_quant = input_details[0]['quantization']
output_quant = output_details[0]['quantization']
output_dtype = output_details[0]['dtype']
input_h, input_w = input_shape[1], input_shape[2]

print(f"Input dtype: {input_dtype}, quant: {input_quant}")
print(f"Output dtype: {output_dtype}, quant: {output_quant}")

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)  # allow camera to warm up

print("Camera initialized. Starting capture...")

# Process frames from PiCamera2
try:
    while True:
        frame = picam2.capture_array()
        pil_image = Image.fromarray(frame).convert("RGB")
        draw_image = pil_image.copy()
        original_w, original_h = pil_image.size
        resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
        image_np = np.array(resized)

        # Preprocess input
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

        # Postprocess
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        if output_dtype in [np.uint8, np.int8]:
            scale, zero_point = output_quant
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        print("Raw output shape:", output_data.shape)
        predictions = output_data.transpose()
        threshold = 0.3
        best_detection = None
        max_confidence = 0.0
        best_box = None

        for pred in predictions:
            x, y, w, h = pred[:4]
            objectness = pred[4]
            class_id = int(pred[5])
            confidence = pred[6]

            if confidence > threshold and confidence > max_confidence:
                max_confidence = confidence
                best_detection = (class_id, confidence)
                best_box = (x, y, w, h)

        if best_detection and best_box:
            x, y, w, h = best_box
            x_min = int((x - w / 2) * original_w)
            y_min = int((y - h / 2) * original_h)
            x_max = int((x + w / 2) * original_w)
            y_max = int((y + h / 2) * original_h)

            draw = ImageDraw.Draw(draw_image)
            label = label_map.get(best_detection[0], f"class_{best_detection[0]}")
            confidence_text = f"{label} ({best_detection[1]*100:.1f}%)"
            draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=3)
            draw.text((x_min, y_min - 10), confidence_text, fill="red")

            # Save annotated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(TEMP_DIR, f"{timestamp}.jpg")
            draw_image.save(save_path)

            print(f"Detected: {label.capitalize()} | Confidence: {best_detection[1]:.2f}")
            Thread(target=background_upload, args=(save_path, label), daemon=True).start()
        else:
            print("No objects detected.")

        print(f"Inference time: {inference_time:.4f} seconds")
        print("-" * 100)
        print("Next inference will be done after 5 seconds")
        print("-" * 100)

        time.sleep(5.0)

except KeyboardInterrupt:
    print("Interrupted by user.")
    picam2.stop()
