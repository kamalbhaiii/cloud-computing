import numpy as np
import os
import time
from datetime import datetime
from PIL import Image, ImageDraw
from threading import Thread
from InquirerPy import inquirer
import tflite_runtime.interpreter as tflite
from background_uploader import upload_image_to_db

# === Directories ===
INPUT_DIR = "input_images"
TEMP_DIR = "temp"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# === Load Label Map ===
label_map = {}
with open("labelmap.txt", "r") as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# === Model Selection ===
tflite_models = [f for f in os.listdir(".") if f.endswith(".tflite")]
if not tflite_models:
    raise FileNotFoundError("No .tflite models found.")

selected_model = inquirer.select(
    message="Select the TFLite model to use:",
    choices=tflite_models
).execute()

print(f"Selected model: {selected_model}")
interpreter = tflite.Interpreter(model_path=selected_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_quant = input_details[0]['quantization']
output_dtype = output_details[0]['dtype']
output_quant = output_details[0]['quantization']

input_h, input_w = input_shape[1], input_shape[2]

print(f"Model input shape: {input_shape}")
print(f"Model input dtype: {input_dtype}")
print(f"Model input quantization: {input_quant}")

# === Image Processing Loop ===
def background_upload(image_path, category):
    upload_image_to_db(image_path, category)

try:
    while True:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print("No new images. Waiting...")
            time.sleep(3)
            continue

        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)
            print(f"Loading image: {img_path}")
            pil_image = Image.open(img_path).convert("RGB")
            draw_image = pil_image.copy()
            original_w, original_h = pil_image.size

            resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
            image_np = np.array(resized)

            # === Input tensor formatting ===
            if input_dtype == np.float32:
                input_tensor = np.expand_dims(image_np.astype(np.float32) / 255.0, axis=0)
            elif input_dtype in [np.uint8, np.int8]:
                scale, zero_point = input_quant
                input_tensor = ((image_np.astype(np.float32) / 255.0) / scale + zero_point).astype(input_dtype)
                input_tensor = np.expand_dims(input_tensor, axis=0)
            else:
                raise ValueError(f"Unsupported input dtype: {input_dtype}")

            print(f"Input tensor shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
            h, w = input_tensor.shape[1:3]
            center_pixel = input_tensor[0, h // 2, w // 2]
            print(f"Input tensor sample pixel (center): {center_pixel}")

            # === Inference ===
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            start_time = time.time()
            interpreter.invoke()
            elapsed = time.time() - start_time
            print(f"Inference time: {elapsed:.3f}s")

            # === Output ===
            output_tensor = interpreter.get_tensor(output_details[0]['index'])[0]
            print(f"Raw output shape: {output_tensor.shape}")
            print(f"Raw output sample (first 5 elements): {output_tensor.flatten()[:5]}")

            if output_dtype in [np.uint8, np.int8]:
                scale, zero_point = output_quant
                output_tensor = (output_tensor.astype(np.float32) - zero_point) * scale

            predictions = output_tensor.transpose()  # Shape: (8400, 8)

            # === Postprocessing ===
            threshold = 0.3
            best_pred = None
            best_conf = 0

            for pred in predictions:
                x, y, w, h = pred[:4]
                class_scores = pred[4:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]

                if confidence > threshold and confidence > best_conf:
                    best_conf = confidence
                    best_pred = {
                        "class_id": class_id,
                        "confidence": confidence,
                        "box": (x, y, w, h)
                    }

            # === Draw / Upload ===
            if best_pred:
                label = label_map.get(best_pred["class_id"], f"class_{best_pred['class_id']}")
                confidence = best_pred["confidence"]
                x, y, w, h = best_pred["box"]

                x_min = int((x - w / 2) * original_w)
                y_min = int((y - h / 2) * original_h)
                x_max = int((x + w / 2) * original_w)
                y_max = int((y + h / 2) * original_h)

                draw = ImageDraw.Draw(draw_image)
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                draw.text((x_min, y_min - 10), f"{label} ({confidence*100:.1f}%)", fill="red")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(TEMP_DIR, f"{timestamp}.jpg")
                draw_image.save(save_path)

                print(f"Detected: {label} ({confidence:.2f})")
                Thread(target=background_upload, args=(save_path, label), daemon=True).start()
            else:
                print("No objects detected.")

            os.remove(img_path)
            print("-" * 50)

except KeyboardInterrupt:
    print("Detection loop interrupted by user.")
