import os
import time
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw
from threading import Thread
from InquirerPy import inquirer

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from background_uploader import upload_image_to_db

# === Directories ===
INPUT_DIR = "input_images"
TEMP_DIR = "temp"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# === Load Label Map ===
label_map = {}
labelmap_path = "labelmap.txt"
if os.path.exists(labelmap_path):
    with open(labelmap_path, "r") as f:
        for line in f:
            idx, label = line.strip().split()
            label_map[int(idx)] = label
else:
    print("⚠️ No labelmap.txt found. Defaulting to class indices.")

# === Model Selection ===
tflite_models = [f for f in os.listdir(".") if f.endswith("edgetpu.tflite")]
if not tflite_models:
    raise FileNotFoundError("No .tflite models found in the current directory.")

selected_model = inquirer.select(
    message="Select the TFLite model to use:",
    choices=tflite_models
).execute()

print(f"Selected model: {selected_model}")

# === Load model with Edge TPU delegate ===
interpreter = make_interpreter(selected_model)
interpreter.allocate_tensors()
print("Model loaded successfully.")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']

input_h, input_w = input_shape[1], input_shape[2]
print(f"Model input shape: {input_shape}")
print(f"Model input dtype: {input_dtype}")

# === Background uploader ===
def background_upload(image_path, category):
    upload_image_to_db(image_path, category)

# === Detection loop ===
print("\n🔁 Starting detection loop. Press Ctrl+C to stop.")
try:
    while True:
        image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            print("No new images. Waiting...")
            time.sleep(3)
            continue

        for img_name in image_files:
            img_path = os.path.join(INPUT_DIR, img_name)
            print(f"\n🔍 Processing: {img_path}")

            # === Load image ===
            pil_image = Image.open(img_path).convert("RGB")
            original_w, original_h = pil_image.size
            resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
            image_np = np.array(resized)

            # === Preprocess input ===
            input_tensor = np.expand_dims(image_np.astype(np.float32) / 255.0, axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)

            # === Inference ===
            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time
            print(f"Inference time: {inference_time:.3f}s")

            # === Postprocess output ===
            output_tensor = interpreter.get_tensor(output_details[0]['index'])[0].transpose()
            print(f"Output tensor shape: {output_tensor.shape}")

            threshold = 0.3
            best_pred = None
            best_conf = 0

            for pred in output_tensor:
                x, y, w, h = pred[:4]
                class_scores = pred[4:]
                class_id = int(np.argmax(class_scores))
                confidence = class_scores[class_id]

                if confidence > threshold and confidence > best_conf:
                    best_conf = confidence
                    best_pred = {
                        "class_id": class_id,
                        "confidence": confidence,
                        "box": (x, y, w, h)
                    }

            # === Draw & Save ===
            if best_pred:
                x, y, w, h = best_pred["box"]
                x_min = int((x - w / 2) * original_w)
                y_min = int((y - h / 2) * original_h)
                x_max = int((x + w / 2) * original_w)
                y_max = int((y + h / 2) * original_h)

                draw_image = pil_image.copy()
                draw = ImageDraw.Draw(draw_image)

                label = label_map.get(best_pred["class_id"], f"class_{best_pred['class_id']}")
                text = f"{label} ({best_pred['confidence']*100:.1f}%)"

                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
                draw.text((x_min, y_min - 10), text, fill="red")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(TEMP_DIR, f"{timestamp}.jpg")
                draw_image.save(save_path)

                print(f"✅ Detected: {text}")
                print(f"💾 Saved to: {save_path}")

                Thread(target=background_upload, args=(save_path, label), daemon=True).start()
            else:
                print("❌ No detection above threshold.")

            os.remove(img_path)
            print("-" * 50)

except KeyboardInterrupt:
    print("🛑 Detection loop interrupted by user.")
