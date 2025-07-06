import os
import time
from datetime import datetime
from threading import Thread
import numpy as np
import cv2
from PIL import Image
from InquirerPy import inquirer
import tflite_runtime.interpreter as tflite
from background_uploader import upload_image_to_db

# === Directories ===
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# === Load label map ===
label_map = {}
with open("labelmap.txt", "r") as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# === Select model ===
tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]
if not tflite_files:
    raise FileNotFoundError("No .tflite models found.")

selected_model = inquirer.select(
    message="Select the TFLite model to use:",
    choices=tflite_files,
    default=tflite_files[0]
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

# === Upload in background ===
def background_upload(image_path, category):
    upload_image_to_db(image_path, category)

print("🔁 Starting real-time detection every 5 seconds. Press 'q' to stop.")
try:
    while True:
        # === Start video capture ===
        cap = cv2.VideoCapture(0)
        time.sleep(0.5)  # Allow camera to initialize

        if not cap.isOpened():
            print("❌ Cannot access the USB camera.")
            cap.release()
            time.sleep(5)
            continue

        ret, frame = cap.read()
        cap.release()
        time.sleep(0.5)  # Allow camera to fully release

        if not ret:
            print("❌ Failed to capture frame.")
            time.sleep(5)
            continue

        # Convert to RGB and resize for inference
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)
        image_np = np.array(resized)

        # === Preprocess input ===
        if input_dtype == np.float32:
            input_tensor = np.expand_dims(image_np.astype(np.float32) / 255.0, axis=0)
        elif input_dtype in [np.uint8, np.int8]:
            scale, zero_point = input_quant
            input_tensor = ((image_np / 255.0) / scale + zero_point).astype(input_dtype)
            input_tensor = np.expand_dims(input_tensor, axis=0)
        else:
            raise ValueError(f"Unsupported input dtype: {input_dtype}")

        # === Inference ===
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time

        # === Postprocess ===
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]
        if output_dtype in [np.uint8, np.int8]:
            scale, zero_point = output_quant
            output_data = (output_data.astype(np.float32) - zero_point) * scale

        predictions = output_data.transpose((1, 0))  # (8400, 8)
        threshold = 0.01
        detections = []

        original_h, original_w = frame.shape[:2]

        for pred in predictions:
            x, y, w, h = pred[:4]
            objectness = pred[4]
            class_probs = pred[5:]
            class_id = np.argmax(class_probs)
            class_score = class_probs[class_id]
            confidence = objectness * class_score

            if confidence > threshold:
                x_min = int((x - w / 2) * original_w)
                y_min = int((y - h / 2) * original_h)
                x_max = int((x + w / 2) * original_w)
                y_max = int((y + h / 2) * original_h)

                label = label_map.get(class_id, f"class_{class_id}")
                text = f"{label} ({confidence*100:.1f}%)"

                # Draw on OpenCV frame
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, text, (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                detections.append(label)

        if detections:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(TEMP_DIR, f"{timestamp}.jpg")
            cv2.imwrite(save_path, frame)

            print(f"🟢 Detected: {detections}")
            for label in detections:
                Thread(target=background_upload, args=(save_path, label), daemon=True).start()
        else:
            print("🔴 No objects detected.")

        print(f"Inference time: {inference_time:.4f} seconds")
        print("-" * 60)

        # === Show frame ===
        cv2.imshow("Live Detection", frame)

        # Wait briefly to show frame (500ms) or break if 'q' is pressed
        if cv2.waitKey(500) & 0xFF == ord('q'):
            print("🛑 Detection stopped by user.")
            break

        cv2.destroyAllWindows()  # Close the detection window
        time.sleep(5)  # Wait before restarting the cycle

except KeyboardInterrupt:
    print("🛑 Detection interrupted by user (Ctrl+C).")

finally:
    cv2.destroyAllWindows()