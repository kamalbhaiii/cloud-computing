import os
import time
from datetime import datetime
from threading import Thread
import numpy as np
import cv2
from PIL import Image
from InquirerPy import inquirer
from background_uploader import upload_image_to_db

from pycoral.utils.dataset import read_label_file
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size
from pycoral.adapters import common

# === Directories ===
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# === Load label map ===
label_map = read_label_file("labelmap.txt")  # Format: index label

# === Select EdgeTPU model ===
tflite_files = [f for f in os.listdir('.') if f.endswith('edgetpu.tflite')]
if not tflite_files:
    raise FileNotFoundError("No .tflite models found.")

selected_model = inquirer.select(
    message="Select the EdgeTPU-compiled TFLite model to use:",
    choices=tflite_files,
    default=tflite_files[0]
).execute()

print(f"Selected model: {selected_model}")
interpreter = make_interpreter(selected_model)
interpreter.allocate_tensors()

input_w, input_h = input_size(interpreter)

# === Upload in background ===
def background_upload(image_path, category):
    upload_image_to_db(image_path, category)

# === Open video stream ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError("Cannot access the USB camera.")

print("🔁 Camera running. Detecting every 5 seconds. Press 'q' to stop.")
last_detection_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            continue

        current_time = time.time()

        # Show live video feed
        cv2.imshow("Live Detection", frame)

        if current_time - last_detection_time >= 5:
            last_detection_time = current_time

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame).resize((input_w, input_h), Image.Resampling.LANCZOS)

            common.set_input(interpreter, pil_image)

            start_time = time.time()
            interpreter.invoke()
            inference_time = time.time() - start_time

            objs = detect.get_objects(interpreter, score_threshold=0.01)

            detections = []
            for obj in objs:
                bbox = obj.bbox
                label = label_map.get(obj.id, f"class_{obj.id}")
                confidence = obj.score
                detections.append(label)

                x_min, y_min, x_max, y_max = bbox.left, bbox.top, bbox.right, bbox.bottom

                scale_x = frame.shape[1] / input_w
                scale_y = frame.shape[0] / input_h

                x_min = int(x_min * scale_x)
                y_min = int(y_min * scale_y)
                x_max = int(x_max * scale_x)
                y_max = int(y_max * scale_y)

                text = f"{label} ({confidence*100:.1f}%)"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                cv2.putText(frame, text, (x_min, max(0, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Detection stopped by user.")
            break

except KeyboardInterrupt:
    print("🛑 Detection interrupted by user (Ctrl+C).")

finally:
    cap.release()
    cv2.destroyAllWindows()
