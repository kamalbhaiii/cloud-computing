import cv2
import numpy as np
import tensorflow as tf
import time

# ====== CONFIG ======
MODEL_PATH = "best_float32-2_edgetpu.tflite"  # Standard TFLite model
LABEL_PATH = "labelmap.txt"
INPUT_SIZE = 320  # Roboflow-provided input size
CONF_THRESHOLD = 0.5

# ====== Load labels ======
with open(LABEL_PATH, "r") as f:
    labels = [line.strip().split()[-1] for line in f.readlines()]  # Extract label name (e.g., 'fish' from '0 fish')
print(f"[INFO] Loaded labels: {labels}")

# ====== Load TFLite model ======
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

print(f"[INFO] Model expects input size: {input_width}x{input_height}")

# ====== Init webcam ======
cap = cv2.VideoCapture(0)  # Use default webcam (laptop's built-in camera)
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1)  # Allow camera to warm up

# ====== Preprocessing ======
def preprocess(image):
    image_resized = cv2.resize(image, (input_width, input_height))
    image_normalized = image_resized.astype(np.float32) / 255.0
    return np.expand_dims(image_normalized, axis=0)

# ====== Postprocessing (YOLOv8-specific output) ======
def draw_detections(image, output_data):
    height, width, _ = image.shape
    detections_found = False

    for detection in output_data:
        if detection[4] < CONF_THRESHOLD:
            continue
        class_id = int(detection[5])
        score = float(detection[4])
        x_center, y_center, w, h = detection[0:4]

        # Scale coords to original image
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        # Draw bounding box and label for the predicted species
        label = f"{labels[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"[INFO] Detected {labels[class_id]} with confidence {score:.2f}")
        detections_found = True

    if not detections_found:
        print("[INFO] No species detected above confidence threshold")

# ====== Main loop ======
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        input_tensor = preprocess(frame)

        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])[0]

        # YOLOv8 output format: [x, y, w, h, confidence, class_id]
        draw_detections(frame, output_data)

        cv2.imshow("Wildlife Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Exiting...")

except Exception as e:
    print(f"[ERROR] An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()