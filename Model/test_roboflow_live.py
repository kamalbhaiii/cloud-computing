import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import time

# ====== CONFIG ======
API_URL = "https://serverless.roboflow.com"
API_KEY = "uUiAY4ufKNxPGS9jwXP6"
MODEL_ID = "datasets-zjnxi/3"
CONF_THRESHOLD = 0.4
TEMP_IMAGE_PATH = "temp_frame.jpg"

# ====== Initialize Roboflow client ======
CLIENT = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# ====== Init webcam ======
cap = cv2.VideoCapture(0)  # Use default webcam
if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1)  # Allow camera to warm up

# ====== Save frame temporarily for inference ======
def save_temp_frame(frame):
    cv2.imwrite(TEMP_IMAGE_PATH, frame)
    return TEMP_IMAGE_PATH

# ====== Draw detections ======
def draw_detections(image, predictions):
    height, width, _ = image.shape
    detections_found = False

    for prediction in predictions:
        if prediction['confidence'] < CONF_THRESHOLD:
            continue
            
        # Get detection details
        x = prediction['x']
        y = prediction['y']
        w = prediction['width']
        h = prediction['height']
        score = prediction['confidence']
        class_name = prediction['class']

        # Calculate bounding box coordinates
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw bounding box and label
        label = f"{class_name}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"[INFO] Detected {class_name} with confidence {score:.2f}")
        detections_found = True

    if not detections_found:
        print("[INFO] No species detected above confidence threshold")
    return image

# ====== Main loop ======
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break

        # Save frame temporarily for API inference
        temp_path = save_temp_frame(frame)

        # Perform inference using Roboflow API
        result = CLIENT.infer(temp_path, model_id=MODEL_ID)
        
        # Extract predictions
        predictions = result.get('predictions', [])
        print(f"[INFO] Received {len(predictions)} predictions")

        # Draw detections on frame
        output_frame = draw_detections(frame, predictions)

        # Display the frame
        cv2.imshow("Wildlife Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Exiting...")

except Exception as e:
    print(f"[ERROR] An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()