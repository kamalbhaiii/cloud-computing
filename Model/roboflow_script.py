import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
from picamera2 import Picamera2
import time
import os
from minio import Minio
from minio.error import S3Error
from datetime import datetime

# ====== CONFIG ======
API_URL = "https://serverless.roboflow.com"
API_KEY = "uUiAY4ufKNxPGS9jwXP6"
MODEL_ID = "datasets-zjnxi/3"
CONF_THRESHOLD = 0.7
TEMP_IMAGE_PATH = "temp_frame.jpg"

# ====== MinIO CONFIG ======
MINIO_ENDPOINT = "192.168.137.178:30010"  # e.g., "localhost:9000" or "minio.example.com:9000"
MINIO_ACCESS_KEY = "kapfjPa2Tif1NBH9Khqu"
MINIO_SECRET_KEY = "2QRJlePEJuiLWEz5oVJOSpyiIAiJ6VMzybNVVqXj"
MINIO_BUCKET = "wildlife-detections"  # Bucket name in MinIO
MINIO_SECURE = False  # Set to True if using HTTPS

# ====== Initialize Roboflow client ======
CLIENT = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# ====== Initialize MinIO client ======
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE
)

# Ensure bucket exists
try:
    if not minio_client.bucket_exists(MINIO_BUCKET):
        minio_client.make_bucket(MINIO_BUCKET)
        print(f"[INFO] Created bucket {MINIO_BUCKET}")
    else:
        print(f"[INFO] Bucket {MINIO_BUCKET} already exists")
except S3Error as e:
    print(f"[ERROR] MinIO bucket setup failed: {e}")
    exit()

# ====== Init Raspberry Pi Camera ======
picam2 = Picamera2()
try:
    config = picam2.create_still_configuration(main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
except Exception as e:
    print(f"[ERROR] Cannot initialize camera: {e}")
    exit()

# ====== Save frame temporarily for inference ======
def save_temp_frame(frame):
    cv2.imwrite(TEMP_IMAGE_PATH, frame)
    return TEMP_IMAGE_PATH

# ====== Upload to MinIO ======
def upload_to_minio(image, class_name):
    try:
        # Generate unique filename with timestamp and class name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detection_{class_name}_{timestamp}.jpg"
        
        # Save image temporarily for upload
        temp_upload_path = f"temp_upload_{timestamp}.jpg"
        cv2.imwrite(temp_upload_path, image)
        
        # Upload to MinIO
        minio_client.fput_object(
            MINIO_BUCKET,
            filename,
            temp_upload_path,
            content_type="image/jpeg"
        )
        print(f"[INFO] Uploaded {filename} to MinIO bucket {MINIO_BUCKET}")
        
        # Clean up temp upload file
        os.remove(temp_upload_path)
    except S3Error as e:
        print(f"[ERROR] Failed to upload to MinIO: {e}")

# ====== Draw detections ======
def draw_detections(image, predictions):
    height, width, _ = image.shape
    detections_found = False
    detected_classes = []

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
        detected_classes.append(class_name)

    if not detections_found:
        print("[INFO] No species detected above confidence threshold")
    return image, detections_found, detected_classes

# ====== Main loop ======
try:
    while True:
        # Capture frame from Picamera2
        frame = picam2.capture_array()

        # Convert frame to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Save frame temporarily for API inference
        temp_path = save_temp_frame(frame)

        # Perform inference using Roboflow API
        result = CLIENT.infer(temp_path, model_id=MODEL_ID)

        # Extract predictions
        predictions = result.get('predictions', [])
        print(f"[INFO] Received {len(predictions)} predictions")

        # Draw detections on frame
        output_frame, detections_found, detected_classes = draw_detections(frame, predictions)

        # Upload to MinIO if detections are found
        if detections_found:
            for class_name in detected_classes:
                upload_to_minio(output_frame, class_name)

        # Display the frame
        cv2.imshow("Wildlife Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except KeyboardInterrupt:
    print("[INFO] Exiting...")

except Exception as e:
    print(f"[ERROR] An error occurred: {e}")

finally:
    # Clean up
    picam2.stop()
    cv2.destroyAllWindows()
    if os.path.exists(TEMP_IMAGE_PATH):
        os.remove(TEMP_IMAGE_PATH)