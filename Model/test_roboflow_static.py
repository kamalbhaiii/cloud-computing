import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# ====== CONFIG ======
API_URL = "https://serverless.roboflow.com"
API_KEY = "uUiAY4ufKNxPGS9jwXP6"
MODEL_ID = "datasets-zjnxi/3"
IMAGE_PATH = "test.jpg"
OUTPUT_PATH = "output_image.jpg"
CONF_THRESHOLD = 0.5

# ====== Initialize Roboflow client ======
CLIENT = InferenceHTTPClient(
    api_url=API_URL,
    api_key=API_KEY
)

# ====== Load image ======
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"[ERROR] Cannot load image at {IMAGE_PATH}")
    exit()

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

# ====== Process image ======
try:
    # Perform inference using Roboflow API
    result = CLIENT.infer(IMAGE_PATH, model_id=MODEL_ID)
    
    # Extract predictions
    predictions = result.get('predictions', [])
    print(f"[INFO] Received {len(predictions)} predictions")

    # Draw detections on image
    output_image = draw_detections(image, predictions)

    # Save the output image
    cv2.imwrite(OUTPUT_PATH, output_image)
    print(f"[INFO] Output image saved to {OUTPUT_PATH}")

    # Display the image
    cv2.imshow("Wildlife Detection", output_image)
    print("[INFO] Press 'q' to exit")
    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

except Exception as e:
    print(f"[ERROR] An error occurred: {e}")

finally:
    cv2.destroyAllWindows()