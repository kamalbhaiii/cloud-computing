import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import time
from picamera2 import Picamera2
from minio import Minio
from minio.error import S3Error
import uuid
import io

# Load the Edge TPU delegate
try:
    from tflite_runtime.interpreter import load_delegate
except ImportError:
    raise ImportError("Cannot import load_delegate from tflite_runtime.interpreter. Ensure tflite-runtime is installed correctly.")

# Load the label map
label_map = {}
with open('labelmap.txt', 'r') as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# Load the Edge TPU-compatible TFLite model with Edge TPU delegate
model_path = 'best_float32_edgetpu.tflite'  # Replace with your Edge TPU model path
try:
    delegate = load_delegate('libedgetpu.so.1')
    print("Edge TPU delegate loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load Edge TPU delegate: {e}")

interpreter = tflite.Interpreter(model_path=model_path, experimental_delegates=[delegate])
try:
    interpreter.allocate_tensors()
    print("Model allocated on Edge TPU successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to allocate tensors on Edge TPU: {e}")

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Initialize MinIO client
minio_client = Minio(
    "your-minio-endpoint:9000",  # Replace with your MinIO endpoint
    access_key="your-access-key",  # Replace with your MinIO access key
    secret_key="your-secret-key",  # Replace with your MinIO secret key
    secure=False  # Set to True if using HTTPS
)
bucket_name = "detections"  # Replace with your bucket name

# Ensure bucket exists
try:
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
        print(f"Bucket {bucket_name} created.")
    else:
        print(f"Bucket {bucket_name} already exists.")
except S3Error as e:
    print(f"MinIO connection failed: {e}. Falling back to display mode.")

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 640)})  # Set resolution to 640x640
picam2.configure(config)
picam2.start()
print("Pi Camera initialized.")

# Non-maximum suppression function
def nms(boxes, scores, iou_threshold=0.5):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        iou = w * h / (areas[i] + areas[order[1:]] - w * h)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# Main loop for real-time detection
cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Preprocess frame
        input_data = np.expand_dims(frame_rgb, axis=0).astype(np.float32)
        input_data /= 255.0  # Normalize to [0, 1]

        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time

        # Get output tensor [1, 8, 8400]
        output = interpreter.get_tensor(output_details[0]['index'])[0]  # [8, 8400]
        output = output.T  # Transpose to [8400, 8]

        # Extract boxes and class probabilities
        num_classes = len(label_map)
        boxes = output[:, :4]  # [8400, 4]
        class_probs = output[:, 4:4+num_classes]  # [8400, num_classes]
        scores = np.max(class_probs, axis=1)  # [8400]
        classes = np.argmax(class_probs, axis=1).astype(int)  # [8400]

        # Filter detections
        valid_indices = np.where(scores > 0.1)[0]  # Confidence threshold
        predicted_class = 'unknown'
        display_frame = frame_rgb.copy()

        if len(valid_indices) == 0:
            valid_indices = [np.argmax(scores)]  # Use highest-scoring detection
        else:
            valid_boxes = boxes[valid_indices]
            valid_scores = scores[valid_indices]
            nms_indices = nms(valid_boxes, valid_scores)
            valid_indices = valid_indices[nms_indices]

        # Process detections
        img_height, img_width = 640, 640
        for idx in valid_indices:
            box = boxes[idx]
            score = scores[idx]
            class_id = classes[idx]
            # Test [x_min, y_min, x_max, y_max] format
            x_min, y_min, x_max, y_max = box
            x1 = int(x_min * img_width)
            y1 = int(y_min * img_height)
            x2 = int(x_max * img_width)
            y2 = int(y_max * img_height)
            # Alternative: Convert from [x_center, y_center, width, height]
            x_center, y_center, width, height = box
            x1_center = int((x_center - width / 2) * img_width)
            y1_center = int((y_center - height / 2) * img_height)
            x2_center = int((x_center + width / 2) * img_width)
            y2_center = int((y_center + height / 2) * img_height)
            # Favor larger area
            area_xy = (x2 - x1) * (y2 - y1)
            area_center = (x2_center - x1_center) * (y2_center - y1_center)
            if area_center > area_xy:
                x1, y1, x2, y2 = x1_center, y1_center, x2_center, y2_center
            label = label_map[class_id]
            if predicted_class == 'unknown':
                predicted_class = label

            # Draw bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f'{label} {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # If detections exist, attempt to upload to MinIO
        if predicted_class != 'unknown':
            try:
                # Convert frame to JPEG
                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
                buffer_io = io.BytesIO(buffer)
                filename = f"pred_{predicted_class}_{uuid.uuid4()}.jpg"
                # Upload to MinIO
                minio_client.put_object(
                    bucket_name,
                    filename,
                    buffer_io,
                    length=len(buffer),
                    content_type='image/jpeg'
                )
                print(f"Uploaded {filename} to MinIO (Class: {predicted_class}, Inference: {inference_time:.4f}s)")
            except S3Error as e:
                print(f"MinIO upload failed: {e}. Displaying detections.")
                cv2.imshow("Detections", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        else:
            # No detections, show frame
            cv2.imshow("Detections", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting gracefully.")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera and windows closed.")