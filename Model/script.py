import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.utils.edgetpu import list_edge_tpus
from picamera2 import Picamera2
import time
from multiprocessing import Process, Queue
import minio_uploader  # Import the uploader script

# Debug: List TPUs
print("[DEBUG] Listing all TPU(s) connected")
for i in list_edge_tpus():
    print(i)

# Debug: Print script start
print("[DEBUG] Starting prediction script...")

# Load labelmap
labels = {}
try:
    with open("labelmap.txt", "r") as f:
        for line in f:
            idx, label = line.strip().split(" ", 1)
            labels[int(idx)] = label
    print("[DEBUG] Labelmap loaded successfully:", labels)
except Exception as e:
    print("[ERROR] Failed to load labelmap:", e)
    exit(1)

# Initialize Edge TPU model
try:
    interpreter = edgetpu.make_interpreter("best_float32_edgetpu.tflite")
    interpreter.allocate_tensors()
    print("[DEBUG] Edge TPU model loaded and tensors allocated")
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    exit(1)

# Get model input and output details
input_details = interpreter.get_input_details()[0]
input_shape = input_details['shape']
height, width = input_shape[1], input_shape[2]
print(f"[DEBUG] Model expects input shape: {input_shape}")

output_details = interpreter.get_output_details()
print(f"[DEBUG] Model output details: {output_details}")

# Initialize camera
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height)})
    picam2.configure(config)
    picam2.start()
    print("[DEBUG] Camera initialized and started")
except Exception as e:
    print("[ERROR] Failed to initialize camera:", e)
    exit(1)

# Thresholds
CONFIDENCE_THRESHOLD = 0.5  # 50% confidence
OBJECTNESS_THRESHOLD = 0.4  # Minimum objectness score
CLASS_SCORE_THRESHOLD = 0.4  # Minimum class score

# Initialize multiprocessing queue and uploader process
upload_queue = Queue()
uploader_process = Process(target=minio_uploader.upload_to_minio, args=(upload_queue,))
uploader_process.daemon = True
uploader_process.start()
print("[DEBUG] MinIO uploader process started")

# Main loop
try:
    while True:
        # Capture frame
        start_time = time.time()
        frame = picam2.capture_array()
        print("[DEBUG] Frame captured")

        # Preprocess frame
        frame = frame[:height, :width, :3]  # Ensure RGB and correct size
        frame = frame.astype(np.float32)
        frame = np.expand_dims(frame, axis=0)  # Add batch dimension
        print("[DEBUG] Frame preprocessed")

        # Run inference
        common.set_input(interpreter, frame)
        interpreter.invoke()
        # Copy output to avoid holding reference to internal tensor
        output = np.copy(common.output_tensor(interpreter, 0))
        print(f"[DEBUG] Inference completed, output shape: {output.shape}")

        # Process object detection output
        # Assuming output shape: (1, 8, 8400)
        # 8 = [x, y, w, h, objectness, class_score_0, class_score_1, class_score_2, class_score_3]
        output = output[0]  # Remove batch dimension: (8, 8400)
        objectness_scores = output[4, :]  # Objectness scores
        class_scores = output[5:9, :]  # Class scores for 4 classes

        # Find detection with highest objectness score
        max_idx = np.argmax(objectness_scores)
        max_objectness = objectness_scores[max_idx]
        print(f"[DEBUG] Highest objectness score: {max_objectness:.4f} at index {max_idx}")

        # Check objectness threshold
        if max_objectness < OBJECTNESS_THRESHOLD:
            print("[DEBUG] No detection with sufficient objectness score (>= {:.2f})".format(OBJECTNESS_THRESHOLD))
            continue

        # Get class scores for this detection
        detection_class_scores = class_scores[:, max_idx]
        max_class_idx = np.argmax(detection_class_scores)
        max_class_score = detection_class_scores[max_class_idx]
        print(f"[DEBUG] Class scores: {detection_class_scores}, Predicted class index: {max_class_idx}")

        # Check class score threshold
        if max_class_score < CLASS_SCORE_THRESHOLD:
            print("[DEBUG] No class with sufficient score (>= {:.2f})".format(CLASS_SCORE_THRESHOLD))
            continue

        # Get predicted label
        predicted_label = labels.get(max_class_idx, "Unknown")
        print(f"[DEBUG] Predicted class: {predicted_label}")

        # Calculate combined confidence
        confidence = max_objectness * max_class_score
        if confidence < CONFIDENCE_THRESHOLD:
            print("[DEBUG] Confidence {:.4f} below threshold ({:.2f})".format(confidence, CONFIDENCE_THRESHOLD))
            continue

        # Print prediction
        print(f"Predicted: {predicted_label} with Confidence: {confidence:.4f}")

        # Send detection to upload queue
        frame_rgb = picam2.capture_array()[:height, :width, :3]  # Capture fresh frame for upload
        upload_queue.put((frame_rgb, predicted_label, confidence))
        print("[DEBUG] Detection sent to upload queue")

        # Debug: Frame rate
        elapsed = time.time() - start_time
        print(f"[DEBUG] Frame processed in {elapsed:.3f} seconds")

except KeyboardInterrupt:
    print("[DEBUG] Stopping script...")
finally:
    picam2.stop()
    picam2.close()
    upload_queue.put(None)  # Signal uploader to stop
    uploader_process.join()
    print("[DEBUG] Camera stopped, uploader process terminated, and resources released")