import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.utils.edgetpu import list_edge_tpus
from picamera2 import Picamera2
import time
from multiprocessing import Process, Queue
from PIL import Image, ImageDraw, ImageFont
import minio_uploader

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
    config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
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
        start_time = time.time()

        # Capture frame
        capture_start = time.time()
        frame = picam2.capture_array()
        capture_time = time.time() - capture_start
        print(f"[DEBUG] Frame captured in {capture_time:.3f} seconds")

        # Preprocess frame
        preprocess_start = time.time()
        frame = frame[:height, :width, :3]  # Ensure RGB and correct size
        frame_input = frame.astype(np.float32)
        frame_input = np.expand_dims(frame_input, axis=0)  # Add batch dimension
        preprocess_time = time.time() - preprocess_start
        print(f"[DEBUG] Frame preprocessed in {preprocess_time:.3f} seconds")

        # Run inference
        inference_start = time.time()
        common.set_input(interpreter, frame_input)
        interpreter.invoke()
        output = np.copy(common.output_tensor(interpreter, 0))
        inference_time = time.time() - inference_start
        print(f"[DEBUG] Inference completed in {inference_time:.3f} seconds, output shape: {output.shape}")

        # Process output
        process_start = time.time()
        output = output[0]  # Remove batch dimension: (8, 8400)
        objectness_scores = output[4, :]  # Objectness scores
        class_scores = output[5:8, :]  # Class scores for 3 classes
        box_coords = output[0:4, :]  # Bounding box coordinates: x, y, w, h

        max_idx = np.argmax(objectness_scores)
        max_objectness = objectness_scores[max_idx]
        # print(f"[DEBUG] Highest objectness score: {max_objectness:.4f} at index {max_idx}")

        if max_objectness < OBJECTNESS_THRESHOLD:
            # print("[DEBUG] No detection with sufficient objectness score (>= {:.2f})".format(OBJECTNESS_THRESHOLD))
            continue

        detection_class_scores = class_scores[:, max_idx]
        max_class_idx = np.argmax(detection_class_scores)
        max_class_score = detection_class_scores[max_class_idx]
        # print(f"[DEBUG] Class scores: {detection_class_scores}, Predicted class index: {max_class_idx}")

        if max_class_score < CLASS_SCORE_THRESHOLD:
            # print("[DEBUG] No class with sufficient score (>= {:.2f})".format(CLASS_SCORE_THRESHOLD))
            continue

        predicted_label = labels.get(max_class_idx, "Unknown")
        # print(f"[DEBUG] Predicted class: {predicted_label}")

        confidence = max_objectness * max_class_score
        if confidence < CONFIDENCE_THRESHOLD:
            # print("[DEBUG] Confidence {:.4f} below threshold ({:.2f})".format(confidence, CONFIDENCE_THRESHOLD))
            continue

        print(f"Predicted: {predicted_label} with Confidence: {confidence:.4f}")
        process_time = time.time() - process_start
        print(f"[DEBUG] Output processed in {process_time:.3f} seconds")

        # Draw bounding box
        draw_start = time.time()
        x, y, w, h = box_coords[:, max_idx]
        # print(f"[DEBUG] Bounding box coords: x={x:.2f}, y={y:.2f}, w={w:.2f}, h={h:.2f}")

        try:
            image = Image.fromarray(frame.astype(np.uint8))
            draw = ImageDraw.Draw(image)

            # Assume coordinates are normalized (0-1); scale to image size
            x1 = int(x * width - w * width / 2)
            y1 = int(y * height - h * height / 2)
            x2 = int(x * width + w * width / 2)
            y2 = int(y * height + h * height / 2)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            # print(f"[DEBUG] Scaled bounding box: ({x1}, {y1}, {x2}, {y2})")

            draw.rectangle((x1, y1, x2, y2), outline="red", width=2)

            label_text = f"{predicted_label}: {confidence:.4f}"
            font = ImageFont.load_default()  # Use default font for speed
            draw.text((x1, y1 - 20), label_text, fill="red", font=font)
            # print("[DEBUG] Bounding box and label drawn")

            frame_rgb = np.array(image)
        except Exception as e:
            print("[ERROR] Failed to draw bounding box: {}".format(e))
            continue
        draw_time = time.time() - draw_start
        print(f"[DEBUG] Bounding box drawn in {draw_time:.3f} seconds")

        # Send detection to upload queue
        queue_start = time.time()
        upload_queue.put((frame_rgb, predicted_label, confidence))
        queue_time = time.time() - queue_start
        print(f"[DEBUG] Detection sent to upload queue in {queue_time:.3f} seconds")

        # Total frame time
        total_time = time.time() - start_time
        print(f"[DEBUG] Frame processed in {total_time:.3f} seconds")

except KeyboardInterrupt:
    print("[DEBUG] Stopping script...")
finally:
    picam2.stop()
    picam2.close()
    upload_queue.put(None)
    uploader_process.join()
    print("[DEBUG] Camera stopped, uploader process terminated, and resources released")