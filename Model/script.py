import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.utils.edgetpu import list_edge_tpus
from picamera2 import Picamera2
import time
from multiprocessing import Process, Queue
import minio_uploader

# Debug: List connected TPUs
print("[DEBUG] Listing connected Edge TPUs")
for tpu in list_edge_tpus():
    print(tpu)

# Debug: Script start
print("[DEBUG] Starting prediction script...")

# Load labelmap
labels = {}
try:
    with open("labelmap.txt", "r") as f:
        for line in f:
            idx, label = line.strip().split(" ", 1)
            labels[int(idx)] = label
    print("[DEBUG] Labelmap loaded:", labels)
except Exception as e:
    print("[ERROR] Failed to load labelmap:", e)
    exit(1)

# Initialize Edge TPU model
try:
    interpreter = edgetpu.make_interpreter("best_int8_edgetpu.tflite")
    interpreter.allocate_tensors()
    print("[DEBUG] Model loaded and tensors allocated")
except Exception as e:
    print("[ERROR] Failed to load model:", e)
    exit(1)

# Get input details
input_details = interpreter.get_input_details()[0]
input_shape = input_details['shape']
height, width = input_shape[1], input_shape[2]
input_scale = input_details['quantization'][0]
input_zero_point = input_details['quantization'][1]
print(f"[DEBUG] Input shape: {input_shape}, scale: {input_scale}, zero_point: {input_zero_point}")

# Initialize camera
try:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
    print("[DEBUG] Camera initialized")
except Exception as e:
    print("[ERROR] Failed to initialize camera:", e)
    exit(1)

# Threshold
CONFIDENCE_THRESHOLD = 0.5  # Configurable, 50%

# Start MinIO uploader process
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
        print(f"[DEBUG] Frame captured in {capture_time:.3f}s")

        # Preprocess frame
        preprocess_start = time.time()
        frame_input = (frame.astype(np.float32) / input_scale + input_zero_point).astype(np.uint8)
        frame_input = np.expand_dims(frame_input, axis=0)
        preprocess_time = time.time() - preprocess_start
        print(f"[DEBUG] Frame preprocessed in {preprocess_time:.3f}s")

        # Run inference
        inference_start = time.time()
        common.set_input(interpreter, frame_input)
        interpreter.invoke()
        output = common.output_tensor(interpreter, 0)
        inference_time = time.time() - inference_start
        print(f"[DEBUG] Inference completed in {inference_time:.3f}s")

        # Process output
        process_start = time.time()
        class_scores = output[0]  # Assuming classification output
        max_idx = np.argmax(class_scores)
        confidence = class_scores[max_idx]
        process_time = time.time() - process_start
        print(f"[DEBUG] Output processed in {process_time:.3f}s")

        # Check threshold and output
        if confidence >= CONFIDENCE_THRESHOLD:
            label = labels.get(max_idx, "Unknown")
            print(f"Predicted: {label} with Confidence: {confidence:.4f}")
            upload_queue.put((frame, label, confidence))
            print("[DEBUG] Detection queued for upload")
        else:
            print(f"[DEBUG] Confidence {confidence:.4f} below threshold {CONFIDENCE_THRESHOLD}")

        total_time = time.time() - start_time
        print(f"[DEBUG] Frame processed in {total_time:.3f}s\n")

except KeyboardInterrupt:
    print("[DEBUG] Stopping script...")
finally:
    picam2.stop()
    picam2.close()
    upload_queue.put(None)
    uploader_process.join()
    print("[DEBUG] Cleanup complete")