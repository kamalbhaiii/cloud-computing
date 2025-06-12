import numpy as np
import logging
import sys
import time
import os
from multiprocessing import Process, Queue
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.utils.edgetpu import list_edge_tpus
from picamera2 import Picamera2
import requests
from io import BytesIO

def setup_logging(mode):
    """Configure logging based on mode (Normal/Debug)."""
    log_level = logging.DEBUG if mode == "debug" else logging.INFO
    
    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(log_level)
    file_handler = logging.FileHandler("wildlife_detection.log")
    file_handler.setLevel(log_level)
    
    # Define format
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    
    root_logger.info(f"Starting script in {mode} mode")
    return root_logger

def load_labels(label_file):
    """Load labelmap from file."""
    labels = {}
    try:
        with open(label_file, "r") as f:
            for line in f:
                idx, label = line.strip().split(" ", 1)
                labels[int(idx)] = label
        logging.debug(f"Labelmap loaded: {labels}")
        return labels
    except Exception as e:
        logging.error(f"Failed to load labelmap: {e}")
        raise

def initialize_tpu_model(model_file):
    """Initialize Edge TPU model and return interpreter and input details."""
    try:
        interpreter = edgetpu.make_interpreter(model_file)
        interpreter.allocate_tensors()
        logging.debug("Model loaded and tensors allocated")
        input_details = interpreter.get_input_details()[0]
        input_shape = input_details["shape"]
        height, width = input_shape[1], input_shape[2]
        input_scale = input_details["quantization"][0]
        input_zero_point = input_details["quantization"][1]
        logging.debug(f"Input shape: {input_shape}, scale: {input_scale}, zero_point: {input_zero_point}")
        return interpreter, width, height, input_scale, input_zero_point
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise

def initialize_camera(width, height):
    """Initialize Picamera2 with specified dimensions."""
    try:
        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (width, height), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        logging.debug("Camera initialized")
        return picam2
    except Exception as e:
        logging.error(f"Failed to initialize camera: {e}")
        raise

def preprocess_frame(frame, input_scale, input_zero_point):
    """Preprocess frame for model input."""
    preprocess_start = time.time()
    frame_input = (frame.astype(np.float32) / input_scale + input_zero_point).astype(np.uint8)
    frame_input = np.expand_dims(frame_input, axis=0)
    preprocess_time = time.time() - preprocess_start
    logging.debug(f"Frame preprocessed in {preprocess_time:.3f}s")
    return frame_input

def soft_nms(boxes, scores, classes, metas=None, iou_thresh=0.5, sigma=0.5, conf_thresh=0.001):
    """
    Soft-NMS with optional meta-information (e.g., tracking ID).
    """
    boxes = np.array(boxes)
    scores = np.array(scores)
    classes = np.array(classes)
    metas = np.array(metas) if metas is not None else np.full(len(scores), -1)
    
    keep = []
    while len(scores) > 0:
        max_idx = np.argmax(scores)
        max_box = boxes[max_idx]
        max_score = scores[max_idx]
        max_class = classes[max_idx]
        max_meta = metas[max_idx]
        keep.append((max_box, max_score, max_class, max_meta))
        boxes = np.delete(boxes, max_idx, axis=0)
        scores = np.delete(scores, max_idx)
        classes = np.delete(classes, max_idx)
        metas = np.delete(metas, max_idx)
        if len(scores) == 0:
            break
        x1 = np.maximum(max_box[0], boxes[:, 0])
        y1 = np.maximum(max_box[1], boxes[:, 1])
        x2 = np.minimum(max_box[2], boxes[:, 2])
        y2 = np.minimum(max_box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_max = (max_box[2] - max_box[0]) * (max_box[3] - max_box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area_max + area_boxes - inter
        IoU = inter / union
        for j, iou in enumerate(IoU):
            if iou > iou_thresh:
                scores[j] *= np.exp(-(iou ** 2) / sigma)
        mask = scores > conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]
        classes = classes[mask]
        metas = metas[mask]
    return keep

def run_inference(interpreter, frame_input):
    """Run inference on the Edge TPU."""
    inference_start = time.time()
    common.set_input(interpreter, frame_input)
    interpreter.invoke()
    output = np.copy(common.output_tensor(interpreter, 0))
    inference_time = time.time() - inference_start
    logging.debug(f"Inference completed in {inference_time:.3f}s, output shape: {output.shape}")
    return output

def process_output(interpreter, labels, thresholds, input_width, input_height):
    """Process the 4 tensor outputs of the Edge TPU model with Soft-NMS."""
    try:
        boxes = interpreter.get_tensor(interpreter.get_output_details()[0]["index"])[0]
        scores = interpreter.get_tensor(interpreter.get_output_details()[1]["index"])[0]
        class_ids = interpreter.get_tensor(interpreter.get_output_details()[2]["index"])[0]
        meta = interpreter.get_tensor(interpreter.get_output_details()[3]["index"])[0]
        conf_thresh = thresholds["confidence"]
        valid = scores >= conf_thresh
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        meta = meta[valid]
        pixel_boxes = []
        for box in boxes:
            cx, cy, w, h = box
            x_min = max(cx - w / 2, 0) * input_width
            y_min = max(cy - h / 2, 0) * input_height
            x_max = min(cx + w / 2, 1) * input_width
            y_max = min(cy + h / 2, 1) * input_height
            pixel_boxes.append([x_min, y_min, x_max, y_max])
        keep = soft_nms(pixel_boxes, scores, class_ids, metas=meta, iou_thresh=0.5, sigma=0.5, conf_thresh=conf_thresh)
        detections = []
        for box, score, class_id, meta_val in keep:
            label = labels.get(int(class_id), f"Class {int(class_id)}")
            detections.append({
                "bbox": [int(x) for x in box],
                "score": float(score),
                "class": int(class_id),
                "label": label,
                "meta": int(meta_val)
            })
        return detections
    except Exception as e:
        logging.error(f"Error processing output: {e}")
        return []

def upload_to_endpoint(queue):
    """Background process to send image and category to the endpoint."""
    endpoint = "http://192.168.137.178:30070/api/images"
    while True:
        item = queue.get()
        if item is None:
            logging.info("Upload process received termination signal")
            break
        frame, label = item
        try:
            # Convert frame to JPEG
            from PIL import Image
            img = Image.fromarray(frame)
            img_buffer = BytesIO()
            img.save(img_buffer, format="JPEG")
            img_buffer.seek(0)
            # Prepare multipart/form-data
            files = {
                "image": ("image.jpg", img_buffer, "image/jpeg"),
                "category": (None, label)
            }
            # Send POST request
            response = requests.post(endpoint, files=files)
            if response.status_code == 200:
                logging.info(f"Successfully uploaded {label} to endpoint")
            else:
                logging.error(f"Failed to upload {label}: {response.status_code} - {response.text}")
        except Exception as e:
            logging.error(f"Error uploading to endpoint: {e}")
        finally:
            img_buffer.close()

def start_background_process(queue):
    """Start the endpoint uploader process."""
    upload_process = Process(target=upload_to_endpoint, args=(queue,))
    upload_process.daemon = True
    upload_process.start()
    logging.info(f"Endpoint uploader process started, PID: {upload_process.pid}")
    logging.debug(f"Uploader process alive: {upload_process.is_alive()}")
    return upload_process

def main():
    # Prompt for mode
    mode = input("Select mode (normal/debug): ").lower().strip()
    if mode not in ["normal", "debug"]:
        print("Invalid mode. Exiting.")
        sys.exit(1)
    # Setup logging
    logger = setup_logging(mode)
    # Configuration
    LABEL_FILE = "labelmap.txt"
    MODEL_FILE = "best_int8_edgetpu.tflite"
    THRESHOLDS = {
        "confidence": 0.5,
        "objectness": 0.4,
        "class_score": 0.4
    }
    # List connected TPUs
    logger.debug("Listing connected Edge TPUs")
    for tpu in list_edge_tpus():
        logger.debug(tpu)
    # Initialize components
    try:
        labels = load_labels(LABEL_FILE)
        interpreter, width, height, input_scale, input_zero_point = initialize_tpu_model(MODEL_FILE)
        picam2 = initialize_camera(width, height)
        upload_queue = Queue()
        upload_process = start_background_process(upload_queue)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        sys.exit(1)
    # Main loop
    try:
        while True:
            start_time = time.time()
            # Capture frame
            capture_start = time.time()
            frame = picam2.capture_array()
            capture_time = time.time() - capture_start
            logger.debug(f"Frame captured in {capture_time:.3f}s")
            # Preprocess and run inference
            frame_input = preprocess_frame(frame, input_scale, input_zero_point)
            output = run_inference(interpreter, frame_input)
            # Process output with Soft-NMS
            detections = process_output(interpreter, labels, THRESHOLDS, width, height)
            if detections:
                for det in detections:
                    label = det["label"]
                    # Only process cat, dog, or bird
                    if label.lower() in ["cat", "dog", "bird"]:
                        message = f"Detected: {label} with confidence {det['score']:.2f} (Meta: {det['meta']})"
                        logger.info(message)
                        upload_queue.put((frame.copy(), label))
                        logger.debug(f"Detection ({label}) queued for upload")
                # Stop camera and main loop upon detection
                logger.info("Detection found, stopping camera and main loop")
                picam2.stop()
                picam2.close()
                break
            total_time = time.time() - start_time
            logger.debug(f"Frame processed in {total_time:.3f}s")
    except KeyboardInterrupt:
        logger.info("Stopping script...")
    finally:
        # Cleanup
        logger.info("Cleaning up...")
        if 'picam2' in locals():
            picam2.stop()
            picam2.close()
        upload_queue.put(None)
        upload_process.join()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()