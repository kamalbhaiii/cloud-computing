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
import telegram_notifier
import minio_uploader

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
    
    # Ensure child loggers propagate to root
    logging.getLogger('telegram_notifier').propagate = True
    logging.getLogger('minio_uploader').propagate = True
    
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

def run_inference(interpreter, frame_input):
    """Run inference on the Edge TPU."""
    inference_start = time.time()
    common.set_input(interpreter, frame_input)
    interpreter.invoke()
    output = np.copy(common.output_tensor(interpreter, 0))
    inference_time = time.time() - inference_start
    logging.debug(f"Inference completed in {inference_time:.3f}s, output shape: {output.shape}")
    return output

def process_output(output, labels, thresholds):
    """Process model output and return detection if thresholds are met."""
    process_start = time.time()
    output = output[0]  # Remove batch dimension
    objectness_scores = output[4, :]  # Objectness scores
    class_scores = output[5:9, :]  # Class scores for 4 classes

    max_idx = np.argmax(objectness_scores)
    max_objectness = objectness_scores[max_idx]
    logging.debug(f"Highest objectness score: {max_objectness:.4f} at index {max_idx}")
    logging.debug(f"Top 5 objectness scores: {np.sort(objectness_scores)[-5:][::-1]}")

    if max_objectness < thresholds["objectness"]:
        logging.debug(f"No detection with objectness >= {thresholds['objectness']}")
        return None

    detection_class_scores = class_scores[:, max_idx]
    max_class_idx = np.argmax(detection_class_scores)
    max_class_score = detection_class_scores[max_class_idx]
    logging.debug(f"Class scores: {detection_class_scores}, Predicted class index: {max_class_idx}")

    if max_class_score < thresholds["class_score"]:
        logging.debug(f"No class with score >= {thresholds['class_score']}")
        return None

    predicted_label = labels.get(max_class_idx, "Unknown")
    confidence = max_objectness * max_class_score

    if confidence < thresholds["confidence"]:
        logging.debug(f"Confidence {confidence:.4f} below threshold {thresholds['confidence']}")
        return None

    process_time = time.time() - process_start
    logging.debug(f"Output processed in {process_time:.3f}s")
    return predicted_label, confidence

def start_background_processes(queue, telegram_token, telegram_user_id):
    """Start MinIO uploader and Telegram notifier processes."""
    telegram_process = Process(target=telegram_notifier.telegram_notifier, args=(queue, telegram_token, telegram_user_id))
    telegram_process.daemon = True
    telegram_process.start()
    logging.info(f"Telegram notifier process started, PID: {telegram_process.pid}")
    logging.debug(f"Telegram process alive: {telegram_process.is_alive()}")

    uploader_process = Process(target=minio_uploader.upload_to_minio, args=(queue,))
    uploader_process.daemon = True
    uploader_process.start()
    logging.info(f"MinIO uploader process started, PID: {uploader_process.pid}")
    logging.debug(f"MinIO process alive: {uploader_process.is_alive()}")

    return telegram_process, uploader_process

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
    TELEGRAM_TOKEN = "7684579144:AAGOhHlKZ9IEKHiCHD9gBT1eE3OgfdNC7no"
    TELEGRAM_USER_ID = ["7246139728", "1411517116", "1112158483"]
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
        telegram_process, uploader_process = start_background_processes(upload_queue, TELEGRAM_TOKEN, TELEGRAM_USER_ID)
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

            # Process output
            result = process_output(output, labels, THRESHOLDS)
            if result is None:
                continue

            predicted_label, confidence = result
            message = f"Predicted: {predicted_label} with Confidence: {confidence:.4f}"
            logger.info(message)

            # Queue detection
            upload_queue.put((frame, message, predicted_label, confidence))
            logger.debug("Detection queued for upload")

            total_time = time.time() - start_time
            logger.debug(f"Frame processed in {total_time:.3f}s")

    except KeyboardInterrupt:
        logger.info("Stopping script...")
    finally:
        # Cleanup
        picam2.stop()
        picam2.close()
        upload_queue.put(None)
        telegram_process.join()
        uploader_process.join()
        logger.info("Cleanup complete")

if __name__ == "__main__":
    main()