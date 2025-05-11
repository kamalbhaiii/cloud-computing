import numpy as np
import time
from picamera2 import Picamera2
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters.common import input_size, set_input
from PIL import Image

# Load the label map
label_map = {}
with open('labelmap.txt', 'r') as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# Load the Edge TPU model using pycoral
model_path = 'best_float32_edgetpu.tflite'
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
print("Model loaded and allocated using pycoral.")

# Get input shape
input_w, input_h = input_size(interpreter)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (input_w, input_h)})
picam2.configure(config)
picam2.start()
print("Pi Camera initialized.")

# Post-processing parameters
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

def iou(box1, box2):
    """Intersection over Union for two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area

def non_max_suppression(predictions, iou_threshold=0.5):
    """Applies NMS on prediction results."""
    if len(predictions) == 0:
        return []

    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    final_predictions = []

    while predictions:
        best = predictions.pop(0)
        final_predictions.append(best)
        predictions = [
            p for p in predictions
            if p[0] != best[0] or iou(p[1:5], best[1:5]) < iou_threshold
        ]

    return final_predictions

try:
    while True:
        # Capture frame
        frame = picam2.capture_array()
        rgb = frame[..., [2, 1, 0]]  # BGR to RGB
        image = Image.fromarray(rgb).resize((input_w, input_h), Image.Resampling.LANCZOS)
        input_tensor = np.asarray(image)

        set_input(interpreter, input_tensor)
        start_time = time.time()
        interpreter.invoke()
        inference_time = time.time() - start_time

        # Get output
        output_details = interpreter.get_output_details()
        output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # (num_boxes, 7)

        print("Output shape:", output_data.shape)
        print("Sample output:", output_data[:5])

        predictions = []
        for pred in output_data:
            # YOLO-style format: [x, y, w, h, confidence, class1_score, ..., classN_score]
            if pred[4] < CONFIDENCE_THRESHOLD:
                continue

            x_center, y_center, width, height = pred[0], pred[1], pred[2], pred[3]
            x1 = int((x_center - width / 2) * frame.shape[1])
            y1 = int((y_center - height / 2) * frame.shape[0])
            x2 = int((x_center + width / 2) * frame.shape[1])
            y2 = int((y_center + height / 2) * frame.shape[0])
            confidence = pred[4]
            class_id = np.argmax(pred[5:])
            score = pred[5 + class_id]

            if score < CONFIDENCE_THRESHOLD:
                continue

            predictions.append((class_id, x1, y1, x2, y2, score))

        final_detections = non_max_suppression(predictions, IOU_THRESHOLD)

        # Display results
        if final_detections:
            for class_id, x1, y1, x2, y2, score in final_detections:
                label = label_map.get(class_id, f"Class {class_id}")
                print(f"Detected: {label} | Confidence: {score:.2f} | Box: [{x1}, {y1}, {x2}, {y2}]")
        else:
            print("No objects detected.")

        print(f"Inference Time: {inference_time:.3f} sec")
        print("-" * 50)

        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    picam2.stop()
    print("Camera closed.")
