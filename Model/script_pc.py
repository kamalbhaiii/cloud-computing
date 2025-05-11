import cv2
import numpy as np
import time
from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
from pycoral.adapters.common import input_size, set_input

print("Starting script...")

# Load the label map
label_map = {}
try:
    with open('labelmap.txt', 'r') as f:
        for line in f:
            idx, label = line.strip().split()
            label_map[int(idx)] = label
    print("Label map loaded successfully.")
except Exception as e:
    print(f"Error loading label map: {e}")
    exit(1)

# Load the Edge TPU model
model_path = 'best_float32_edgetpu.tflite'
try:
    print("Loading Edge TPU model...")
    interpreter = make_interpreter(model_path)
    print("Model loaded. Allocating tensors...")
    interpreter.allocate_tensors()
    print("Tensors allocated successfully.")
except Exception as e:
    print(f"Error loading model or allocating tensors: {e}")
    exit(1)

# Get input details
input_w, input_h = input_size(interpreter)
print(f"Model input size: {input_w}x{input_h}")

# Initialize webcam
print("Initializing webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)
print("Webcam initialized.")

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

# Custom YOLO output parser
def parse_yolo_output(interpreter, score_threshold=0.5, objectness_threshold=0.5, iou_threshold=0.3):
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape: [8, 8400]
    output = output.transpose()  # Shape: [8400, 8]

    x_centers = output[:, 0]
    y_centers = output[:, 1]
    widths = output[:, 2]
    heights = output[:, 3]
    objectness = output[:, 4]
    class_probs = output[:, 5:]

    valid = objectness >= objectness_threshold
    x_centers = x_centers[valid]
    y_centers = y_centers[valid]
    widths = widths[valid]
    heights = heights[valid]
    objectness = objectness[valid]
    class_probs = class_probs[valid]

    if len(x_centers) == 0:
        return []

    class_ids = np.argmax(class_probs, axis=1)
    class_scores = np.max(class_probs, axis=1)
    scores = objectness * class_scores

    valid = scores >= score_threshold
    x_centers = x_centers[valid]
    y_centers = y_centers[valid]
    widths = widths[valid]
    heights = heights[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(x_centers) == 0:
        return []

    xmins = (x_centers - widths / 2) * input_w
    ymins = (y_centers - heights / 2) * input_h
    xmaxs = (x_centers + widths / 2) * input_w
    ymaxs = (y_centers + heights / 2) * input_h

    box_widths = xmaxs - xmins
    box_heights = ymaxs - ymins
    valid = (box_widths >= 20) & (box_heights >= 20)
    xmins = xmins[valid]
    ymins = ymins[valid]
    xmaxs = xmaxs[valid]
    ymaxs = ymaxs[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(xmins) == 0:
        return []

    boxes = np.stack([xmins, ymins, xmaxs, ymaxs], axis=1).tolist()

    indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, iou_threshold)
    if isinstance(indices, np.ndarray):
        indices = indices.flatten()
    return [
        {'bbox': (int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])),
         'id': class_ids[i],
         'score': scores[i]}
        for i in indices
    ]

try:
    while True:
        print("Attempting to capture frame...")
        t1 = time.time()
        ret, frame = cap.read()
        capture_time = time.time() - t1
        if not ret:
            print("Error: Failed to capture frame. Exiting loop.")
            break
        print("Frame captured successfully.")

        t1 = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
        preprocess_time = time.time() - t1
        print("Preprocessing completed.")

        print("Edge TPU devices:", list_edge_tpus())
        print("Running inference...")
        t1 = time.time()
        set_input(interpreter, resized)
        interpreter.invoke()
        inference_time = time.time() - t1
        print("Inference completed.")

        print("Parsing output...")
        t1 = time.time()
        objs = parse_yolo_output(interpreter, score_threshold=0.5, objectness_threshold=0.5, iou_threshold=0.3)
        parse_time = time.time() - t1
        print(f"Detected {len(objs)} objects.")

        t1 = time.time()
        display_frame = rgb.copy()
        for obj in objs[:10]:
            bbox = obj['bbox']
            label = label_map.get(obj['id'], obj['id'])
            score = obj['score']
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(display_frame, f'{label} {score:.2f}', (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        display_time = time.time() - t1
        print("Display frame prepared.")

        print("Updating display...")
        cv2.imshow("Detections", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        print("Display updated.")

        print(f"Times: Capture={capture_time:.4f}s, Preprocess={preprocess_time:.4f}s, Inference={inference_time:.4f}s, Parse={parse_time:.4f}s, Display={display_time:.4f}s")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quit key pressed. Exiting loop.")
            break

except Exception as e:
    print(f"Error in main loop: {e}")
finally:
    print("Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam and windows closed.")