import numpy as np
import cv2
from PIL import Image
from picamera2 import Picamera2
import tflite_runtime.interpreter as tflite
import time

# Load class labels from labelmap.txt
label_map = {}
with open("labelmap.txt", "r") as f:
    for line in f:
        idx, label = line.strip().split(" ", 1)
        label_map[int(idx)] = label

# Load the Edge TPU model
interpreter = tflite.Interpreter(
    model_path="best_float32-2_edgetpu.tflite",
    experimental_delegates=[tflite.load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

# Initialize the PiCamera2
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(2)

print("Press 'q' to quit.")

while True:
    frame = picam2.capture_array()
    img_resized = cv2.resize(frame, (input_width, input_height))
    if img_resized.shape[2] == 4:
        img_resized = img_resized[:, :, :3]  # Drop alpha channel if present
    input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    detections = output_data[0]

    objectness_threshold = 0.4
    class_threshold = 0.25

    for det in detections:
        x, y, w, h = det[0:4]
        objectness = det[4]
        class_scores = det[5:]

        if objectness > objectness_threshold:
            class_id = np.argmax(class_scores)
            confidence = class_scores[class_id]
            if confidence > class_threshold:
                # Rescale boxes to original image size
                frame_h, frame_w = frame.shape[:2]
                x *= frame_w / input_width
                w *= frame_w / input_width
                y *= frame_h / input_height
                h *= frame_h / input_height

                x1 = int(x - w / 2)
                y1 = int(y - h / 2)
                x2 = int(x + w / 2)
                y2 = int(y + h / 2)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = label_map.get(class_id, f"Class {class_id}")
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show result
    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
