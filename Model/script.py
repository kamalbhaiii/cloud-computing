import cv2
import numpy as np
import time
from picamera2 import Picamera2

from pycoral.utils.edgetpu import make_interpreter, run_inference
from pycoral.adapters.common import input_size, set_input
from pycoral.adapters.detect import get_objects

# Load the label map
label_map = {}
with open('labelmap.txt', 'r') as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# Load the Edge TPU model using pycoral
model_path = 'best_float32_edgetpu.tflite'  # Use quantized model for Coral TPU
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()
print("Model loaded and allocated using pycoral.")

# Get input details
input_w, input_h = input_size(interpreter)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (input_w, input_h)})
picam2.configure(config)
picam2.start()
print("Pi Camera initialized.")

cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)

try:
    while True:
        frame = picam2.capture_array()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_w, input_h))

        set_input(interpreter, resized)

        start_time = time.time()
        run_inference(interpreter)
        inference_time = time.time() - start_time

        objs = get_objects(interpreter, score_threshold=0.1)

        display_frame = rgb.copy()

        for obj in objs:
            bbox = obj.bbox
            label = label_map.get(obj.id, obj.id)
            score = obj.score

            cv2.rectangle(display_frame, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2)
            cv2.putText(display_frame, f'{label} {score:.2f}', (bbox.xmin, bbox.ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Detections", cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR))
        print(f"Inference time: {inference_time:.4f} seconds")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting gracefully.")
finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("Camera and windows closed.")
