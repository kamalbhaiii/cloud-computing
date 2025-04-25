from picamera2 import Picamera2
import cv2
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# ====== CONFIG ======
MODEL_PATH = "best_float32_edgetpu.tflite"
LABEL_PATH = "labelmap.txt"
INPUT_SIZE = 320  # Roboflow will provide this
CONF_THRESHOLD = 0.4

# ====== Load labels ======
with open(LABEL_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# ====== Load TFLite model with EdgeTPU delegate ======
interpreter = Interpreter(
    model_path=MODEL_PATH,
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_height = input_details[0]['shape'][1]
input_width = input_details[0]['shape'][2]

print(f"[INFO] Model expects input size: {input_width}x{input_height}")

# ====== Init camera ======
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1)

# ====== Preprocessing ======
def preprocess(image):
    image_resized = cv2.resize(image, (input_width, input_height))
    image_normalized = image_resized.astype(np.float32) / 255.0  
    return np.expand_dims(image_normalized, axis=0)

# ====== Postprocessing (YOLOv8-specific output) ======
def draw_detections(image, output_data):
    height, width, _ = image.shape

    for detection in output_data:
        if detection[4] < CONF_THRESHOLD:
            continue
        class_id = int(detection[5])
        score = float(detection[4])
        x_center, y_center, w, h = detection[0:4]

        # Scale coords to original image
        x_center *= width
        y_center *= height
        w *= width
        h *= height

        x1 = int(x_center - w / 2)
        y1 = int(y_center - h / 2)
        x2 = int(x_center + w / 2)
        y2 = int(y_center + h / 2)

        label = f"{labels[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ====== Main loop ======
while True:
    frame = picam2.capture_array()
    input_tensor = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # YOLOv8 output format: [x, y, w, h, confidence, class_id]
    # output_data shape: (N, 6)
    draw_detections(frame, output_data)

    cv2.imshow("Wildlife Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

