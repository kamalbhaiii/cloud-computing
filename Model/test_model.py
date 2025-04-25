import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# ====== CONFIG ======
MODEL_PATH = "best_float32_edgetpu.tflite"
LABEL_PATH = "labelmap.txt"
IMAGE_PATH = "test.jpg"  # Change this to your image file
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

# ====== Load and preprocess image ======
def preprocess(image):
    resized = cv2.resize(image, (input_width, input_height))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)

# ====== Postprocess and draw detections ======
def draw_detections(image, detections):
    h, w, _ = image.shape
    for det in detections:
        if det[4] < CONF_THRESHOLD:
            continue
        class_id = int(det[5])
        score = float(det[4])
        x_center, y_center, box_w, box_h = det[0:4]

        # Scale back to original image size
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        label = f"{labels[class_id]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ====== Run inference ======
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"[ERROR] Couldn't read image: {IMAGE_PATH}")
    exit()

input_tensor = preprocess(image)
interpreter.set_tensor(input_details[0]['index'], input_tensor)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])[0]

draw_detections(image, output_data)

# ====== Show result ======
cv2.imshow("Prediction", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
