import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter, load_delegate

# Load labels
def load_labels(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return {int(line.split()[0]): line.strip().split(maxsplit=1)[1] for line in lines}

labels = load_labels("labelmap.txt")

# Load model
interpreter = Interpreter(
    model_path="best_float32-2_edgetpu.tflite",
    experimental_delegates=[load_delegate("libedgetpu.so.1")]
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
img_size = input_shape[1]

# Load and preprocess image
image = cv2.imread("test.jpg")
original_image = image.copy()
image_resized = cv2.resize(image, (img_size, img_size))
input_data = image_resized.astype(np.float32) / 255.0
input_data = np.expand_dims(input_data, axis=0)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
predictions = interpreter.get_tensor(output_details[0]['index'])[0]

# Decode predictions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

boxes = []
conf_threshold = 0.5
iou_threshold = 0.5

for pred in predictions:
    x, y, w, h = pred[:4]
    objectness = sigmoid(pred[4])
    class_probs = sigmoid(pred[5:])
    class_id = np.argmax(class_probs)
    class_score = class_probs[class_id]
    conf = objectness * class_score
    if conf > conf_threshold:
        # Convert to corner coordinates
        x1 = int((x - w / 2) * original_image.shape[1] / img_size)
        y1 = int((y - h / 2) * original_image.shape[0] / img_size)
        x2 = int((x + w / 2) * original_image.shape[1] / img_size)
        y2 = int((y + h / 2) * original_image.shape[0] / img_size)
        boxes.append([x1, y1, x2, y2, conf, class_id])

# Check if any boxes were detected
if boxes:
    # Draw boxes and save image
    for box in boxes:
        x1, y1, x2, y2, conf, class_id = box
        label = labels.get(class_id, str(class_id))
        cv2.rectangle(original_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(original_image, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    top_label = labels.get(int(boxes[0][5]), "object")
    save_path = f"test_{top_label}.jpg"
else:
    save_path = "test_no_detection.jpg"

cv2.imwrite(save_path, original_image)
print(f"[INFO] Saved output to {save_path}")
