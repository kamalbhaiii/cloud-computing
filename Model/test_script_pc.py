import cv2
import numpy as np
import tensorflow as tf

# Load the label map
label_map = {}
with open('labelmap.txt', 'r') as f:
    for line in f:
        idx, label = line.strip().split()
        label_map[int(idx)] = label

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='best_float32-2.tflite')  # Replace with your .tflite model path
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print output details for debugging
print("Output details:", output_details)

# Load and preprocess the test image, resizing to 640x640
image = cv2.imread('test_dog.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
resized_image = cv2.resize(image_rgb, (640, 640))  # Resize to 640x640
input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)

# Normalize input (YOLOv8 typically expects [0, 1])
input_data /= 255.0

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor [1, 8, 8400]
output = interpreter.get_tensor(output_details[0]['index'])[0]  # [8, 8400]
output = output.T  # Transpose to [8400, 8]

# Extract boxes and class probabilities
num_classes = len(label_map)
boxes = output[:, :4]  # [8400, 4] (potential x_min, y_min, x_max, y_max or other format)
class_probs = output[:, 4:4+num_classes]  # [8400, num_classes]
scores = np.max(class_probs, axis=1)  # [8400]
classes = np.argmax(class_probs, axis=1).astype(int)  # [8400]

# Apply non-maximum suppression
def nms(boxes, scores, iou_threshold=0.5):
    x1 = boxes[:, 0]  # x_min
    y1 = boxes[:, 1]  # y_min
    x2 = boxes[:, 2]  # x_max
    y2 = boxes[:, 3]  # y_max
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        iou = w * h / (areas[i] + areas[order[1:]] - w * h)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

# Filter detections
valid_indices = np.where(scores > 0.1)[0]  # Confidence threshold
if len(valid_indices) == 0:
    valid_indices = [np.argmax(scores)]  # Use highest-scoring detection if none above threshold
else:
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    nms_indices = nms(valid_boxes, valid_scores)
    valid_indices = valid_indices[nms_indices]

# Image dimensions (now 640x640)
img_height, img_width = 640, 640

# Debug: Print detection details
print("Detections:")
for i, idx in enumerate(valid_indices):
    box = boxes[idx]
    score = scores[idx]
    class_id = classes[idx]
    # Test [x_min, y_min, x_max, y_max] format
    x_min, y_min, x_max, y_max = box
    x1 = int(x_min * img_width)
    y1 = int(y_min * img_height)
    x2 = int(x_max * img_width)
    y2 = int(y_max * img_height)
    # Alternative: Convert from [x_center, y_center, width, height] if needed
    x_center, y_center, width, height = box
    x1_center = int((x_center - width / 2) * img_width)
    y1_center = int((y_center - height / 2) * img_height)
    x2_center = int((x_center + width / 2) * img_width)
    y2_center = int((y_center + height / 2) * img_height)
    print(f"Detection {i}: Class={label_map[class_id]} (ID={class_id}), Score={score:.4f}, "
          f"Box (x,y,x,y)=[{x1}, {y1}, {x2}, {y2}], "
          f"Box (center,w,h)=[{x1_center}, {y1_center}, {x2_center}, {y2_center}]")

# Draw bounding boxes and determine predicted class
predicted_class = 'unknown'
for i, idx in enumerate(valid_indices):
    box = boxes[idx]
    score = scores[idx]
    class_id = classes[idx]
    # Test [x_min, y_min, x_max, y_max] format
    x_min, y_min, x_max, y_max = box
    x1 = int(x_min * img_width)
    y1 = int(y_min * img_height)
    x2 = int(x_max * img_width)
    y2 = int(y_max * img_height)
    # Alternative: Convert from [x_center, y_center, width, height]
    x_center, y_center, width, height = box
    x1_center = int((x_center - width / 2) * img_width)
    y1_center = int((y_center - height / 2) * img_height)
    x2_center = int((x_center + width / 2) * img_width)
    y2_center = int((y_center + height / 2) * img_height)

    # Favor the box with larger area (test both formats)
    area_xy = (x2 - x1) * (y2 - y1)
    area_center = (x2_center - x1_center) * (y2_center - y1_center)
    if area_center > area_xy:
        x1, y1, x2, y2 = x1_center, y1_center, x2_center, y2_center
    label = label_map[class_id]
    
    # Draw the bounding box
    cv2.rectangle(resized_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(resized_image, f'{label} {score:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Use the first valid class as predicted class
    if predicted_class == 'unknown':
        predicted_class = label

# Save the 640x640 image
output_filename = f'pred_{predicted_class}.jpg'
cv2.imwrite(output_filename, resized_image)

print(f'Image saved as {output_filename}')