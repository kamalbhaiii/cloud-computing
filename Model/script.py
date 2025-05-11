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

# Get input details
input_w, input_h = input_size(interpreter)

# Initialize PiCamera2
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (input_w, input_h)})
picam2.configure(config)
picam2.start()
print("Pi Camera initialized.")

def parse_yolo_output(output, threshold=0.1):
    """
    Parse the YOLO output from Edge TPU inference result.
    output: The raw tensor output from the model
    threshold: Confidence score threshold for filtering detections
    Returns: List of detected objects with class, score, and bounding box
    """
    detections = []
    # Each detection consists of 8 values: [x, y, w, h, conf, class_1, class_2, ...]
    for i in range(output.shape[1]):  # Iterate through the 8 elements per detection
        detection = output[0, i]
        # Assuming the first 4 are coordinates and confidence is at index 4
        x, y, w, h, conf = detection[0], detection[1], detection[2], detection[3], detection[4]
        if conf > threshold:  # Only consider detections with confidence > threshold
            class_idx = np.argmax(detection[5:])  # Get the class index with the highest score
            class_label = label_map.get(class_idx, "Unknown")
            detections.append({
                'class': class_label,
                'confidence': conf,
                'bbox': (x, y, w, h)
            })
    return detections

try:
    while True:
        # Capture frame as numpy array
        frame = picam2.capture_array()

        # Convert BGR to RGB using NumPy
        rgb = frame[..., [2, 1, 0]]  # Swap BGR to RGB by reordering channels

        # Resize using PIL
        pil_image = Image.fromarray(rgb)
        resized = pil_image.resize((input_w, input_h), Image.Resampling.LANCZOS)  # Updated for Pillow 10+
        resized_array = np.array(resized)

        # Prepare input for the model
        set_input(interpreter, resized_array)

        # Run inference
        start_time = time.time()
        interpreter.invoke()  # Run the inference
        inference_time = time.time() - start_time

        # Get the output tensor and parse it
        output = interpreter.tensor(interpreter.get_output_details()[0]['index'])()[0]
        detections = parse_yolo_output(output)

        # Print detections to terminal
        if detections:
            for det in detections:
                print(f"Predicted: {det['class']} with Confidence: {det['confidence']:.2f}")
                print(f"Bounding box: {det['bbox']}")
        else:
            print("No objects detected.")

        print(f"Inference time: {inference_time:.4f} seconds")
        print("-" * 50)

        # Small delay to prevent overwhelming the terminal
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting gracefully.")
finally:
    picam2.stop()
    print("Camera closed.")
