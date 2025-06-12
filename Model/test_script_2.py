import numpy as np
from pycoral.utils import edgetpu
from pycoral.adapters import common
from pycoral.utils.edgetpu import list_edge_tpus
from picamera2 import Picamera2
import cv2
import time
import threading
import queue
import requests
from io import BytesIO
from PIL import Image

# Configuration
MODEL_PATH = 'best_int8_edgetpu.tflite'
LABEL_MAP_PATH = 'labelmap.txt'
THRESHOLD = 0.5
API_ENDPOINT = 'http://192.168.137.178:30070/api/images'
INPUT_SIZE = (640, 640)

# Load label map
def load_labels(path):
    with open(path, 'r') as f:
        labels = {}
        for line in f:
            id, label = line.strip().split(' ', 1)
            labels[int(id)] = label
    return labels

# Convert single tensor to four tensors (boxes, classes, scores, num_detections)
def convert_tensor_to_tensors(output_tensor, threshold=0.5):
    print(f"Output tensor shape: {output_tensor.shape}")
    print(f"Output tensor sample: {output_tensor[:5]}")  # Print first few detections for debugging

    # Initialize outputs
    boxes = []
    classes = []
    scores = []
    num_detections = 0

    try:
        # Assuming output_tensor is [num_detections, 6] with [y_min, x_min, y_max, x_max, score, class]
        for detection in output_tensor:
            if len(detection) < 6:
                print(f"Warning: Detection has unexpected format: {detection}")
                continue
            score = float(detection[4])
            if score > threshold:
                # Ensure coordinates are normalized (0-1)
                box = [float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3])]
                boxes.append(box)
                classes.append(int(detection[5]))
                scores.append(score)
                num_detections += 1
    except Exception as e:
        print(f"Error processing output tensor: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([0])

    # Convert to numpy arrays
    boxes = np.array(boxes, dtype=np.float32)
    classes = np.array(classes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    num_detections = np.array([num_detections], dtype=np.float32)

    print(f"Converted: {num_detections[0]} detections")
    return boxes, classes, scores, num_detections

# Background process to send data to API
def send_to_api(queue):
    while True:
        try:
            image, category = queue.get()
            print(f"Sending to API: Category={category}")

            # Convert image to bytes
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            img_byte_arr = BytesIO()
            img_pil.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            # Prepare multipart/form-data
            files = {'image': ('image.jpg', img_byte_arr, 'image/jpeg')}
            data = {'category': category}

            # Send POST request
            response = requests.post(API_ENDPOINT, files=files, data=data)
            print(f"API Response: {response.status_code} - {response.text}")
            
            queue.task_done()
        except Exception as e:
            print(f"Error in API send: {e}")
            queue.task_done()

def main():
    # Check for EdgeTPU
    edge_tpus = list_edge_tpus()
    if not edge_tpus:
        print("No EdgeTPU detected!")
        return
    print(f"Found EdgeTPU: {edge_tpus}")

    # Initialize interpreter
    try:
        interpreter = edgetpu.make_interpreter(MODEL_PATH)
        interpreter.allocate_tensors()
        print("Interpreter initialized")

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"Model input details: {input_details}")
        print(f"Model output details: {output_details}")
    except Exception as e:
        print(f"Error initializing interpreter: {e}")
        return

    # Load labels
    try:
        labels = load_labels(LABEL_MAP_PATH)
        print(f"Loaded labels: {labels}")
    except Exception as e:
        print(f"Error loading labels: {e}")
        return

    # Initialize camera
    try:
        camera = Picamera2()
        camera_config = camera.create_preview_configuration(main={"size": INPUT_SIZE})
        camera.configure(camera_config)
        camera.start()
        print("Camera initialized")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    # Start background API sender
    api_queue = queue.Queue()
    api_thread = threading.Thread(target=send_to_api, args=(api_queue,), daemon=True)
    api_thread.start()
    print("API sender thread started")

    try:
        detection_made = False
        while not detection_made:
            # Capture image
            frame = camera.capture_array()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print(f"Captured frame shape: {frame_rgb.shape}")

            # Preprocess image
            try:
                resized = cv2.resize(frame_rgb, INPUT_SIZE)
                input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
                print(f"Input data shape: {input_data.shape}")
                common.set_input(interpreter, input_data)
                print("Image preprocessed")
            except Exception as e:
                print(f"Error preprocessing image: {e}")
                continue

            # Run inference
            try:
                start_time = time.time()
                interpreter.invoke()
                # Copy output tensor to avoid reference issues
                output_tensor = np.copy(common.output_tensor(interpreter, 0))
                print(f"Inference time: {time.time() - start_time:.3f}s")
            except Exception as e:
                print(f"Error during inference: {e}")
                continue

            # Convert output to four tensors
            try:
                boxes, classes, scores, num_detections = convert_tensor_to_tensors(output_tensor, THRESHOLD)
            except Exception as e:
                print(f"Error converting tensors: {e}")
                continue

            # Process detections
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            for i in range(int(num_detections[0])):
                class_id = int(classes[i])
                score = scores[i]
                box = boxes[i]

                # Scale box coordinates to original frame size
                h, w = frame.shape[:2]
                y_min = int(box[0] * h)
                x_min = int(box[1] * w)
                y_max = int(box[2] * h)
                x_max = int(box[3] * w)

                # Draw box and label
                cv2.rectangle(frame_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                label = f"{labels[class_id]}: {score:.2f}"
                cv2.putText(frame_bgr, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Queue for API if valid category
                if labels[class_id] in ['cat', 'dog', 'bird']:
                    api_queue.put((frame_bgr, labels[class_id]))
                    print(f"Queued detection: {labels[class_id]}")

                detection_made = True  # Stop after first detection

            # Display frame
            cv2.imshow('Detection', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Cleanup completed")

if __name__ == '__main__':
    main()