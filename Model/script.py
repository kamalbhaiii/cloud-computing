import cv2
from ultralytics import YOLO
import time

# Load the Coral-compatible YOLOv8 model
model = YOLO("best_float32_edgetpu.tflite")

# Open the Pi Camera
cap = cv2.VideoCapture(0)

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("âŒ Failed to open Pi Camera.")
    exit()

print("ðŸ“¸ Pi Camera opened successfully. Running inference...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("âŒ Failed to capture frame.")
            break

        start_time = time.time()
        
        # Run inference
        results = model.predict(source=frame, device='tpu')
        
        # Show results
        annotated_frame = results[0].plot()
        end_time = time.time()

        # FPS display
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display annotated frame
        cv2.imshow("YOLOv8 + Coral TPU Inference", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("ðŸ›‘ Inference stopped by user.")

# Cleanup
cap.release()
cv2.destroyAllWindows()
