import os
import base64
import uuid
from flask import Flask
from flask_socketio import SocketIO, emit
from PIL import Image
from io import BytesIO
from inference_sdk import InferenceHTTPClient
from threading import Thread
from background_uploader import upload_image_to_db  # import background logic

# Flask setup
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="uUiAY4ufKNxPGS9jwXP6"
)
MODEL_ID = "new_datasets_cloud_computing/2"
ALLOWED_LABELS = {"cat", "dog", "bird"}

# Temp folder in script directory
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
os.makedirs(TEMP_DIR, exist_ok=True)

def background_upload(image_path, category):
    # This runs in a separate thread to avoid blocking main event loop
    upload_image_to_db(image_path, category)

@socketio.on("frame")
def handle_frame(base64_data):
    try:
        # Decode image
        image_data = base64.b64decode(base64_data)
        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Save image to ./temp/<unique_id>.jpg
        unique_filename = f"temp_{uuid.uuid4().hex}.jpg"
        file_path = os.path.join(TEMP_DIR, unique_filename)
        image.save(file_path, format="JPEG")

        # Run inference
        try:
            result = CLIENT.infer(file_path, model_id=MODEL_ID)
            predictions = result.get("predictions", [])
            if not predictions:
                emit("prediction", {"prediction": None})
                os.remove(file_path)
                return

            best = max(predictions, key=lambda x: x["confidence"])
            label = best["class"].lower()
            conf = best["confidence"]

            if label in ALLOWED_LABELS:
                emit("prediction", {"prediction": label, "confidence": conf})
                # Start background upload thread ONLY if prediction exists
                Thread(target=background_upload, args=(file_path, label), daemon=True).start()
            else:
                emit("prediction", {"prediction": None})
                os.remove(file_path)  # delete if no allowed label

        except Exception as infer_err:
            print(f"[ERROR] Inference error: {infer_err}")
            emit("prediction", {"prediction": None})
            os.remove(file_path)

    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        emit("prediction", {"prediction": None})

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000)
