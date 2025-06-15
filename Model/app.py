from flask import Flask, request, jsonify
import io
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load TFLite model
MODEL_PATH = 'best_int8.tflite'
LABEL_PATH = 'labelmap.txt'

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_height, input_width = input_shape[1], input_shape[2]
input_dtype = input_details[0]['dtype']

# Load labels
with open(LABEL_PATH, 'r') as f:
    labels = [line.strip().split(' ', 1)[1] for line in f]

def preprocess_image(image: Image.Image):
    image = image.resize((input_width, input_height))
    image_array = np.asarray(image)

    image_array = image_array.astype(input_dtype)

    if input_dtype == np.float32:
        image_array = image_array / 255.0

    input_tensor = np.expand_dims(image_array, axis=0)
    return input_tensor

@app.route('/predict', methods=['POST'])
def predict():
    if 'frame' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['frame']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")

    print(f"[INFO] Received image: size={image.size}, mode={image.mode}")

    # Preprocess and set input
    input_tensor = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], input_tensor)

    # Inference
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"[DEBUG] Raw model output: {output_data}")

    # Decode prediction
    scores = output_data.flatten()
    predicted_index = int(np.argmax(scores))
    confidence = float(scores[predicted_index])

    if 0 <= predicted_index < len(labels):
        prediction = labels[predicted_index]
    else:
        prediction = 'Unknown'

    return jsonify({
        'prediction': prediction,
        'confidence': round(confidence, 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
