from pycoral.utils.edgetpu import make_interpreter, list_edge_tpus
from pycoral.adapters import common
from PIL import Image
import numpy as np
import os

# ==== Konfiguration ====
MODEL_PATH = "./best_full_integer_quant-3_edgetpu.tflite"
INPUT_IMAGE = "/home/klnaasan/Downloads/pexels-muffinsaurs-994174.jpg"
LABEL_PATH = "./labelmap.txt"
SCORE_THRESHOLD = 0.1  # Testweise niedrig gesetzt

# ==== TPU prüfen ====
def check_tpu():
    tpus = list_edge_tpus()
    if not tpus:
        raise RuntimeError("❌ Kein Edge TPU erkannt.")
    print("✅ Edge TPU erkannt.")
    return True

# ==== Modell laden ====
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Modell nicht gefunden: {model_path}")
    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    print("✅ Modell geladen.")
    return interpreter

# ==== Labels laden ====
def load_labels(label_path):
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"❌ Label-Datei fehlt: {label_path}")
    with open(label_path, 'r') as f:
        lines = f.readlines()
        labels = {}
        for i, line in enumerate(lines):
            line = line.strip()
            if line == "":
                continue
            if " " in line:
                idx, name = line.split(" ", 1)
                labels[int(idx)] = name
            else:
                labels[i] = line
    print("✅ Labels geladen.")
    return labels

# ==== Bild vorbereiten ====
def preprocess_image(image_path, input_size):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Bild nicht gefunden: {image_path}")
    image = Image.open(image_path).convert('RGB').resize(input_size, Image.Resampling.LANCZOS)
    return image

def dequantize_output(tensor, output_details):
    """Dequantisiert Tensor mit Unterstützung für Per-Tensor und Per-Channel Quantisierung."""
    quant = output_details[0]['quantization_parameters']
    scales = quant['scales']
    zero_points = quant['zero_points']

    tensor = tensor.astype(np.float32)

    if len(scales) == 1:
        # Per-tensor quantization
        tensor = scales[0] * (tensor - zero_points[0])
    else:
        # Per-channel quantization
        for i in range(tensor.shape[-1]):
            tensor[..., i] = scales[i] * (tensor[..., i] - zero_points[i])
    return tensor

# ==== YOLO-Ausgabe verarbeiten ====
def process_yolo_output(output_tensor, labels, image_size, threshold=0.3):
    boxes = []
    predictions = output_tensor[0]  # Shape: (8400, 8)
    image_width, image_height = image_size

    for pred in predictions:
        x, y, w, h = pred[:4]
        objectness = pred[4]
        class_probs = pred[5:]

        score = objectness * np.max(class_probs)
        class_id = int(np.argmax(class_probs))

        if score >= threshold:
            # Box-Koordinaten absichern
            x = max(0, min(x, image_width))
            y = max(0, min(y, image_height))
            w = max(1, min(w, image_width - x))
            h = max(1, min(h, image_height - y))

            label = labels.get(class_id, f"Unbekannt ({class_id})")
            boxes.append({
                "bbox": [round(x, 1), round(y, 1), round(w, 1), round(h, 1)],
                "score": round(float(score), 3),
                "class_id": class_id,
                "label": label
            })
    return boxes

# ==== Hauptfunktion ====
if __name__ == "__main__":
    try:
        check_tpu()
        interpreter = load_model(MODEL_PATH)
        labels = load_labels(LABEL_PATH)

        input_size = common.input_size(interpreter)
        image = preprocess_image(INPUT_IMAGE, input_size)
        print("✅ Bild vorbereitet.")

        common.set_input(interpreter, image)
        interpreter.invoke()
        print("✅ Inferenz abgeschlossen.")

        # Tensor holen + dequantisieren
        raw_output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        output_tensor = dequantize_output(raw_output, interpreter.get_output_details())
        
        # Debug: Tensor-Statistik
        print(f"🧪 Output Tensor: shape={output_tensor.shape}, min={np.min(output_tensor):.2f}, max={np.max(output_tensor):.2f}")

        # Originalgröße des Bildes für BBox-Skalierung
        original_image = Image.open(INPUT_IMAGE)
        results = process_yolo_output(output_tensor, labels, original_image.size, threshold=SCORE_THRESHOLD)

        if not results:
            print("⚠️ Keine Objekte erkannt.")
        else:
            print(f"✅ {len(results)} Objekte erkannt:\n")
            for det in results:
                print(f"→ Label: {det['label']}, Score: {det['score']}, BBox: {det['bbox']}")

    except Exception as e:
        print(f"❌ Fehler: {e}")