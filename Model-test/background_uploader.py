import mimetypes
import os
import requests

def upload_image_to_db(image_path: str, category: str):
    url = "http://server.local/api/images"

    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"  # fallback

    with open(image_path, "rb") as img_file:
        files = {
            "image": (os.path.basename(image_path), img_file, mime_type)
        }
        data = {"category": category}

        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            print(f"[INFO] Image uploaded successfully: {category}")
            os.remove(image_path)
        except requests.RequestException as e:
            print(f"[ERROR] Failed to upload image: {e.response.text}")
            os.remove(image_path)
