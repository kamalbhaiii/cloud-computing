import os
from minio import Minio # type: ignore
from minio.error import S3Error # type: ignore
from PIL import Image
import numpy as np
import time
from datetime import datetime

def upload_to_minio(queue):
    # MinIO configuration
    minio_client = Minio(
        "192.168.137.178:30010",
        access_key="UAW9vG03CEqhmAJoOaff",  # Replace with your MinIO access key
        secret_key="ZDCoFLJDLQPw848mbxZ7KAYUrCoY88MsHEoC40qk",  # Replace with your MinIO secret key
        secure=False  # Set to True if using HTTPS
    )
    bucket_name = "wildlife-detections"

    # Create bucket if it doesn't exist
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            print("[DEBUG] Created MinIO bucket:", bucket_name)
        else:
            print("[DEBUG] MinIO bucket already exists:", bucket_name)
    except S3Error as e:
        print("[ERROR] Failed to create/check MinIO bucket:", e)
        return

    # Create temporary directory for images
    temp_dir = "temp_images_ignore"
    os.makedirs(temp_dir, exist_ok=True)
    print("[DEBUG] Temporary image directory created:", temp_dir)

    # Process queue
    while True:
        item = queue.get()
        if item is None:
            print("[DEBUG] Received stop signal, exiting uploader")
            break

        frame, label, confidence = item
        print("[DEBUG] Received detection from queue: label={}, confidence={:.4f}".format(label, confidence))

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{label}_{timestamp}_{confidence:.4f}.jpg"
        temp_path = os.path.join(temp_dir, filename)

        # Save image
        try:
            image = Image.fromarray(frame.astype(np.uint8))
            image.save(temp_path, "JPEG")
            print("[DEBUG] Image saved locally:", temp_path)
        except Exception as e:
            print("[ERROR] Failed to save image {}: {}".format(temp_path, e))
            continue

        # Upload to MinIO
        try:
            minio_client.fput_object(bucket_name, filename, temp_path)
            print("[DEBUG] Uploaded image to MinIO: {}/{}".format(bucket_name, filename))
        except S3Error as e:
            print("[ERROR] Failed to upload image {} to MinIO: {}".format(filename, e))
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
                print("[DEBUG] Deleted temporary image:", temp_path)
            except OSError as e:
                print("[ERROR] Failed to delete temporary image {}: {}".format(temp_path, e))