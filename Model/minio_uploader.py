import os
from minio import Minio
from minio.error import S3Error
from PIL import Image
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("minio_uploader.log")
    ]
)

def upload_to_minio(queue):
    """Upload images from queue to MinIO."""
    # MinIO configuration
    minio_client = Minio(
        "192.168.137.178:30010",
        access_key="UAW9vG03CEqhmAJoOaff",
        secret_key="ZDCoFLJDLQPw848mbxZ7KAYUrCoY88MsHEoC40qk",
        secure=False
    )
    bucket_name = "wildlife-detections"

    # Create bucket if it doesn't exist
    try:
        if not minio_client.bucket_exists(bucket_name):
            minio_client.make_bucket(bucket_name)
            logging.info(f"Created MinIO bucket: {bucket_name}")
        else:
            logging.info(f"MinIO bucket already exists: {bucket_name}")
    except S3Error as e:
        logging.error(f"Failed to create/check MinIO bucket: {e}")
        return

    # Create temporary directory for images
    temp_dir = "temp_images_ignore"
    os.makedirs(temp_dir, exist_ok=True)
    logging.info(f"Temporary image directory created: {temp_dir}")

    # Process queue
    while True:
        item = queue.get()
        if item is None:
            logging.info("Received stop signal, exiting uploader")
            break

        frame, message, label, confidence = item
        logging.debug(f"Received detection: label={label}, confidence={confidence:.4f}")

        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{label}_{timestamp}_{confidence:.4f}.jpg"
        temp_path = os.path.join(temp_dir, filename)

        # Save image
        try:
            image = Image.fromarray(frame.astype(np.uint8))
            image.save(temp_path, "JPEG")
            logging.debug(f"Image saved locally: {temp_path}")
        except Exception as e:
            logging.error(f"Failed to save image {temp_path}: {e}")
            continue

        # Upload to MinIO
        try:
            minio_client.fput_object(bucket_name, filename, temp_path)
            logging.debug(f"Uploaded image to MinIO: {bucket_name}/{filename}")
        except S3Error as e:
            logging.error(f"Failed to upload image {filename} to MinIO: {e}")
        finally:
            # Clean up temporary file
            try:
                os.remove(temp_path)
                logging.debug(f"Deleted temporary image: {temp_path}")
            except OSError as e:
                logging.error(f"Failed to delete temporary image {temp_path}: {e}")