import requests
import cv2
import tempfile
import os
import logging

def send_telegram_message(bot_token, user_ids, message, image=None):
    """Send Telegram message and optional image."""
    for id in user_ids:
        try:
            # Send text message
            response = requests.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                data={"chat_id": id, "text": message, "parse_mode": "HTML"},
                timeout=5
            )
            logging.debug(f"Text message response: {response.status_code}, {response.json()}")
            if response.status_code != 200:
                logging.error(f"Failed to send text message: {response.json()}")
            # Send photo if provided
            if image is not None:
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                    cv2.imwrite(temp_file.name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    logging.debug(f"Image saved to {temp_file.name}")
                    with open(temp_file.name, 'rb') as img_file:
                        response = requests.post(
                            f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                            data={"chat_id": user_id},
                            files={"photo": img_file},
                            timeout=5
                        )
                        logging.debug(f"Photo response: {response.status_code}, {response.json()}")
                        if response.status_code != 200:
                            logging.error(f"Failed to send photo: {response.json()}")
                    os.unlink(temp_file.name)
        except Exception as e:
            logging.error(f"Failed to send Telegram notification: {e}")

def telegram_notifier(queue, bot_token, user_id):
    """Process queue items and send Telegram notifications."""
    logging.info(f"Telegram notifier started, PID: {os.getpid()}")
    try:
        while True:
            item = queue.get()
            if item is None:
                logging.info("Telegram notifier stopping...")
                break
            logging.debug(f"Queue item: {item}")
            frame, message, predicted_label, confidence = item
            send_telegram_message(bot_token, user_id, message, image=frame)
    except Exception as e:
        logging.error(f"Telegram notifier crashed: {e}")
        raise