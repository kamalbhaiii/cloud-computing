import requests
import cv2
import tempfile
import os

def send_telegram_message(bot_token, user_id, message, image=None):
    try:
        # Send text message
        requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            data={"chat_id": user_id, "text": message, "parse_mode": "HTML"}
        )

        # Send photo if provided
        if image is not None:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                cv2.imwrite(temp_file.name, image)
                with open(temp_file.name, 'rb') as img_file:
                    requests.post(
                        f"https://api.telegram.org/bot{bot_token}/sendPhoto",
                        data={"chat_id": user_id},
                        files={"photo": img_file}
                    )
                os.unlink(temp_file.name)
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram notification: {e}")


def telegram_notifier(queue, bot_token, user_id):
    print("[DEBUG] Telegram notifier started")
    while True:
        item = queue.get()
        if item is None:
            print("[DEBUG] Telegram notifier stopping...")
            break
        frame, message, confidence = item
        send_telegram_message(bot_token, user_id, message, image=frame)
