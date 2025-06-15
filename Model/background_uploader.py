import requests

def upload_image_to_db(image_path: str, category: str):
    print(image_path, category)
    url = "http://192.168.137.178:30070/api/images"
    
    with open(image_path, "rb") as img_file:
        files = {"image": img_file}  # Proper binary file upload
        data = {"category": category}  # Form data
        
        try:
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            print(f"[INFO] Image uploaded successfully: {category}")
        except requests.RequestException as e:
            print(f"[ERROR] Failed to upload image: {e.response.text}")
