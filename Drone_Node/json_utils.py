import json
import threading
from PIL import Image
import io
import os

lock = threading.Lock()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COLLECTED_FILE = os.path.join(BASE_DIR, "collected.json")

def load_json():
    with lock:
        try:
            with open(COLLECTED_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"failed to read collect.json file: {e}")
            return {}

def save_result(result):
    with lock:
        try:
            # load the collected json that has all the traps results
            with open(COLLECTED_FILE, "r") as f:
                collected = json.load(f)
                
                # save the results to the associated trap
                trap_id = result["trap_id"]
                collected[trap_id]["results"] = result
                collected[trap_id]["collected"] = True
                
                # modify the collected flag
                collected["data_collected"] = True
                
            # save the changes to collect.json
            with open(COLLECTED_FILE, "w") as f:
                json.dump(collected, f, indent=2)
            print("[collect] results saved")
            return True
        except Exception as e:
            print(f"[collect] failed to save trap results to collected.json: {e}")
            return False

def save_image(image, trap_id):
    with lock:
        try:
            image_path = os.path.join(BASE_DIR, "collected_images", trap_id + ".jpg")
            image.save(image_path)
            print("[collect] image saved as", image_path)
            return True
        except Exception as e:
            print(f"[collect] failed to save image: {e}")
            return False
            
def load_image(trap_id):
    with lock:
        try:
            image_path = os.path.join(BASE_DIR, "collected_images", trap_id + ".jpg")
            image = Image.open(image_path).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")
            image_bytes = buffer.getvalue()
            
            print("[upload] image loaded successfully")
            return image_bytes, len(image_bytes)
        except Exception as e:
            print(f"[upload] error in loading images: {e}")
            
def reset_json():
    with lock:
        try:
            with open(COLLECTED_FILE, "r") as f:
                data = json.load(f)
                for trap_id in data["TRAPS"]:
                    data[trap_id]["collected"] = False
                    data[trap_id]["results"] = {}
                    
                data["data_collected"] = False # reset the flag
            with open(COLLECTED_FILE, "w") as f:
                json.dump(data, f, indent=2)
            print("[Upload] reset completed, drone ready to start next collection trip.")
        except Exception as e:
            print(f"[Upload] failed to reset collect.json file.")

def get_flag():
    with lock:
        try:
            with open(COLLECTED_FILE, "r") as f:
                data = json.load(f)
                return data.get("data_collected", False)
        except Exception as e:
            print(f"[json_utils] failed to read flag: {e}")
            return False
        
    

