import json
from datetime import datetime
import threading

config_lock = threading.Lock()
CONFIG_FILE = "config.json"

def load_config():
   with config_lock:
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"failed to read config: {e}")
            return {}

def save_classification(label, confidence):
    with config_lock:
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                        
                config["latest_classification"] = {
                    "result": label,
                    "confidence": confidence,
                    "timestamp": datetime.utcnow().isoformat()
                }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"failed to save classification: {e}")
       
