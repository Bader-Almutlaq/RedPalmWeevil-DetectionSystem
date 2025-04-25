import socket
import json
import io
from PIL import Image
import serial
import pynmea2
import nmap
from config_utils import load_config
import time

config = load_config()

trap_id = config["trap_id"]
PORT = config["network"]["PORT"]
SUBNET = config["network"]["SUBNET"]
RECEIVER_IP = config["network"]["HOST"]

def wait_for_receiver(target_ip=None, subnet=SUBNET):
    """
    Ping the available addresses and wait until the given
    IP address to show up in the network list.
    The function blocks until the receiver IP is found.

    Args:
        target_ip (str): IP address for the target machine
        subnet (srt): Subnet to scan (e.g., '192.168.1.0/24').

    Returns:
        None. Function returns only when the receiver is found.
    """
    
    nm = nmap.PortScanner()
    print(f"[NMAP] Scanning {subnet} for receiver...")

    while True:
        try:
            nm.scan(hosts=subnet, arguments='-sn')
            for host in nm.all_hosts():
                if host == target_ip:
                    print(f"[NMAP] Receiver with IP {target_ip} is in range.")
                    return
            print("[NMAP] Receiver not found, retrying...")
            time.sleep(config["hyperparameters"]["ping_drone_interval"])
        except Exception as e:
            print(f"[NMAP] Error: {e}")
            time.sleep(5)
            

# === Read GPS from NEO-6M ===
def get_gps_coordinates():
    """
    Reads the latest GPS coordinates from a NEO-6M module connected via serial.

    Returns:
        tuple: (latitude, longitude) as floats, or (None, None) if an error occurs.
    """
    try:
        gps_serial = serial.Serial("/dev/ttyAMA0", baudrate=9600, timeout=1)
        while True:
            line = gps_serial.readline().decode('ascii', errors='replace')
            if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                msg = pynmea2.parse(line)
                return msg.latitude, msg.longitude
    except Exception as e:
        print(f"GPS error: {e}")
        return None, None
        

# === Function to run send logic ===
def send_data():
    """
    Main function that collects classification results, GPS coordinates, and captured image,
    then sends them to the receiver node via a TCP socket.

    Steps:
    - Waits until the receiver node is online.
    - Retrieves current GPS coordinates.
    - Prepares the result json.
    - Loads the latest image.
    - Sends all data (results + image) to the receiver.

    Returns:
        None
    """
    wait_for_receiver(RECEIVER_IP)    
    
    latitude, longitude = get_gps_coordinates()

    config = load_config()
    
    data = {
         "trap_id" : config["trap_id"],
         "result": config["latest_classification"]["result"],
         "confidence" : config["latest_classification"]["confidence"],
         "timestamp" : config["latest_classification"]["timestamp"],
         "gps": {
            "latitude": latitude,
            "longitude": longitude
        }
    }

    try:
        json_bytes = json.dumps(data).encode('utf-8')
        json_size = len(json_bytes)

        image_path = config["hyperparameters"]["output_path"]
        image = Image.open(image_path).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        image_bytes = buffer.getvalue()
        image_size = len(image_bytes)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((RECEIVER_IP, PORT))
            s.sendall(json_size.to_bytes(4, byteorder='big'))
            s.sendall(json_bytes)
            s.sendall(image_size.to_bytes(4, byteorder='big'))
            s.sendall(image_bytes)

        print("[SEND] Data sent successfully.")

    except Exception as e:
        print(f"[SEND] Error: {e}")
