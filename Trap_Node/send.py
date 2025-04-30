import socket
import json
import io
from PIL import Image
import serial
import pynmea2
from config_utils import load_config
import time

config = load_config()

trap_id = config["trap_id"]
PORT = config["network"]["PORT"]
SUBNET = config["network"]["SUBNET"]
RECEIVER_IP = config["network"]["HOST"]
expected_message = b"drone_ready"
ACK_MESSAGE = b"ack"


# === Passive Listener ===
def wait_for_drone_broadcast():
    """
    Listens for a broadcast message from the drone indicating it's ready.
    Blocks until the expected message is received.

    Args:
        port (int): The UDP port to listen on.
        expected_message (bytes): The expected broadcast message.

    Returns:
        None
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', PORT))
    print(f"[SEND] Listening for drone broadcast on UDP port {PORT}...")

    while True:
        data, addr = sock.recvfrom(1024)
        if data == expected_message:
            print(f"[SEND] ? Drone broadcast received from {addr[0]}")
            return

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
    - Waits for a drone signal.
    - Retrieves current GPS coordinates.
    - Prepares the result json.
    - Loads the latest image.
    - Sends all data (results + image) to the receiver.

    Returns:
        None
    """
    wait_for_drone_broadcast()

    latitude, longitude = get_gps_coordinates()

    config = load_config()

    data = {
         "trap_id": config["trap_id"],
         "result": config["latest_classification"]["result"],
         "confidence": config["latest_classification"]["confidence"],
         "timestamp": config["latest_classification"]["timestamp"],
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

            ## === Wait for acknowledgment from the drone with retries ===
            ack_received = False
            retries = 3
            for _ in range(retries):
                try:
                    s.settimeout(2)  # Wait for 2 seconds to receive the acknowledgment
                    ack_msg = s.recv(1024)
                    if ack_msg == ACK_MESSAGE:
                        print("[SEND] Acknowledgment received from drone.")
                        ack_received = True
                except socket.timeout:
                    print(f"[SEND] Timeout and no acknowledgment received, sending again...")
                    s.sendall(json_size.to_bytes(4, byteorder='big'))
                    s.sendall(json_bytes)
                    s.sendall(image_size.to_bytes(4, byteorder='big'))
                    s.sendall(image_bytes)  
                    print("[SEND] Data sent successfully.")
                    
            if not ack_received:
                print("[SEND] Failed to receive acknowledgment after 3 retries")
                return False

    except Exception as e:
        print(f"[SEND] Error: {e}")
