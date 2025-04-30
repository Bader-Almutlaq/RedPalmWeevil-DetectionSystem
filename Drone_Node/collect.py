import json
import socket
from PIL import Image
import json_utils 
import io
import time

# ==== Config ====
BROADCAST_IP = '192.168.4.255'  # the broadcast IP of the drone's hotspot
PORT = 12345  # must be the same port that the traps listen on
HOST = ''     # listen on all interfaces
ACK_MESSAGE = b"ack"

def wait_for_trap_with_broadcast():
    """
    Repeatedly broadcasts 'drone_ready' until a trap connects.
    Returns the socket connection once a trap connects.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_sock:
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.bind((HOST, PORT))
        server_sock.listen(1)
        print(f"[collect] Listening on {HOST}:{PORT}")

        # Setup UDP broadcast socket
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        udp_sock.settimeout(2)

        while True:
            try:
                udp_sock.sendto(b"drone_ready", (BROADCAST_IP, PORT))
                print(f"[broadcast] Sent 'drone_ready' to {BROADCAST_IP}:{PORT}")
            except Exception as e:
                print(f"[broadcast] Error: {e}")

            # Wait up to 2 seconds for a TCP connection
            server_sock.settimeout(2)
            try:
                conn, addr = server_sock.accept()
                print(f"[collect] Trap connected from {addr}")
                return conn  # exit loop and return connection
            except socket.timeout:
                continue  # No connection yet, send another broadcast

def collect_data_from_trap():
    conn = wait_for_trap_with_broadcast()
    with conn:
        print(f"[collect] conn is {conn}, type: {type(conn)}")

        # First receive JSON
        json_bytes = conn.recv(4)
        json_size = int.from_bytes(json_bytes, byteorder='big')
        print(f"[collect] JSON SIZE : {json_size} bytes")

        received = b''
        while len(received) < json_size:
            chunk = conn.recv(4096)
            if not chunk:
                break
            received += chunk

        # Extract JSON part
        json_part = received[:json_size]
        leftover = received[json_size:]

        json_data = json_part.decode('utf-8')
        parsed = json.loads(json_data)
        print("[collect] Parsed JSON:")
        print(json.dumps(parsed, indent=4))

        # Save JSON
        if not(json_utils.save_result(parsed)):
            return

        # Receive image size
        while len(leftover) < 4:
            leftover += conn.recv(4 - len(leftover))
        size_bytes = leftover[:4]
        leftover = leftover[4:]
        img_size = int.from_bytes(size_bytes, byteorder='big')
        print(f"[collect] Image size {img_size} bytes")

        # Receive image data
        while len(leftover) < img_size:
            leftover += conn.recv(4096)

        image_data = leftover[:img_size]
        image = Image.open(io.BytesIO(image_data))

        # Save image
        if not(json_utils.save_image(image, parsed["trap_id"])):
            return

        # === Send ACK back to trap ===
        try:
            conn.sendall(ACK_MESSAGE)
            print(f"[ack] Sent acknowledgment to trap.")
        except Exception as e:
            print(f"[ack] Failed to send acknowledgment: {e}")

        return parsed