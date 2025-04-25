import json
import socket
from PIL import Image
import json_utils 
import io

# ==== Config ====
HOST = '192.168.4.1' # The drone pi ip address
PORT = 12345 # must be the same port that the trap sends to

def collect_data_from_trap():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"[collect] Listening on {HOST}:{PORT}")

        conn, addr = s.accept()
        with conn:
            print(f"[collect] conn is {conn}, type: {type(conn)}")

            # First receive json
            json_bytes = conn.recv(4)
            json_size = int.from_bytes(json_bytes, byteorder='big')
            print(f"[collect] JSON SIZE : {json_size} bytes")

            received = b''
            while len(received) < json_size:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                received += chunk

            # Only keep the JSON part
            json_part = received[:json_size]
            leftover = received[json_size:]  # This might already contain start of the image

            json_data = json_part.decode('utf-8')
            parsed = json.loads(json_data)
            print("[collect] Parsed JSON:")
            print(json.dumps(parsed, indent=4))

            # Save JSON
            if not(json_utils.save_result(parsed)):
                return
            # Now receive the next 4 bytes (image size)
            # Use leftover buffer first
            while len(leftover) < 4:
                leftover += conn.recv(4 - len(leftover))
            size_bytes = leftover[:4]
            leftover = leftover[4:]

            img_size = int.from_bytes(size_bytes, byteorder='big')
            print(f"[collect] Image size {img_size} bytes")

            # Now receive the rest of the image
            while len(leftover) < img_size:
                leftover += conn.recv(4096)

            image_data = leftover[:img_size]
            image = Image.open(io.BytesIO(image_data))

            # save the image with the corresponding trap id
            if not(json_utils.save_image(image, parsed["trap_id"])):
                return

            return parsed
