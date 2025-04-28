import os
import json
import threading
import socket
from flask import Flask, render_template, send_from_directory, jsonify

# Configuration
HOST = '' #listen on all interfaces
PORT = 12345
SAVE_DIR = './received_data'
JSON_PATH = os.path.join(SAVE_DIR, 'data.json')

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

app = Flask(__name__)

# ---------- Socket Server ----------
def receive_all(sock, size):
    data = b''
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("Connection lost during data receive")
        data += packet
    return data

def start_socket_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"[server] Listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            with conn:
                print(f"[server] Connected by {addr}")
                try:
                    json_size_bytes = receive_all(conn, 4)
                    json_size = int.from_bytes(json_size_bytes, byteorder='big')

                    json_data_bytes = receive_all(conn, json_size)
                    data = json.loads(json_data_bytes.decode('utf-8'))

                    with open(JSON_PATH, "w") as f:
                        json.dump(data, f, indent=2)
                    print(f"[server] JSON saved to {JSON_PATH}")

                    for trap_id in data.get("TRAPS", {}):
                        if data[trap_id].get("collected"):
                            image_size_bytes = receive_all(conn, 4)
                            image_size = int.from_bytes(image_size_bytes, byteorder='big')
                            image_data = receive_all(conn, image_size)

                            image_path = os.path.join(SAVE_DIR, f"{trap_id}.jpg")
                            with open(image_path, 'wb') as img_file:
                                img_file.write(image_data)
                            print(f"[server] Image for trap {trap_id} saved.")

                except Exception as e:
                    print(f"[server] Error: {e}")

# ---------- Flask Web Server ----------
@app.route('/')
def index():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            data = json.load(f)
    else:
        data = {}
    return render_template('index.html', data=data)

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(SAVE_DIR, filename)

@app.route('/data')
def get_data():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            data = json.load(f)
    else:
        data = {}
    return jsonify(data)

# ---------- Entry Point ----------
if __name__ == '__main__':
    threading.Thread(target=start_socket_server, daemon=True).start()
    app.run(host='0.0.0.0', port=12344)
