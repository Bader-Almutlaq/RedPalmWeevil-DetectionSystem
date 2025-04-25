import threading
import time
import classify
import send
from config_utils import load_config

config = load_config()

inference_interval = config["hyperparameters"]["inference_interval"]
running = config["hyperparameters"]["running"]  # shared flag to control both threads

def classification_loop():
    while running:
        classify.run_inference() # for one image do capture -> classify -> save
        time.sleep(inference_interval)

def send_loop():
    while running:
        send.send_data() # Wait until the receiver (drone pi) comes in range
                         # then prepare and send data

if __name__ == "__main__":
    try:
        # Clear the log file on every run (i.e., every boot)
        open("/home/RPW_Project/logs.txt", "w").close()
        print("[Main] Script strated")

        # Start both threads
        t1 = threading.Thread(target=classification_loop)
        t2 = threading.Thread(target=send_loop)

        t1.start()
        t2.start()

        # Keep main thread alive while others run
        t1.join()
        t2.join()

    except KeyboardInterrupt:
        print("\n[MAIN] Ctrl+C detected. Shutting down...")
        running = False
        t1.join()
        t2.join()
        print("[MAIN] Clean exit.")
