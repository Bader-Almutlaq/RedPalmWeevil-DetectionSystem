import threading
import time
import collect
import upload

running = True
COLLECT_INTERVAL = 1
UPLOAD_INTERVAL = 1

def collect_loop():
    while running:
        collect.collect_data_from_trap() # take result from one trap and save it
        time.sleep(COLLECT_INTERVAL)

def upload_loop():
    while running:
        upload.main()
        time.sleep(UPLOAD_INTERVAL)

if __name__ == "__main__":
    try:
        # Clear the log file on every run (i.e., every boot)
        open("/home/RPW_Project/Drone_Node/logs.txt", "w").close()
        print("[Main] Script started")

        # Start both threads
        t1 = threading.Thread(target=collect_loop)
        t2 = threading.Thread(target=upload_loop)

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
