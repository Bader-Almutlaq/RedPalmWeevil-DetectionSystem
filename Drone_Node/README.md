# Drone Node – Red Palm Weevil Detection System

## Purpose

This component runs on a drone-mounted Raspberry Pi and is responsible for hosting local network, listening to incoming results from traps, and sending the gathered classification results to the central server.

## Files

- **`main.py`**: The main script that coordinates the `collect.py` and `upload.py` scripts.
- **`collect.json`**: Shared json that saves the collected results, and loads the collected results to the server.
- **`json_utils.py`**: Contains the functions to `save` and `load` to the json, and maintains the critical sections using thread locks.
- **`collect.py`**: Listen to incoming results from traps, then saves them in `collect.json`.
- **`upload.py`**: Uploads the collected results from traps to the server.
- **`logs.txt`**: Log file to capture runtime information.

## Installation & Setup

1. **Download Raspberry Pi OS**:
   - Use the [Raspberry Pi Imager](https://www.raspberrypi.org/software/) to install Raspberry Pi OS to your SD card.

2. **Set Up Raspberry Pi**:
   - Boot your Raspberry Pi with the SD card.

3. **Set Up Loacl Network (Access Point)**:
   - Thanks to a post I found on [the Raspberry Pi forums](https://forums.raspberrypi.com/), I was able to get the Access Point to work properly. These commands are straight from that thread, make sure to replace my-password with your password — all credit to the original poster:

    ```bash
    nmcli con delete TEST-AP
    nmcli con add type wifi ifname wlan0 mode ap con-name TEST-AP ssid hotspot autoconnect false
    nmcli con modify TEST-AP wifi.band bg
    nmcli con modify TEST-AP wifi.channel 3
    nmcli con modify TEST-AP wifi.cloned-mac-address 00:12:34:56:78:9a
    nmcli con modify TEST-AP wifi-sec.key-mgmt wpa-psk
    nmcli con modify TEST-AP wifi-sec.psk "my-password"
    nmcli con modify TEST-AP ipv4.method shared ipv4.address 192.168.4.1/24
    nmcli con modify TEST-AP ipv6.method disabled
    nmcli con up TEST-AP
    ```

4. **Install System Packages & Dependencies**:
   - All the Packages used are included in the raspberry pi os except for pillow, to install it run:
   ```bash
   sudo apt-get install python3-pil
   ```

5. **Clone Repository**:
   ```bash
   git clone https://github.com/Bader-Almutlaq/RedPalmWeevil-DetectionSystem.git
   cd RedPalmWeevil-DetectionSystem/Drone_Node
   ```

6. **Configure the System**:
   - Edit each file according to your set up, probably in most cases you will only change the IP addresses.
   - To view the IP addrees on raspberry pi run:
   ```bash
   hostname -I
   ```

7. **Run the System**:
   ```bash
   python3 main.py
   ```

---

## Optional: Running Headlessly with Cron

To make sure the drone node runs automatically on boot:

1. Open crontab:
   ```bash
   crontab -e
   ```

2. Find paths:
   ```bash
   which python3       # to get the Python path
   realpath main.py    # to get the script's full path
   ```

3. Add to crontab:
   ```bash
   @reboot <insert full path to python3> <insert full path to main.py>
   ```
