# Red Palm Weevil Detection - Trap Node

## Purpose

The "Trap Node" folder contains the code necessary for setting up the Raspberry Pi-based trap node for the detection and localization of the Red Palm Weevil (RPW). This system uses image classification to detect RPW from captured images and send the results to a central server for further analysis.

## Files

- **`main.py`**: The main script that coordinates the image capture and classification process.
- **`config.json`**: Configuration file for setting parameters such as model paths and server details.
- **`classify.py`**: Contains the image classification logic for detecting RPW.
- **`send.py`**: Handles the communication between the Raspberry Pi trap node and the Flask server for sending results.
- **`config_utils.py`**: Utility functions for loading and handling configuration data.
- **`logs.txt`**: Log file to capture runtime information.

## Installation & Setup

1. **Download Raspberry Pi OS**:
   - Use the [Raspberry Pi Imager](https://www.raspberrypi.org/software/) to download and install Raspberry Pi OS onto an SD card.
   - Select The Raspberry Pi OS for the most stable experience.

2. **Set up Raspberry Pi**:
   - Insert the SD card and boot your Raspberry Pi.
   - Connect to Wi-Fi and **ensure that both raspberry pi's are connected to the same network**.

3. **Install Dependencies**:
   - Open a terminal and run:
     ```bash
     sudo apt-get update
     sudo apt-get install python3-pip
     pip3 install -r requirements.txt
     ```

4. **A: Clone Repository**:
   - Clone the GitHub repository to your Raspberry Pi:
     ```bash
     git clone https://github.com/Bader-Almutlaq/RedPalmWeevil-DetectionSystem.git
     cd RedPalmWeevil-DetectionSystem/Trap_Node
     ```

    - Or you can manually install the `Trap_Node` folder

5. **Configure the System**:
   - Edit **`config.json`** to set your model paths, and network details.

6. **Run the System**:
   - Start the process with:
     ```bash
     python3 main.py
     ```

### Optional: Running Heedlessly with Crontab

To run the script automatically after booting (headless), use `cron`:

1. Open the crontab file:
   ```bash
   crontab -e
   ```

2. It will ask for the preferred text editor pick any text editor you like or enter `1` for `nano` text editor

3. then, find the full path to Python and your `main.py` file:
   - To get the Python path, run:
     ```bash
     which python3
     ```
   - To get the full path to your script, run:
     ```bash
     realpath main.py
     ```

4. Add the following line to the crontab file, replacing placeholders with your actual paths:
   ```bash
   @reboot "insert full path to python3" "insert full path to main.py" >> "insert full path to logs.txt" > 2>&1
   ```

5. Now on boot the script will run and the outputs will be written to **`logs.txt`** for monitoring and troubleshooting.

## Helpful Notes

- Ensure your Raspberry Pi is connected to a camera module for image capture.
- The system assumes a Flask server is running on another Raspberry Pi for receiving results.
- For continuous operation, you can set the scripts to run as background processes, just follow the steps given [here](#optional-running-heedlessly-with-crontab)


