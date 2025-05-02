# IoT-Based Red Palm Weevil Detection System

## Table of Contents
- [IoT-Based Red Palm Weevil Detection System](#iot-based-red-palm-weevil-detection-system)
  - [Table of Contents](#table-of-contents)
  - [Team Members](#team-members)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [System Design](#system-design)
    - [1. Hardware Design](#1-hardware-design)
    - [2. Dataset Preparation](#2-dataset-preparation)
    - [3. Model Development and Training](#3-model-development-and-training)
    - [4. Classification Pipeline](#4-classification-pipeline)
    - [5. Communication Design](#5-communication-design)
  - [Results](#results)
    - [Inference Time Analysis](#inference-time-analysis)
    - [RAM Consumption Analysis](#ram-consumption-analysis)
    - [Computational Load (FLOPs)](#computational-load-flops)
    - [Communication Time and Delay](#communication-time-and-delay)
  - [Technologies Used](#technologies-used)
    - [Hardware](#hardware)
    - [Software](#software)
    - [Libraries \& Dependencies](#libraries--dependencies)
- [Repository Structure](#repository-structure)
  - [Installation \& Setup](#installation--setup)
    - [Prerequisites](#prerequisites)
    - [Deployment Steps](#deployment-steps)

## Team Members
- **Bader A. AlMutlaq**
- **Abdullah A. AlShalan**
- **Yazeed K. AlSayyari**
- **Ibrahim A. AlObaid**


**Supervisor:** Dr. Adel Soudani (King Saud University)

## Project Overview
This project aims to develop an IoT-based system for the early detection and localization of red palm weevils in palm trees. The system utilizes a trap equipped with pheromones to attract weevils, a camera to capture images and locally process them using a microcontroller, and a drone to collect and transfer data for analysis. By leveraging AI and machine learning techniques, the system enhances early detection and response, addressing the challenges associated with manual inspection in high-temperature environments.


## Features
- **Automated Weevil Detection:** Uses AI-based image processing to detect red palm weevils.
- **IoT-Enabled Traps:** Equipped with cameras and pheromone-based attractants.
- **Drone-Based Data Collection:** Transfers data from traps to a central processing unit.
- **Real-Time Monitoring:** Provides instant alerts and geolocation of detected infestations.
- **Scalability:** Can be expanded to cover large palm plantations.

## System Design
This section details the design steps of the proposed IoMT system for early RPW detection. The design ensures that the system effectively detects RPWs, communicates the results, and retrieves them back to the server.

### 1. Hardware Design
The proposed hardware design integrates traditional trapping methods with modern IoT components to enable autonomous red palm weevil detection and geolocation.

- **Trap Structure:**  
  The trap is based on a conventional bucket design commonly used in RPW management. It contains pheromones that attract RPWs into the bucket. The trap’s lid functions as the mounting platform for all electronic components. A Raspberry Pi 5 is fixed to the lid, with a downward-facing camera installed to capture high-resolution images of the bucket's interior. To ensure consistent and adequate lighting for image capture, an LED strip is mounted around the camera module.

- **Localization and Power Supply:**  
  For precise geolocation, a GPS NEO-6M module is interfaced with the Raspberry Pi. This module enables accurate mapping of trap locations during data reporting. The entire system is powered by a rechargeable battery, ensuring autonomous field operation for extended periods without human intervention.

- **Drone Integration:**  
  The drone used for data collection is also equipped with a Raspberry Pi. This secondary Raspberry Pi establishes a Wi-Fi-based communication link with each trap using Python’s `wifi` libraries. The use of standard Raspberry Pi modules on both ends ensures compatibility and seamless integration, regardless of the drone model. The drone’s Pi collects processed image data, classification results, and GPS coordinates from each trap during its scheduled flight.

---

### 2. Dataset Preparation

A well-curated and diverse dataset is critical for training an effective red palm weevil (RPW) detection model. To ensure the model performs well in real-world conditions, we adopted a two-part strategy: collecting images from online sources and simulating real trap environments.

#### Data Collection

We divided our data collection into two primary sources:

- **1. Online Dataset:**  
  We sourced **500 RPW images** and **900 images of other insects** (non-RPW, or NRPW) from the web. Although this approach provided a starting point, the images often included noise and background elements unrelated to the trap environment, limiting their representativeness. The limited quantity also posed a challenge for training deep models, which motivated us to apply extensive data augmentation.

- **2. Simulated Trap Dataset:**  
  Due to the unavailability of live RPWs and limited cooperation from local farms, we created a simulated trap dataset. We printed **200 RPW** and **200 NRPW** images and placed them inside the actual trap to capture images under controlled, realistic conditions. Each printed insect image was photographed multiple times (~4x) under different angles and lighting conditions, resulting in approximately **800 raw images**.


#### Dataset Augmentation

To enrich the dataset and address class imbalance, we applied a set of data augmentation techniques tailored for each subset:

- **Online Dataset Augmentation:**  
  We performed 10 augmentations on the RPW and NRPW images, including:
  - Horizontal and vertical flipping  
  - Hue adjustment  
  - Sharpening  
  - Brightness enhancement and reduction  
  - Rotations at ±30° and ±60°

  This resulted in an expanded dataset of:
  - **5000 RPW images**  
  - **9000 NRPW images**

- **Trap Dataset Augmentation:**  
  Augmentation was applied differently to RPW and NRPW images to balance the dataset:
  - **RPW:** Vertical and horizontal flips, hue adjustment, sharpening, brightness changes, ±30°, and ±60° rotations  
    → Resulting in **5288 images**  
  - **NRPW:** Same as above but with ±45° rotations instead  
    → Resulting in **4260 images**
#### Final Dataset Summary

After preprocessing and augmentation, the final dataset consists of:

- **RPW Images:** 10288  
- **NRPW Images:** 13260  
- **Total Images:** 23548  

This dataset offers significant diversity in terms of lighting, positioning, and backgrounds, which enhances the model’s ability to generalize and detect RPWs accurately in varying real-world conditions.

---

### 3. Model Development and Training

The machine learning pipeline for RPW detection consists of two core stages: **object detection** and **image classification**. Each stage was developed, trained, and evaluated separately using specialized tools and datasets.


#### Object Detection

To localize insect instances within trap images, we implemented a **YOLOv8-based object detection** pipeline.

- **Training Setup:**
  - **Dataset:** 1,900 high-resolution images annotated with bounding boxes using Roboflow.
  - **Augmentation Techniques:** Horizontal/vertical flips, ±90° rotations, ±30% brightness/contrast adjustments, Gaussian blur (3×3 kernel), and cutout (20% mask ratio).
  - **Final Dataset Size:** 3,800 images (after augmentation).
  - **Model:** YOLOv8 (Ultralytics).
  - **Environment:** Trained for 30 epochs on Google Colab using a T4 GPU, with a batch size of 16.

- **Evaluation Results:**

  | Metric          | Value |
  | --------------- | ----- |
  | Precision       | 0.982 |
  | Recall          | 0.956 |
  | mAP@0.50        | 0.980 |
  | mAP@0.50:0.95   | 0.654 |
  | Validation Loss | 2.072 |

  Training trends revealed improving detection fidelity over time, with precision rising from 0.882 to 0.982 and recall stabilizing between 0.94 and 0.97. The model showed consistent performance gains in detecting and localizing insects across varying IoU thresholds.


#### Image Classification

For object classification, multiple deep learning architectures were fine-tuned using transfer learning.

- **Models Evaluated:**  
  - EfficientNetB0, B2, B3, B4  
  - MobileNetV3

- **Training Details:**
  - **Initial Dataset:** 14,000 web-collected images (70% train / 15% val / 15% test).
  - **Transfer Learning:** 50–80% of model layers frozen.
  - **Learning Rate:** 0.005 with early stopping (patience = 3).
  - **Training Platform:** Google Colab with limited GPU.
  - **Epochs:** 10 (fine-tuning converged quickly).

- **Initial Evaluation (Web Images):**

  | Model              | Accuracy  | Precision | Recall    | F1 Score  |
  | ------------------ | --------- | --------- | --------- | --------- |
  | MobileNetV3        | 0.847     | 0.960     | 0.756     | 0.846     |
  | EfficientNetB0     | 0.920     | 0.935     | 0.919     | 0.927     |
  | EfficientNetB2     | 0.860     | 0.821     | 0.956     | 0.883     |
  | **EfficientNetB3** | **0.931** | **0.933** | **0.943** | **0.938** |
  | EfficientNetB4     | 0.898     | 0.983     | 0.829     | 0.900     |

- **Final Evaluation (Full Dataset of 23,548 Images):**

  | Model              | Accuracy  | Precision | Recall    | F1 Score  |
  | ------------------ | --------- | --------- | --------- | --------- |
  | MobileNetV3        | 1.000     | 1.000     | 1.000     | 1.000     |
  | EfficientNetB0     | 0.999     | 0.999     | 1.000     | 0.999     |
  | EfficientNetB2     | 1.000     | 1.000     | 1.000     | 1.000     |
  | **EfficientNetB3** | **1.000** | **1.000** | **1.000** | **1.000** |
  | EfficientNetB4     | 1.000     | 1.000     | 1.000     | 1.000     |

EfficientNetB3 was selected for final deployment due to its strong performance and superior generalization in both initial and final evaluations.

---

### 4. Classification Pipeline

The deployed smart trap follows a systematic pipeline from image capture to data transmission:

1. **Image Capture**  
   - Conducted using a Raspberry Pi installed inside the trap.
   - Captures 5MP images at timed intervals without human intervention.

2. **Object Detection (YOLOv8)**  
   - Captured images are analyzed to detect insects using the trained YOLOv8 model.
   - Output: bounding boxes and coordinates for each detected insect.

3. **Preprocessing**  
   - Detected regions are cropped using bounding box coordinates.
   - Cropped images are resized to 300×300 pixels and normalized.

4. **Classification (EfficientNetB3)**  
   - Each cropped image is passed through the EfficientNetB3 classifier.
   - Outputs: class label and confidence score.

5. **Result Generation**  
   - Full-size image is annotated with class labels and bounding boxes.
   - A metadata file is generated containing classification results, timestamp, and GPS data.

---

### 5. Communication Design

The communication framework is a critical component of the smart trap system, ensuring reliable data transfer between the traps, the drone, and the central server. This communication pipeline is divided into two main stages:

- **Trap-to-Drone Communication**
- **Drone-to-Server Communication**

Both stages utilize the built-in Wi-Fi capabilities of the Raspberry Pi, with the drone acting as a mobile data aggregator and gateway to the server. The overall architecture enables offline data collection with asynchronous server upload, ideal for field deployment.


#### Trap-to-Drone Communication

Each smart trap is configured to store its most recent classification output, including the confidence score and GPS coordinates, in a local buffer. The trap operates in passive mode, continuously listening for an incoming drone signal.

The drone, upon initiating a data collection mission, activates a Wi-Fi hotspot and begins broadcasting a beacon signal every 2 seconds. Once a trap detects the signal and successfully connects to the drone’s hotspot, it transmits the following payload:

- Processed image (with bounding boxes and class labels)
- Classification result and confidence level
- GPS coordinates of the trap

The drone maintains a structured JSON file to manage and track data collection from each trap. A sample file structure is shown below:

```json
{
  "data_collected": false,
  "TRAPS": ["1", "2"],
  "1": {
    "collected": false,
    "results": {}
  },
  "2": {
    "collected": false,
    "results": {}
  }
}
```

- **`data_collected`**: Boolean flag indicating whether any traps were successfully queried.
- **`TRAPS`**: List of all active trap IDs to be polled during the trip.
- **`<trap_id>.collected`**: Tracks whether data from a specific trap has been received.
- **`<trap_id>.results`**: Stores the full result payload including classification, confidence, and GPS data.

This structure enables the drone to track data collection status in real time and facilitates efficient upload to the server once the round is complete.

#### Drone-to-Server Communication

After completing the trap polling process, the drone connects to the central server using a secured Wi-Fi or mobile hotspot link. It transmits the compiled JSON file along with the associated image data.

Upon receipt, the server performs the following actions:

1. **Parses the JSON file** to extract and organize data per trap.
2. **Stores the image and metadata** (e.g., timestamps, classifications, GPS coordinates) in a database.
3. **Updates a web-based dashboard**, which provides users with a real-time view of trap activity and detection results.

This architecture ensures that users can access detection results from remote locations, track weevil infestations geographically, and make timely decisions based on up-to-date data.

## Results

This section presents the performance evaluation of the system across several dimensions: inference time, memory usage, computational load (FLOPs), and communication delay. The results provide insights into system efficiency, resource consumption, and scalability.

### Inference Time Analysis

To evaluate inference efficiency, we measured the total processing time for 10 batches of image crops, each with 10 images. The images varied in insect quantity from 1 to 10. The results showed that inference time generally increased with image complexity:

- **Crop 1:** 370 ms
- **Crop 10:** 1250 ms

Minor fluctuations were observed (e.g., Crop 3 was faster than Crop 2), likely due to system scheduling or data variations. Overall, the trend confirms that processing time scales with insect count, reflecting increased computational load as visual data becomes more complex.

### RAM Consumption Analysis

Memory usage was assessed over 10 batches of images (100 total), with insect counts increasing from 1 to 10 per image. Average RAM usage for core components remained consistent:

- **Classification:** 44.57 MB
- **YOLO Detection:** 11.71 MB
- **Image Processing:** 14.69 MB

Total memory usage varied slightly from **673 MB to 702 MB**, likely due to background system overhead rather than model behavior.

### Computational Load (FLOPs)

The CPU’s computational demand was evaluated using Giga floating-point operations per second (GFLOPs) across the same 10 batches:

- **Total GFLOPs increased** from **7.1** (Crop 1) to **33.8** (Crop 10).
- **YOLO Detection:** Constant at **4.1 GFLOPs** per batch, indicating a fixed cost.
- **Classification:** Grew linearly with input complexity (approximately 3 GFLOPs per crop).

After normalization:
- Classification GFLOPs per crop stayed flat.
- YOLO GFLOPs per crop declined from 4.1 to 0.41, showing it becomes less significant in heavier loads.

### Communication Time and Delay

We tested trap-to-drone communication over 50 trials to determine the efficiency of data collection in the field. Each trial recorded:

- **Receive Time:** Time from trap receiving drone signal to data being saved.
- **Total Time:** Time from signal reception to acknowledgment received.

Results:
- **Average Receive Time:** 0.88 seconds
- **Average Total Time:** 0.99 seconds

These times indicate reliable, low-latency performance suitable for field deployment, and they can be used to optimize drone speed and conserve battery life.

---

## Technologies Used

### Hardware
- **Raspberry Pi 5:** Edge computing and image processing.
- **GPS NEO-6M:** Provides trap geolocation.
- **Camera Module:** Captures images of trap content.
- **Drone:** Performs autonomous data collection.
- **LED Strip:** Ensures consistent lighting for image capture.

### Software
- **Programming Language:** Python 3.x
- **AI & Machine Learning:** PyTorch, torchvision, YOLOv8 (Ultralytics)
- **Computer Vision:** OpenCV, Pillow
- **Web Framework:** Flask
- **Visualization:** Matplotlib, Seaborn
- **Data Communication:** pyserial, pynmea2, python-nmap
- **System Integration:** socket, threading, JSON

### Libraries & Dependencies
- `ultralytics==8.3.115`
- `torch==2.7.0`
- `torchvision==0.22.0`
- `opencv-python==4.11.0.86`
- `Flask==2.2.2`
- `pillow==11.2.1`
- `pyserial==3.5`
- `pynmea2==1.19.0`
- `python-nmap==0.7.1`
- `picamera2==0.3.25`
- `matplotlib`, `seaborn`, `socket`, `threading`, `json`

---
# Repository Structure

This repository contains the necessary code and data to detect and classify red palm weevil (RPW) using an IoT-based system. Below is an overview of the directory structure:
```bash
RedPalmWeevil-DetectionSystem/
├── classify.py                        # Script for classifying the images and performing inference.
├── Models_Training_TrapImages.ipynb    # Jupyter notebook for training the model with trap images.
├── Models_Training_NoTrapImages.ipynb  # Jupyter notebook for training the model without trap images.
├── README.md                          # The main README file for the project.
├── requirements.txt                   # List of dependencies for the project.
├── test/                               # Folder for testing data (ignored in git due to large size).
│   ├── NRPW_crop_augmented/            # Augmented images of NRPW crops used for testing.
│   ├── NRPW_trap/                     # Trap images of NRPW used for testing.
│   ├── RPW_crop_augmented/            # Augmented images of RPW crops used for testing.
│   ├── RPW_trap/                      # Trap images of RPW used for testing.
│   └── temp.png                       # Temporary image used during testing.
├── data/                               # Folder for data (ignored in git due to large size).
│   ├── NRPW/                          # Contains NRPW images for training and inference.
│   └── RPW/                           # Contains RPW images for training and inference.
├── positive/                           # Folder for storing positive images with RPW detections.
│   └── classified_20250423_190533.jpg  # Example of a classified positive image with RPW.
├── saved_full_models/                  # Folder for storing full trained models.
│   ├── efficientnet_b0_rpw.pth         # EfficientNet-B0 model for RPW detection.
│   ├── efficientnet_b2_rpw.pth         # EfficientNet-B2 model for RPW detection.
│   ├── efficientnet_b3_rpw.pth         # EfficientNet-B3 model for RPW detection.
│   ├── efficientnet_b4_rpw.pth         # EfficientNet-B4 model for RPW detection.
│   ├── mobilenetv3_rpw.pth            # MobileNetV3 model for RPW detection.
│   └── yolo.pt                        # YOLO model for RPW detection.
├── trap_node/                          # Code and configuration for the IoT trap node.
│   ├── classify_benchmark.py           # Benchmarking script for classification performance.
│   ├── classify.py                     # Main script for performing classification on trap images.
│   ├── config_utils.py                 # Utility functions for handling configurations.
│   ├── config.json                     # Configuration file for the trap node.
│   ├── logs.txt                        # Log file for the trap node.
│   ├── main.py                         # Main script for trap node operations.
│   ├── models/                         # Folder for models used in the trap node (e.g., trained models).
│   ├── README.md                       # Documentation for the trap node.
│   ├── requirements.txt                # List of dependencies for the trap node.
│   └── send.py                         # Script for sending data from the trap to the server.
├── drone_node/                         # Code and configuration for the drone node.
│   ├── collect.py                      # Script for collecting data from the trap via drone.
│   ├── collected.json                  # JSON file for storing collected data.
│   ├── json_utils.py                   # Utility functions for handling JSON data.
│   ├── logs.txt                        # Log file for the drone node.
│   ├── main.py                         # Main script for drone node operations.
│   ├── README.md                       # Documentation for the drone node.
│   ├── upload.py                       # Script for uploading collected data to the server.
├── preprocess/                         # Folder for preprocessing scripts for data and images.
│   ├── crop_and_classifie_batch.py     # Script for cropping and classifying image batches.
│   ├── data_augmentation.py            # Script for performing data augmentation.
│   ├── image_crop.py                   # Script for cropping images to relevant regions.
│   ├── image_numbering.py              # Script for numbering cropped images.
│   ├── mix_datasets.py                 # Script for mixing datasets for training.
│   ├── scale_image.py                  # Script for scaling images to required dimensions.
│   └── unmix_datasets.py               # Script for unmixing mixed datasets.
├── saved_models_states/                # Folder for storing the best model states during training.
│   ├── efficientnet_b0_best_model.pth  # Best state of EfficientNet-B0 model.
│   ├── efficientnet_b2_best_model.pth  # Best state of EfficientNet-B2 model.
│   ├── efficientnet_b3_best_model.pth  # Best state of EfficientNet-B3 model.
│   ├── efficientnet_b4_best_model.pth  # Best state of EfficientNet-B4 model.
│   └── mobilenet_best_model.pth       # Best state of MobileNet model.
├── server/                             # Code and configuration for the server handling data.
│   ├── received_data/                  # Folder for storing received data on the server.
│   ├── server.py                       # Main server script for handling requests.
│   ├── static/                         # Static files served by the server (e.g., images, CSS).
│   └── templates/                      # HTML templates for the server (e.g., web interface).
```

## Installation & Setup

### Prerequisites
Ensure the following components are available and configured:

- Raspberry Pi 5 with a connected camera module
- GPS NEO-6M module connected to the Pi via serial interface
- Drone equipped with a Raspberry Pi for Wi-Fi communication
- Python 3.x installed with all dependencies (see above)

### Deployment Steps

1. **Install Required Libraries**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up the IoT Trap**
   - Connect and configure the camera module and GPS.
   - Attach the LED strip and test illumination.
   - Load the image capture and classification script on the Raspberry Pi.

3. **Deploy the AI Model**
   - Use the pre-trained YOLOv8 and EfficientNet model in the full_models_states folder or train your own.
   - Load the model using:
     ```python
     from ultralytics import YOLO
     model = YOLO("best.pt")  # Replace with your trained weights
     ```

4. **Configure the Drone**
   - Set the drone’s Raspberry Pi to act as a Wi-Fi hotspot.
   - Load the data collection script to listen for trap connections and receive JSON + image data.


