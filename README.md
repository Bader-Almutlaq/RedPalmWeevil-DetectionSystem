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
    - [4. Smart Trap Detection Procedure](#4-smart-trap-detection-procedure)
    - [5. Communication Design](#5-communication-design)
  - [Technologies Used](#technologies-used)
  - [Installation \& Setup](#installation--setup)
    - [Prerequisites](#prerequisites)
    - [Deployment Steps](#deployment-steps)
  - [Expected Impact](#expected-impact)
  - [Future Enhancements](#future-enhancements)

## Team Members
- **Bader A. AlMutlaq**
- **Abdullah A. AlShalan**
- **Yazeed K. AlSayyari**
- **Ibrahim A. AlObaid**


**Supervisor:** Dr. Adel Soudani (King Saud University)

## Project Overview
This project aims to develop an IoT-based system for the early detection and localization of red palm weevils in palm trees. The system utilizes a trap equipped with pheromones to attract weevils, a camera to capture images, and a drone to collect and transfer data for analysis. By leveraging AI and machine learning techniques, the system enhances early detection and response, addressing the challenges associated with manual inspection in high-temperature environments.


## Features
- **Automated Weevil Detection:** Uses AI-based image processing to detect red palm weevils.
- **IoT-Enabled Traps:** Equipped with cameras and pheromone-based attractants.
- **Drone-Based Data Collection:** Transfers data from traps to a central processing unit.
- **Real-Time Monitoring:** Provides instant alerts and geolocation of detected infestations.
- **Scalability:** Can be expanded to cover large palm plantations.

## System Design
This section details the design steps of the proposed IoMT system for early RPW detection. The design ensures that the system effectively detects RPWs, communicates the results, and retrieves them back to the server.
### 1. Hardware Design
- **Trap Structure:**  
  The trap uses a traditional bucket design containing pheromones to attract RPWs. Plastic poles attach to the top edge of the bucket and connect to a platform that supports the ESP-32 CAM. The camera is positioned downward to capture the full interior of the bucket, ensuring optimal light capture and comprehensive imaging.
  
- **Localization & Power:**  
  A GPS NEO-6M module is connected to the ESP-32 CAM to provide precise location data, while a rechargeable battery powers the trap.
  
- **Drone Integration:**  
  The drone is equipped with an ESP-32 module to communicate with the ESP-32 CAM via the WiFi library, ensuring reliable communication regardless of the drone model.

### 2. Dataset Preparation
Acquiring a high-quality and diverse dataset is essential for training an accurate RPW detection model.

#### Data Collection
- **Images from the Web and External Datasets:**  
  Gather images of RPWs and other insects from various online sources and established datasets. This approach ensures a broad variety of images, though external images might differ from those captured in the trap environment.

#### Dataset Filtering and Augmentation
- **Preprocessing:**  
  Enhance images by removing noise and correcting lighting inconsistencies (e.g., via histogram adjustments).
  
- **Augmentation:**  
  Apply techniques such as rotation, flipping, and scaling to diversify the dataset further. All images are scaled to match the ESP-32’s resolution.

### 3. Model Development and Training
The machine learning model is developed and trained using Edge Impulse (EI) through the following stages:

1. **Upload the Dataset to EI:**  
   All gathered images are uploaded to the EI platform.
   
2. **Labeling:**  
   Using EI’s graphical interface, images are annotated with appropriate labels (e.g., RPW detected or not).
   
3. **Model Training:**  
   A suitable ML architecture—such as Convolutional Neural Networks (CNNs)—is selected, and hyperparameters are configured. Training is performed on EI’s cloud infrastructure with GPU acceleration.
   
4. **Exporting the Model:**  
   The trained model is exported in a format optimized for edge devices (e.g., TensorFlow Lite) for deployment on the ESP-32 CAM.

### 4. Smart Trap Detection Procedure
The smart trap follows a systematic process to detect RPW presence and communicate findings:

1. **Capturing the Image:**  
   The trap captures an image of its interior, which is stored locally.
   
2. **Image Classification:**  
   The stored image is processed using the deployed EI ML model. The model returns a classification confidence value (ranging from 0 to 1) that is logged in a data block.
   
3. **Recording GPS Location:**  
   The trap records its exact GPS coordinates using the integrated GPS module.
   
4. **Drone Communication:**  
   - **Signal Check:** If the drone’s signal is not received, the trap waits for a predefined timeout period before repeating the detection process.
   - **On Signal Reception:** The trap transmits the classification result and GPS data to the drone.


#### Evaluation Process
- **Test Dataset:**  
  A labeled dataset (including both RPW-positive and RPW-negative images) is prepared.
  
- **System Execution:**  
  The system processes the test dataset, recording outcomes as true positives (TP), true negatives (TN), false positives (FP), and false negatives (FN).
  
- **Metric Computation and Analysis:**  
  Metrics are computed to assess system performance, guiding any necessary model refinements.

### 5. Communication Design
Data communication is achieved through a two-step process:

#### Smart Trap to Drone
- The trap sends a JSON packet containing the classification result (with confidence level), GPS coordinates, and any additional sensor data (e.g., temperature, humidity) via the WiFi protocol.

#### Drone to Server
- The drone returns to the server and transmits the gathered data using WiFi or Bluetooth protocols. The server then processes this information for further analysis and decision-making.

#### Communication Packet Structure
- **Smart Trap to Drone Packet:**  
  - **Packet ID:** Unique identifier  
  - **Time Stamp:** Time of transmission  
  - **Classification Result:** Confidence level of RPW detection  
  - **Location Snapshot:** GPS coordinates  
  - **Sensor Data:** Optional additional data

- **Drone to Server Packet:**  
  - **Drone ID:** Unique identifier for the drone  
  - **Drone Status:** Battery level and signal strength  
  - _Includes all fields from the Trap-to-Drone packet._

## Technologies Used
- **Hardware:** ESP-32 CAM, GPS NEO-6M, Drone, Sensors
- **Software:** Python, OpenCV, Edge Impulse

## Installation & Setup
### Prerequisites
- ESP-32 CAM with camera module
- GPS module (NEO-6M) for localization
- Drone equipped with an ESP-32 for communication
- Python 3.x installed

### Deployment Steps
1. **Set up the IoT trap** with the camera and sensors.
2. **Configure the drone** for data collection.
3. **Deploy the AI model** on edge device.
4. **Run the monitoring script**


## Expected Impact
- **Improved Early Detection:** Reduces reliance on manual inspection.
- **Cost Efficiency:** Lowers operational costs compared to traditional methods.
- **Scalability:** Enables large-scale monitoring of palm plantations.

## Future Enhancements
- Integration with satellite imaging for broader detection.
- Development of a mobile application for real-time monitoring.
- Enhanced AI model for higher accuracy.


