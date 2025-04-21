# 🐛 Red Palm Weevil Detection - Cropping YOLOv8 Inference Results
##colab link : https://colab.research.google.com/drive/1hULgYs3y9MFe5V9gGf6IPp3l9f7Fa8py?usp=sharing
This project demonstrates how to use a custom-trained **YOLOv8 model** (exported from Roboflow) to **detect and crop** Red Palm Weevil (RPW) insects from high-resolution images using **Google Colab**.

---

## 📁 1. Google Colab File

The Colab notebook includes:

- 📦 Installing required dependencies (`ultralytics`, `opencv-python`)
- 🔍 Loading a YOLOv8 model (`.pt` file) exported from Roboflow
- 🖼️ Running inference on images stored in a folder
- ✂️ Cropping detected regions and saving them as `160x160` pixel images
- 📤 Downloading processed images

---

## 🚀 2. How to Run the Colab Notebook

### 🔧 Step-by-step Instructions

#### 📌 Step 1: Upload Your Files
- Upload your exported YOLOv8 model (`best.pt`) and your image folder (e.g., `light Phone caam RPW full res`) to the Colab environment.

#### 📌 Step 2: Install Required Packages

```python
!pip install ultralytics opencv-python

