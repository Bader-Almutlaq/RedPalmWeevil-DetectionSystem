# =============================================================================
#  input  : Calls `take_image()` function to simulate capturing an image 
#            (currently loads a local file using OpenCV).
#
#  output : Processes the image using YOLOv8 to detect objects (e.g. RPW), 
#            classifies the cropped regions using a custom EfficientNet-B4 model,
#            annotates the image with bounding boxes and confidence scores, 
#            and saves the final result as a timestamped image in the 'positive' folder.
#            The format of the output file is: classified_YYYYMMDD_HHMMSS.jpg
#
#  logic  : - Uses YOLOv8 for object detection to locate potential RPW regions.
#            - Crops each detected box and passes it through a classification model 
#              to determine if it’s RPW (Red Palm Weevil) or NRPW (Not RPW).
#            - Draws red boxes for RPW, yellow for NRPW, and prints confidence scores.
#            - Tracks and returns the highest confidence score of RPW detections.
#
#  requirements : 
#       1. Install the required libraries:
#          - torch
#          - torchvision
#          - ultralytics
#          - opencv-python
#          - pillow
#       2. Ensure `best.pt` (YOLOv8 model) and `efficientnet_b4_rpw.pth` 
#          (classification model) are downloaded and available in the working directory.
#
#  folders:
#       - The 'positive' folder will be created (if not existing) to store outputs.
#
#  Notes:
#       - You can modify `take_image()` to integrate with real-time camera capture.
#       - Classification supports different models (MobileNet, EfficientNet-B0/B4).
# =============================================================================


import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from datetime import datetime
from torchvision import transforms
from ultralytics import YOLO
from torchvision.models import (
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
    efficientnet_b0, EfficientNet_B0_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
)

# === MODEL CONFIGS ===
MODEL_CONFIGS = {
    "mobilenet": {
        "builder": mobilenet_v3_large,
        "weights": MobileNet_V3_Large_Weights.DEFAULT,
        "classifier_index": 0,
        "input_size": 224,
    },
    "efficientnet_b0": {
        "builder": efficientnet_b0,
        "weights": EfficientNet_B0_Weights.DEFAULT,
        "classifier_index": 1,
        "input_size": 224,
    },
    "efficientnet_b4": {
        "builder": efficientnet_b4,
        "weights": EfficientNet_B4_Weights.DEFAULT,
        "classifier_index": 1,
        "input_size": 380,
    },
}

# === CONFIGURATION ===
model_path_yolo = "best.pt"
classification_model_name = "efficientnet_b4"
classification_model_path = "efficientnet_b4_rpw.pth"
num_classes = 2
input_image_path = "20250420_135107.jpg"
output_folder = f"positive"
os.makedirs(output_folder, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === LOAD CLASSIFICATION MODEL ===
def load_model(model_name, path, num_classes=2):
    config = MODEL_CONFIGS[model_name]
    model = config["builder"](weights=None)
    in_features = model.classifier[config["classifier_index"]].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model.to(device)

# === Image Transform for Classification ===
def get_transform(model_name):
    size = MODEL_CONFIGS[model_name]["input_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    
def take_image():
    #implemnt taking image logic here for now 
    return cv2.imread(input_image_path)

# === Main Processing ===
def main():
    yolo_model = YOLO(model_path_yolo)
    classifier = load_model(classification_model_name, classification_model_path, num_classes)
    transform = get_transform(classification_model_name)

    
    image = take_image()
    
    
    if image is None:
        print(f"[!] Failed to read image: {input_image_path}")
        return

    results = yolo_model(image)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    image_draw = image.copy()

    maxvalue_confidence = 0.0

    for idx, (box, conf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = map(int, box[:4])
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            print(f"[!] Empty crop at box {idx}")
            continue

        pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        input_tensor = transform(pil_crop).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = classifier(input_tensor)
            pred = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][pred].item()

        label = "RPW" if pred == 1 else "NRPW"
        color = (0, 0, 255) if label == "RPW" else (0, 255, 255)

        if label == "RPW" and confidence > maxvalue_confidence:
            maxvalue_confidence = confidence

        cv2.rectangle(image_draw, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image_draw, label + f" {confidence:.2f}", (x1 + 5, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 3)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"classified_{timestamp}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    
    cv2.imwrite(output_path, image_draw)

    print(f"[✓] Processed and saved: LastImage.jpg")
    print(f"Max RPW confidence: {maxvalue_confidence:.4f}")
    return maxvalue_confidence

if __name__ == "__main__":
    main()
