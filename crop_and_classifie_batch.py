import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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
input_folder = "light Phone caam RPW full res"     # Folder of images to process
output_folder = "output_rpw_crops"  # Where to save confident RPW crops
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

# === TRANSFORM ===
def get_transform(model_name):
    size = MODEL_CONFIGS[model_name]["input_size"]
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])

# === MAIN ===
def main():
    yolo_model = YOLO(model_path_yolo)
    classifier = load_model(classification_model_name, classification_model_path, num_classes)
    transform = get_transform(classification_model_name)

    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"[!] Failed to read image: {image_path}")
            continue

        results = yolo_model(image)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        RPW_classification = None

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_crop).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = classifier(input_tensor)
                probs = F.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                confidence = probs[0, 1].item()  # Confidence for RPW (class 1)

            if pred == 1 and confidence > 0.50:
                RPW_classification = confidence
                crop_name = f"{os.path.splitext(filename)[0]}_RPW_{confidence:.2f}.jpg"
                crop_path = os.path.join(output_folder, crop_name)
                cv2.imwrite(crop_path, crop)
                print(f"[✔] {filename} → RPW with {confidence:.2f}, saved: {crop_path}")
                break  # Stop checking boxes for this image

        if RPW_classification is None:
            print(f"[–] {filename} → No RPW > 0.50")

if __name__ == "__main__":
    main()
