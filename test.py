import torch
import cv2
import os
import time as t
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ======= Step 1: config =======
model_path = "your_model.pth"
model_path_yolo = "best.pt"
output_folder = "./positive"
# setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loading the model
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# ======= Step 2: Define image transformation (adjust size if needed) =======
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ======= Step 3: Predict a single image (PIL) =======
def predict_from_pil(model, pil_image, transform, device="cpu"):
    image = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()


# ======= Step 4: Predict a batch of images (list of PIL) =======
def predict_batch(model, pil_images, transform, device="cpu"):
    tensors = [transform(img) for img in pil_images]
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        outputs = model(batch)
        _, preds = torch.max(outputs, 1)
    return preds.tolist()


# === MAIN ===
def main():
    yolo_model = YOLO(model_path_yolo)

    while True:
        t.sleep(10)
        image = take_img()

        if image is None:
            print(f"[!] Failed to capture image")
            continue

        results = yolo_model(image)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        cropped_images = []
        input_images = []

        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            if crop.size != 0:
                cropped_images.append(crop)
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_crop).unsqueeze(0).to(device)
                input_images.append(input_tensor)
        pred = predict_batch(model=model, pil_images=input_images, transform=transform)
        for idx, pre in enumerate(pred):
            if pre > 0.50:
                crop_name = f"positive{idx}.jpg"
                crop_path = os.path.join(output_folder, crop_name)
                cv2.imwrite(crop_path, crop)


if __name__ == "__main__":
    main()
