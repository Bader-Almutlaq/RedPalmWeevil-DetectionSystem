import torch
import cv2
import os
from datetime import datetime
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ======= Step 1: Configurations =======
model_path = "saved_full_models/efficientnet_b4_rpw.pth"
model_path_yolo = "saved_full_models/yolo.pt"
output_folder = "./positive"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
except Exception as e:
    print(f"[!] Error loading classification model: {e}")
    model = None  # Prevent crash

transform = transforms.Compose(
    [
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(pil_image, model=model, transform=transform, device=device):
    try:
        with torch.no_grad():
            output = model(pil_image)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        return predicted.item(), confidence.item()
    except Exception as e:
        print(f"[!] Prediction error: {e}")
        return -1, 0.0


def draw(image_draw, label, confidence, point_1, point_2, color):
    try:
        cv2.rectangle(image_draw, point_1, point_2, color, 2)
        cv2.putText(
            image_draw,
            label + f" {confidence:.2f}",
            (point_1[0] + 5, point_1[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            color,
            3,
        )
    except Exception as e:
        print(f"[!] Drawing error: {e}")
    return image_draw


def save_image(image):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"classified_{timestamp}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"[!] Failed to save image: {e}")


def main():
    if model is None:
        print("[!] Model not loaded. Skipping...")
        return False

    try:
        yolo_model = YOLO(model_path_yolo)
    except Exception as e:
        print(f"[!] Error loading YOLO model: {e}")
        return False

    try:
        image = cv2.imread("data/NRPW/NRPW-6.jpg")
        if image is None:
            raise ValueError("Image is None (possibly missing or unreadable)")
    except Exception as e:
        print(f"[!] Failed to read image: {e}")
        return False

    maxvalue_confidence = 0.0

    try:
        print("=" * 95)
        print("Yolo info:", end="")
        results = yolo_model(image)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
    except Exception as e:
        print(f"[!] YOLO detection error: {e}")
        return False

    print("=" * 95)
    print("Model Results:")

    for idx, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_crop).unsqueeze(0).to(device)

            pred, conf = predict(pil_image=input_tensor)
            label = "RPW" if pred == 1 else "NRPW"
            color = (0, 0, 255) if label == "RPW" else (0, 255, 255)

            if label == "RPW" and conf > maxvalue_confidence:
                maxvalue_confidence = conf

            image = draw(
                image_draw=image,
                label=label,
                confidence=conf,
                point_1=(x1, y1),
                point_2=(x2, y2),
                color=color,
            )
        except Exception as e:
            print(f"[!] Error processing box {idx}: {e}")
            continue

    save_image(image)
    print(f"Label: {label}")
    print(f"Max RPW confidence: {maxvalue_confidence:.4f}")
    print("=" * 95)

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("[!] Script failed but exited cleanly.")
