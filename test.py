import torch
import cv2
import os
import time as t
from datetime import datetime
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ======= Step 1: configrations =======
model_path = "saved_full_models/efficientnet_b4_rpw.pth"
model_path_yolo = "saved_full_models/yolo.pt"
output_folder = "./positive"
# setting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# loading the model
model = torch.load(model_path, map_location=device, weights_only=False)
model.eval()

# ======= Step 2: Define image transformation=======
transform = transforms.Compose(
    [
        transforms.Resize((380, 380)),  # for EfficientNetB4
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ======= Step 3: Predict a single image (PIL image) =======
def predict(pil_image, model=model, transform=transform, device=device):
    with torch.no_grad():
        output = model(pil_image)
        probabilites = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilites, 1)
        print(predicted)
    return predicted.item(), confidence.item()


def draw(image_draw, label, confidence, point_1, point_2, color):
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
    return image_draw


def save_image(image):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"classified_{timestamp}.jpg"
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, image)


# === MAIN ===
def main():
    yolo_model = YOLO(model_path_yolo)

    while True:
        t.sleep(10)
        # image = take_img()
        image = cv2.imread("data/NRPW/NRPW-6.jpg")
        maxvalue_confidence = 0.0
        if image is None:
            print(f"[!] Failed to capture image")
            continue

        results = yolo_model(image)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        print("=========================")
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            if crop.size != 0:
                pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                input_tensor = transform(pil_crop).unsqueeze(0).to(device)

                pred, conf = predict(pil_image=input_tensor)

                label = "RPW" if pred == 1 else "NRPW"
                color = (0, 0, 255) if label == "RPW" else (0, 255, 255)

                if label == "RPW" and conf > maxvalue_confidence:
                    maxvalue_confidence = conf
                image_draw = draw(
                    image_draw=image,
                    label=label,
                    confidence=conf,
                    point_1=(x1, y1),
                    point_2=(x2, y2),
                    color=color,
                )
        save_image(image_draw)


if __name__ == "__main__":
    main()
