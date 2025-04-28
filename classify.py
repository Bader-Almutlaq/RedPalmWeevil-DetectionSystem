import torch
import cv2
import os
from datetime import datetime
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# ======= Step 1: Configurations =======
model_path = (
    "saved_full_models/efficientnet_b3_rpw.pth"  # Path to the classification model
)
model_path_yolo = "saved_full_models/yolo.pt"  # Path to the YOLO object detection model
output_folder = "./positive"  # Folder where output images will be saved

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Use GPU if available

# Load classification model
try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
except Exception as e:
    print(f"[!] Error loading classification model: {e}")
    model = None  # Prevent crash if model loading fails

# Transformation applied to image crops before classification
# Model input sizes:
# - MobileNet:        224 x 224
# - EfficientNet-B0:  224 x 224
# - EfficientNet-B2:  260 x 260
# - EfficientNet-B3:  300 x 300
# - EfficientNet-B4:  380 x 380
transform = transforms.Compose(
    [
        transforms.Resize((380, 380)),  # TODO: Change this based on the model
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def predict(pil_image, model=model, transform=transform, device=device):
    """
    Predict the class of the input PIL image using the classification model.

    Args:
        pil_image (PIL.Image): The cropped image region.
        model (torch.nn.Module): The classification model.
        transform (torchvision.transforms): Transformations to apply.
        device (torch.device): CPU or CUDA device.

    Returns:
        tuple: Predicted class index and confidence score.
    """
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
    """
    Draw bounding box and label on the image.

    Args:
        image_draw (numpy.array): Image to draw on.
        label (str): Label for the object.
        confidence (float): Confidence score.
        point_1 (tuple): Top-left coordinate of bounding box.
        point_2 (tuple): Bottom-right coordinate of bounding box.
        color (tuple): Color of the rectangle (B, G, R).

    Returns:
        numpy.array: Image with drawn rectangle and label.
    """
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
    """
    Save the processed image to disk with a timestamp.

    Args:
        image (numpy.array): Image to save.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"classified_{timestamp}.jpg"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)
        print("Image was saved")
    except Exception as e:
        print(f"[!] Failed to save image: {e}")


def main():
    """
    Main function to run object detection and classification pipeline.

    Returns:
        bool: True if execution succeeded, False otherwise.
    """
    if model is None:
        print("[!] Model not loaded. Skipping...")
        return False

    # Load YOLO object detection model
    try:
        yolo_model = YOLO(model_path_yolo)
    except Exception as e:
        print(f"[!] Error loading YOLO model: {e}")
        return False

    # Read input image
    try:
        # TODO: Replace this with image capturing logic
        image = cv2.imread("test/temp.png")
        if image is None:
            raise ValueError("Image is None (possibly missing or unreadable)")
    except Exception as e:
        print(f"[!] Failed to read image: {e}")
        return False
    
    img_label = "Nothing"
    maxvalue_confidence = 0.0

    # Detect objects using YOLO
    try:
        print("=" * 95)
        print("Yolo info:", end="")
        results = yolo_model(image)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
    except Exception as e:
        print(f"[!] YOLO detection error: {e}")
        return False

    print("=" * 95)

    # Process each detected object
    for idx, box in enumerate(boxes):
        try:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Convert to PIL and apply transform
            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_crop).unsqueeze(0).to(device)

            # Predict class and confidence
            pred, conf = predict(pil_image=input_tensor)
            label = "RPW" if pred == 1 else "NRPW"
            color = (0, 0, 255) if label == "RPW" else (0, 255, 0)

            if label == "RPW" and conf > maxvalue_confidence:
                img_label = "RPW"
                maxvalue_confidence = conf
            elif label == "NRPW" and img_label != "RPW":
                img_label = "NRPW"
                maxvalue_confidence = conf
            # Draw prediction on image
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

    # Save final image
    save_image(image)

    # Log final results
    print("Model Results:")
    print(f"Label: {img_label}")
    print(f"Max RPW confidence: {maxvalue_confidence:.4f}")
    print("=" * 95)

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("[!] Script failed but exited cleanly.")
