import torch
import cv2
import os
from datetime import datetime
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
# from picamera2 import Picamera2
import time
from config_utils import save_classification
from config_utils import load_config
import thop

# ======= Step 1: Configurations =======
config = load_config()
model_path = config["hyperparameters"]["model_path"]  # Path to the classification model
model_path_yolo = config["hyperparameters"][
    "yolo_path"
]  # Path to the YOLO object detection model
input_path = config["hyperparameters"][
    "input_path"
]  # Path to save the captured input image
output_path = config["hyperparameters"][
    "output_path"
]  # Path to save the processed output image


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Use GPU if available

# Load classification model
try:
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
except Exception as e:
    print(f"[classify] [!] Error loading classification model: {e}")
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


def take_image():
    """
    Takes and save image in the input_path
    """
    # try:
    #     picam2 = Picamera2()
    #     picam2.configure(picam2.create_still_configuration())  # No preview

    #     picam2.start()
    #     time.sleep(1)  # Give the camera a moment to warm up

    #     picam2.capture_file(input_path)  # Save image to file without opening a window

    #     picam2.close()
    # except Exception as e:
    #     print(f"[classify] [!] Prediction error: {e}")


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
        print(f"[classify] [!] Prediction error: {e}")
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
        print(f"[classify] [!] Drawing error: {e}")
    return image_draw


def run_inference(PRINT_INFO=False):
    """Main inference pipeline with metrics tracking"""
    # Modified model loading section
    cls_mem = 0.0  # Initialize outside try-block for global access
    try:
        if device.type == 'cuda':
            start_mem = torch.cuda.memory_allocated()
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        if device.type == 'cuda':
            cls_mem = (torch.cuda.memory_allocated() - start_mem) / (1024**2)
        else:  # Handle CPU memory estimation
            # Approximate CPU memory using parameter size (MB)
            param_size = sum(p.numel() * p.element_size() for p in model.parameters())
            cls_mem = param_size / (1024**2)
    except Exception as e:
        print(f"[classify] [!] Error loading classification model: {e}")
        model = None
        cls_mem = 0.0  # Ensure defined even if loading fails
    
    # Initialize metrics
    yolo_mem, flops_cls, flops_yolo = 0, 0, 0
    yolo_load_time, yolo_inference_time = 0.0, 0.0
    classification_time_total, num_crops = 0.0, 0

    if model is None:
        print("[classify] [!] Model not loaded. Skipping...")
        return False

    try:
        if PRINT_INFO:
            load_start = time.time()
            if device.type == 'cuda':
                start_mem = torch.cuda.memory_allocated()
        
        # Modified line - add .to(device) here
        yolo_model = YOLO(model_path_yolo).to(device)  # ← MAIN CHANGE HERE
        
        if PRINT_INFO:
            yolo_load_time = time.time() - load_start
            if device.type == 'cuda':
                yolo_mem = (torch.cuda.memory_allocated() - start_mem) / (1024**2)
    except Exception as e:
        print(f"[classify] [!] Error loading YOLO model: {e}")
        return False

    # FLOPs calculation
    if PRINT_INFO:
        try:
            # Classification model FLOPs
            dummy_input = torch.randn(1, 3, 380, 380).to(device)
            flops_cls, _ = thop.profile(model, inputs=(dummy_input,))
        except Exception as e:
            print(f"[classify] [!] Classification FLOPs error: {e}")
            flops_cls = 0

        try:
            # YOLO FLOPs calculation fixes
            yolo_model.to(device)  # ← Ensure model is on right device
            dummy_input = torch.randn(1, 3, 640, 640).to(device)
            # Access underlying PyTorch model with .model
            flops_yolo, _ = thop.profile(yolo_model.model, inputs=(dummy_input,))
        except Exception as e:
            print(f"[classify] [!] YOLO FLOPs error: {e}")
            flops_yolo = 0

    # Image capture and processing
    try:
        take_image()
        # Read image and convert to RGB immediately
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError("Failed to read image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    except Exception as e:
        print(f"[classify] [!] Image error: {e}")
        return False

    # YOLO inference timing
    try:
        if PRINT_INFO:
            infer_start = time.time()
        
        results = yolo_model(image)[0]
        
        if PRINT_INFO:
            yolo_inference_time = time.time() - infer_start
    except Exception as e:
        print(f"[classify] [!] YOLO inference error: {e}")
        return False

    # Process detections
    boxes = results.boxes.xyxy.cpu().numpy()
    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box[:4])
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Classification timing
            if PRINT_INFO:
                cls_start = time.time()

            pil_crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            input_tensor = transform(pil_crop).unsqueeze(0).to(device)
            pred, conf = predict(input_tensor)

            if PRINT_INFO:
                classification_time_total += time.time() - cls_start
                num_crops += 1

            rpw_color = (0, 0, 255)    # Red in BGR
            nrpw_color = (0, 255, 0)   # Green in BGR
            label = "RPW" if pred == 1 else "NRPW"
            color = rpw_color if label == "RPW" else nrpw_color  # Use predefined colors
            
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image_bgr = draw(
                image_draw=image_bgr,
                label=label,
                confidence=conf,
                point_1=(x1, y1),
                point_2=(x2, y2),
                color=color,
            )
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"[classify] [!] Box processing error: {e}")

    # Save final image with error handling
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with PIL to maintain RGB format
        pil_image = Image.fromarray(image)
        pil_image.save(output_path)
        print(f"[classify] Output image saved to {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"[classify] [!] Failed to save image: {e}")
        print(f"[classify] Output path: {os.path.abspath(output_path)}")

    # Print metrics tables
    if PRINT_INFO:
        # Memory Table
        print("\nMemory Usage (MB):")
        print("-" * 50)
        print(f"{'Category':<20} | {'Memory':<10}")
        print("-" * 50)
        print(f"{'Classification':<20} | {cls_mem:.2f}")
        print(f"{'YOLO':<20} | {yolo_mem:.2f}")
        print(f"{'Total':<20} | {cls_mem + yolo_mem:.2f}")
        print("-" * 50)

        # TFLOPs Table
        tflops_cls_total = (flops_cls * num_crops) / 1e12 if num_crops else 0
        tflops_yolo = flops_yolo / 1e12
        print("\nTFLOPs:")
        print("-" * 70)
        print(f"{'Category':<25} | {'Total':<15} | {'Per Crop/Image':<15}")
        print("-" * 70)
        print(f"{'Classification':<25} | {tflops_cls_total:.4f} | {flops_cls / 1e12:.4f}")
        print(f"{'YOLO':<25} | {tflops_yolo:.4f} | {tflops_yolo:.4f}")
        print(f"{'Total':<25} | {tflops_cls_total + tflops_yolo:.4f} | -")
        print("-" * 70)

        # Timing Table
        total_time = yolo_load_time + yolo_inference_time + classification_time_total
        print("\nTiming (seconds):")
        print("-" * 70)
        print(f"{'Category':<25} | {'Total':<15} | {'Per Crop/Image':<15}")
        print("-" * 70)
        if num_crops > 0:
            print(f"{'Classification':<25} | {classification_time_total:.4f} | {classification_time_total/num_crops:.4f}")
        else:
            print(f"{'Classification':<25} | {classification_time_total:.4f} | 0.0000")
        
        if len(boxes) > 0:
            yolo_total_time = yolo_load_time + yolo_inference_time
            print(f"{'YOLO':<25} | {yolo_total_time:.4f} | {yolo_total_time/len(boxes):.4f}")
        else:
            print(f"{'YOLO':<25} | {0.0000:.4f} | 0.0000")
        
        print(f"{'Total':<25} | {total_time:.4f} | -")
        print("-" * 70)

    return True

run_inference(PRINT_INFO=True)  # Set to False to disable metrics