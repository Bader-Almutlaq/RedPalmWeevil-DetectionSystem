import os
import cv2
from ultralytics import YOLO

# === CONFIGURATION ===
model_path = "best.pt"  # Path to your YOLOv8 model
input_folder = "light Phone caam RPW full res"
output_folder = f"{input_folder}_cropped"
os.makedirs(output_folder, exist_ok=True)

# Load the model
model = YOLO(model_path)
image_extensions = ('.jpg', '.jpeg', '.png')

# Process each image
for filename in os.listdir(input_folder):
    if filename.lower().endswith(image_extensions):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[!] Failed to read image: {image_path}")
            continue

        results = model(image)[0]

        for idx, box in enumerate(results.boxes.xyxy):
            x1, y1, x2, y2 = map(int, box)
            cropped = image[y1:y2, x1:x2]

            if cropped.size == 0:
                print(f"[!] Empty crop for {filename} box {idx}, skipping.")
                continue

            resized = cv2.resize(cropped, (160, 160), interpolation=cv2.INTER_AREA)

            base_filename, ext = os.path.splitext(filename)
            output_filename = f"{base_filename}_crop_{idx}{ext}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, resized)
            print(f"[✓] Saved: {output_path}")
