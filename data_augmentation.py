import os
import cv2
import albumentations as A
from tqdm import tqdm

# Paths
INPUT_DIR = "./test/RPW-trap"
OUTPUT_DIR = "./test/RPW-trap-augmented"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the exact augmentation pipeline (each one separate)
augmentations = [
    ("hflip", A.Compose([A.HorizontalFlip(p=1.0)])),
    ("vflip", A.Compose([A.VerticalFlip(p=1.0)])),
    (
        "brightness",
        A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.3, contrast_limit=0.3, p=1.0
                )
            ]
        ),
    ),
    (
        "hue",
        A.Compose(
            [
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=1.0
                )
            ]
        ),
    ),
    ("sharpen", A.Compose([A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)])),
    ("rot_plus30", A.Compose([A.Rotate(limit=(30, 30), p=1.0)])),
    ("rot_minus30", A.Compose([A.Rotate(limit=(-30, -30), p=1.0)])),
]

# Process images
image_files = [
    f for f in os.listdir(INPUT_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

for img_file in tqdm(image_files):
    img_path = os.path.join(INPUT_DIR, img_file)
    image = cv2.imread(img_path)

    if image is None:
        continue

    base_name = os.path.splitext(img_file)[0]

    for name, aug in augmentations:
        augmented = aug(image=image)
        aug_img = augmented["image"]
        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}.jpg")
        cv2.imwrite(out_path, aug_img)
