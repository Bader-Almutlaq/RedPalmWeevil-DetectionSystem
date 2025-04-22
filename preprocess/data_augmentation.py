import os
import cv2
import albumentations as A
import numpy as np
from tqdm import tqdm

# Paths
INPUT_DIR = "../test/RPW_crop"
OUTPUT_DIR = "../test/RPW_crop_augmented"
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
                    hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=1.0
                )
            ]
        ),
    ),
    ("sharpen", A.Compose([A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0)])),
    ("rot_plus45", A.Compose([A.Rotate(limit=(30, 30), p=1.0)])),
    ("rot_minus45", A.Compose([A.Rotate(limit=(-30, -30), p=1.0)])),
]


# Function to handle rotation with white background
def rotate_with_white_background(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation, filling the background with white (255, 255, 255)
    rotated_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(180, 180, 180),
    )
    return rotated_image


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

    # Save original image
    original_out_path = os.path.join(OUTPUT_DIR, f"{base_name}_original.jpg")
    cv2.imwrite(original_out_path, image)

    # Save augmented versions
    for name, aug in augmentations:
        augmented = aug(image=image)
        aug_img = augmented["image"]

        # Handle rotation separately with white background
        if name == "rot_plus45":
            aug_img = rotate_with_white_background(image, 30)
        elif name == "rot_minus45":
            aug_img = rotate_with_white_background(image, -30)

        out_path = os.path.join(OUTPUT_DIR, f"{base_name}_{name}.jpg")
        cv2.imwrite(out_path, aug_img)
