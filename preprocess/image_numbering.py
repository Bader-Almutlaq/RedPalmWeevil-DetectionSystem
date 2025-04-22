# This script renames and numbers images in a folder in order,
# then saves the renamed copies in a new target folder.

import os
import shutil


name = "RPW"  # Class name to use as a prefix in filenames
starting_counter = 1  # Initial number for renaming files

# Define source and target folders based on the class name
source_folder = f"../test/{name}"
target_folder = f"../test/{name}_new"

# Create the target folder if it doesn't exist
os.makedirs(target_folder, exist_ok=True)

# List to store all image filenames from the source folder
image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
imgs = []

# Traverse the source folder and collect image filenames
for root, _, files in os.walk(source_folder):
    for file in files:
        if file.lower().endswith(image_extensions):
            imgs.append(file)

print(f"The number of images read: {len(imgs)}")

# Rename and copy each image to the target folder
for filename in imgs:
    # Full path to the source image
    source_path = os.path.join(source_folder, filename)

    # Extract the file extension (.jpg, .png, etc.)
    _, ext = os.path.splitext(filename)

    # Generate the new filename with numbering
    new_filename = f"{name}-{starting_counter}{ext}"

    # Full path for the destination image
    dest_path = os.path.join(target_folder, new_filename)

    # Copy the image to the new location with the new name
    shutil.copy2(source_path, dest_path)

    # Increment the counter for the next image
    starting_counter += 1
