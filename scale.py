import os
from PIL import Image

# Input and output directories
input_folder = "path_to_input_folder"
output_folder = "path_to_output_folder"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image (you can add more extensions if needed)
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
        # Construct full file path
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # Open and resize the image
        with Image.open(input_path) as img:
            img_resized = img.resize((1080, 1080))

            # Save the resized image to the output folder
            img_resized.save(output_path)

            # Optionally, print the file being processed
            print(f"Resized {filename} and saved to {output_folder}")
