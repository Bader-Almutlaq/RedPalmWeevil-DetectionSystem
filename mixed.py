import os
import shutil
import random


def create_mixed_dataset(
    base_folder, output_folder, extensions=(".jpg", ".jpeg", ".png")
):
    """
    Combine images from multiple subfolders into a single mixed folder,
    rename them based on their source folder, and shuffle the ordering.

    Args:
        base_folder: Path to the parent folder containing dataset subfolders
        output_folder: Path where mixed images will be saved
        extensions: Tuple of image file extensions to include
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Expected subfolder names
    insects_folder = "Dataset-Other-Insects"
    rpw_folder = "Dataset-RPW"

    # List to store all valid image files and a dictionary to track file counts per folder
    all_image_files = []
    files_by_folder = {}

    # Process each folder specifically
    for subfolder in [insects_folder, rpw_folder]:
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Warning: Folder {subfolder_path} not found")
            continue

        folder_files = []  # Store images from the current subfolder

        # Walk through all files in the subfolder
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.lower().endswith(extensions):
                    # Get full source path and store it with folder info
                    source_path = os.path.join(root, file)
                    folder_files.append((source_path, subfolder, file))

        all_image_files.extend(folder_files)  # Add to the main list
        files_by_folder[subfolder] = len(folder_files)  # Store count for summary

    # Shuffle the list of all collected images to mix them randomly
    random.shuffle(all_image_files)

    # Copy files to the output folder with new naming conventions
    for i, (source_path, subfolder, original_filename) in enumerate(all_image_files, 1):
        # Extract the file extension
        _, ext = os.path.splitext(original_filename)

        # Create a new filename based on the original folder
        if subfolder == insects_folder:
            new_filename = f"{i:d}-Insects{ext}"
        elif subfolder == rpw_folder:
            new_filename = f"{i:d}-RPW{ext}"
        else:
            new_filename = f"{i:d}-Other{ext}"  # Fallback for unexpected folders

        # Construct the destination path
        dest_path = os.path.join(output_folder, new_filename)

        # Copy the file to the new location
        shutil.copy2(source_path, dest_path)

    # Print summary of the process
    total_copied = len(all_image_files)
    print(f"Mixed dataset created successfully in: {output_folder}")
    print(f"Total images copied: {total_copied}")
    for folder, count in files_by_folder.items():
        suffix = "Insects" if folder == insects_folder else "RPW"
        print(f"  - {folder}: {count} images (renamed with suffix '{suffix}')")

    # Create a metadata file to track original sources and new filenames
    with open(os.path.join(output_folder, "metadata.txt"), "w") as f:
        f.write("Number,Original_Folder,Original_Filename,New_Filename\n")
        for i, (source_path, subfolder, original_filename) in enumerate(
            all_image_files, 1
        ):
            _, ext = os.path.splitext(original_filename)
            if subfolder == insects_folder:
                new_filename = f"{i:d}-Insects{ext}"
            elif subfolder == rpw_folder:
                new_filename = f"{i:d}-RPW{ext}"
            else:
                new_filename = f"{i:d}-Other{ext}"
            f.write(f"{i:04d},{subfolder},{original_filename},{new_filename}\n")
    print(f"Metadata file created with original and new file information")


if __name__ == "__main__":
    # Configuration: Define input and output folders
    base_folder = "./data"
    output_folder = "data/mixed_dataset"

    # Set random seed for reproducibility (optional, remove for truly random results)
    random.seed(42)

    # Run the dataset creation function
    create_mixed_dataset(base_folder, output_folder)
