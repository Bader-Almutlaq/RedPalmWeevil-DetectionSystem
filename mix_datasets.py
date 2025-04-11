# This file is used to create a mixed dataset.
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
    os.makedirs(output_folder, exist_ok=True)

    nrpw_folder = "NRPW"
    rpw_folder = "RPW"

    all_image_files = []
    files_by_folder = {}

    for subfolder in [nrpw_folder, rpw_folder]:
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.exists(subfolder_path):
            print(f"Warning: Folder {subfolder_path} not found")
            continue

        folder_files = []
        for root, _, files in os.walk(subfolder_path):
            for file in files:
                if file.lower().endswith(extensions):
                    source_path = os.path.join(root, file)
                    folder_files.append((source_path, subfolder, file))

        all_image_files.extend(folder_files)
        files_by_folder[subfolder] = len(folder_files)

    random.shuffle(all_image_files)

    # Track new names with per-class counters
    class_counters = {nrpw_folder: 0, rpw_folder: 0, "Other": 0}

    metadata_records = []

    for source_path, subfolder, original_filename in all_image_files:
        _, ext = os.path.splitext(original_filename)

        if subfolder == nrpw_folder:
            class_counters[nrpw_folder] += 1
            new_filename = f"NRPW-{class_counters[nrpw_folder]}{ext}"
        elif subfolder == rpw_folder:
            class_counters[rpw_folder] += 1
            new_filename = f"RPW-{class_counters[rpw_folder]}{ext}"
        else:
            class_counters["Other"] += 1
            new_filename = f"Other-{class_counters['Other']}{ext}"

        dest_path = os.path.join(output_folder, new_filename)
        shutil.copy2(source_path, dest_path)

        metadata_records.append((subfolder, original_filename, new_filename))

    total_copied = len(all_image_files)
    print(f"Mixed dataset created successfully in: {output_folder}")
    print(f"Total images copied: {total_copied}")
    for folder, count in files_by_folder.items():
        suffix = "NRPW" if folder == nrpw_folder else "RPW"
        print(f"  - {folder}: {count} images (renamed with suffix '{suffix}')")

    # Save metadata file
    metadata_path = os.path.join(output_folder, "metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("Number,Original_Folder,Original_Filename,New_Filename\n")
        for i, (subfolder, original_filename, new_filename) in enumerate(
            metadata_records, 1
        ):
            f.write(f"{i:04d},{subfolder},{original_filename},{new_filename}\n")
    print(f"Metadata file created: {metadata_path}")


if __name__ == "__main__":
    base_folder = "./data"
    output_folder = "data/mixed_dataset"

    random.seed(42)

    create_mixed_dataset(base_folder, output_folder)
