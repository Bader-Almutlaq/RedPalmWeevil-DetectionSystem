import os
import shutil

folder = "./data/mixed_dataset"
target_1 = "./data/insects.class"
target_2 = "./data/rpw.class"
os.makedirs(target_1, exist_ok=True)
os.makedirs(target_2, exist_ok=True)

insect = []
rpw = []

for root, _, files in os.walk(folder):
    for file in files:
        if "Insect" in file:
            insect.append(file)
        else:
            rpw.append(file)

for i, filename in enumerate(insect, 1):
    source_path = os.path.join(folder, filename)
    _, ext = os.path.splitext(filename)
    dest_path = os.path.join(target_1, f"insect-{i}{ext}")
    shutil.copy2(source_path, dest_path)

for i, filename in enumerate(rpw, 1):
    source_path = os.path.join(folder, filename)
    _, ext = os.path.splitext(filename)
    dest_path = os.path.join(target_2, f"rpw-{i}{ext}")
    shutil.copy2(source_path, dest_path)