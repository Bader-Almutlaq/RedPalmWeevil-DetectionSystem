# This files separates the mixed dataset into two folders and numbers the images.
import os
import shutil

folder = "./data/mixed_dataset"
target_1 = "./data/RPW.class"
target_2 = "./data/NRPW.class"
os.makedirs(target_1, exist_ok=True)
os.makedirs(target_2, exist_ok=True)

rpw = []
nrpw = []

for root, _, files in os.walk(folder):
    for file in files:
        if "RPW" in file:
            rpw.append(file)
        else:
            nrpw.append(file)

print(f"The number of NRPWs: {len(rpw)}")
print(f"The number of NRPWs: {len(nrpw)}")


for i, filename in enumerate(rpw, 1):
    source_path = os.path.join(folder, filename)
    _, ext = os.path.splitext(filename)
    dest_path = os.path.join(target_1, f"RPW-{i}{ext}")
    shutil.copy2(source_path, dest_path)

for i, filename in enumerate(rpw, 1):
    source_path = os.path.join(folder, filename)
    _, ext = os.path.splitext(filename)
    dest_path = os.path.join(target_2, f"NRPW-{i}{ext}")
    shutil.copy2(source_path, dest_path)
