import pandas as pd
import numpy as np
import pickle
from PIL import Image, ImageOps
from io import BytesIO
import matplotlib.pyplot as plt
import os
import tempfile
import subprocess

# === PATHS ===
username = r"..."  # replace if your OSCAR username is different

csv_path = f"/users/{username}/dlfinalproject2025/preprocessing/face.csv"
zip_dir = f"/gpfs/scratch/{username}/Stimuli"  # contains Stimuli.z01 and Stimuli.zip
output_path = f"/gpfs/scratch/{username}/cifar_batch_graypad.pkl"
target_size = (224, 224)  # Target output size (square)

# === EXTRACT SPLIT ZIP ===
with tempfile.TemporaryDirectory() as tmpdir:
    print("Extracting split ZIP archive with 7z...")
    zip_path = os.path.join(zip_dir, "Stimuli.zip")
    result = subprocess.run(["7z", "x", zip_path, f"-o{tmpdir}"], capture_output=True, text=True)
    
    if result.returncode != 0:
        print("7z extraction failed:", result.stderr)
        exit(1)

    print("Extraction complete")

    # === READ FACE IMAGE LIST ===
    face_df = pd.read_csv(csv_path)
    face_images = set(face_df['Filename'].str.strip().str.lower())

    # === LABEL MAP ===
    label_map = {'non_face': 0, 'face': 1}

    # === CONTAINERS ===
    data = []
    labels = []
    filenames = []

    def resize_and_pad(img, size, fill=(128, 128, 128)):
        img.thumbnail(size, Image.Resampling.LANCZOS)
        pad_w = size[0] - img.size[0]
        pad_h = size[1] - img.size[1]
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return ImageOps.expand(img, padding, fill)

    # === FIND IMAGE FILES ===
    image_files = []
    for root, _, files in os.walk(tmpdir):
        for name in files:
            if name.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_files.append(os.path.join(root, name))

    # === PROCESS IMAGES ===
    for i, file in enumerate(image_files):
        try:
            img = Image.open(file).convert('RGB')
            img = resize_and_pad(img, target_size)
            img_np = np.array(img).transpose(2, 0, 1).reshape(-1)

            filename = os.path.basename(file).lower()
            label = 'face' if filename in face_images else 'non_face'

            data.append(img_np)
            labels.append(label_map[label])
            filenames.append(filename)

            print(f"Done: {filename}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")

# === SAVE AS PICKLE ===
cifar_dict = {
    b'data': np.stack(data),
    b'labels': labels,
    b'filenames': filenames
}

with open(output_path, 'wb') as f:
    pickle.dump(cifar_dict, f)

print(f"Saved {len(data)} padded, labeled images to {output_path}")
