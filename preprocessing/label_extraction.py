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
# username = r"..."  # replace if your OSCAR username is different

csv_path = f"face.csv"
zip_dir = f""  # contains Stimuli.z01 and Stimuli.zip
output_path = f"cifar_batch_graypad.pkl"
target_size = (112, 112)  # Target output size (square)

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
    face_data = []
    face_labels = []
    face_filenames = []

    nonface_data = []
    nonface_labels = []
    nonface_filenames = []


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

            if label == 'face':
                face_data.append(img_np)
                face_labels.append(1)
                face_filenames.append(filename)
            else:
                nonface_data.append(img_np)
                nonface_labels.append(0)
                nonface_filenames.append(filename)
            print(f"Done: {filename}")
        except Exception as e:
            print(f"Failed to process {file}: {e}")

min_class_size = min(len(face_data), len(nonface_data))
print(f"Balancing to {min_class_size} examples per class...")

# Randomly pick min_class_size from both
face_indices = np.random.choice(len(face_data), min_class_size, replace=False)
nonface_indices = np.random.choice(len(nonface_data), min_class_size, replace=False)

balanced_data = np.concatenate([
    np.array(face_data)[face_indices],
    np.array(nonface_data)[nonface_indices]
], axis=0)

balanced_labels = np.concatenate([
    np.array(face_labels)[face_indices],
    np.array(nonface_labels)[nonface_indices]
], axis=0)

balanced_filenames = (
    np.array(face_filenames)[face_indices].tolist() +
    np.array(nonface_filenames)[nonface_indices].tolist()
)

# === SAVE AS PICKLE ===
cifar_dict = {
    b'data': balanced_data,
    b'labels': balanced_labels.tolist(),
    b'filenames': balanced_filenames
}

with open(output_path, 'wb') as f:
    pickle.dump(cifar_dict, f)

print(f"Saved {len(balanced_data)} balanced, padded, labeled images to {output_path}")
