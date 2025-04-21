from retinaface import RetinaFace
import pandas as pd
import os
from PIL import Image

# Convert all PNGs in raw_images to RGB JPEGs
input_dir = "preprocessing/raw_images"
output_dir = "preprocessing/converted_images"
os.makedirs(output_dir, exist_ok=True)

converted = 0

for fname in os.listdir(input_dir):
    if not fname.lower().endswith(".png"):
        continue

    in_path = os.path.join(input_dir, fname)
    out_name = os.path.splitext(fname)[0] + ".jpg"
    out_path = os.path.join(output_dir, out_name)

    try:
        with Image.open(in_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(out_path, "JPEG")
            converted += 1
    except Exception as e:
        print(f"[!] Failed to convert {fname}: {e}")

print(f"✅ Converted {converted} PNGs to JPEGs in {output_dir}")


# === Set path to your image folder ===
image_dir = "preprocessing/converted_images"  # e.g., where all images from image_path column are stored

results = []

for fname in os.listdir(image_dir):
    if not fname.endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(image_dir, fname)
    try:
        faces = RetinaFace.detect_faces(img_path)
    except Exception as e:
        print(f"Error with {fname}: {e}")
        continue

    if isinstance(faces, dict):
        for face in faces.values():
            x1, y1, x2, y2 = face["facial_area"]
            results.append({
                "image_path": fname,
                "x_min": x1,
                "y_min": y1,
                "x_max": x2,
                "y_max": y2
            })

# Save face ROI results
roi_df = pd.DataFrame(results)
roi_df.to_csv("preprocessing/face_rois_from_retinaface.csv", index=False)
