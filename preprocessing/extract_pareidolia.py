import zipfile
from PIL import Image
from io import BytesIO

# Path to zip file
zip_path = r"C:\Users\Taher Vahanvaty\Downloads\willwx free_viewing master db-ROIs_Stimuli_primate_face.zip"

# Open and process the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    print("Filename,x,y,w,h,W,H")
    for file_name in zip_ref.namelist():
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                with zip_ref.open(file_name) as file:
                    img = Image.open(BytesIO(file.read()))
                    W, H = img.size
                    x, y, w, h = 0, 0, 0, 0  # No bounding box
                    print(f"{file_name},{x},{y},{w},{h},{W},{H}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
