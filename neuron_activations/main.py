import os
import pandas as pd
from tqdm import tqdm
import subprocess

def convert_session_id(session_id):
    # session_id looks like 'sub-Bf_ses-20211114T152023'
    parts = session_id.split('_')
    monkey = parts[0].replace('sub-', '')  # 'Bf'
    date = parts[1].replace('ses-', '')    # '20211114T152023'
    short_date = date[2:8]                 # '211114'
    return monkey + short_date             # 'Bf211114'

def process_all_sessions():
    sessions_root = "../000628"
    bank_array_csv = "../bank_array_regions.csv"
    bank_df = pd.read_csv(bank_array_csv)

    session_files = []

    # Step 1: Find all *_image.nwb files
    for monkey_folder in sorted(os.listdir(sessions_root)):
        full_monkey_path = os.path.join(sessions_root, monkey_folder)
        if not os.path.isdir(full_monkey_path):
            continue

        for filename in os.listdir(full_monkey_path):
            if filename.endswith("_image.nwb"):
                session_id = filename.replace("_image.nwb", "")  # e.g., sub-Bf_ses-20211114T152023
                session_files.append((monkey_folder, session_id))

    # Step 2: Process each session
    for monkey, session_id in tqdm(session_files, desc="Processing sessions"):
        nwb_image_path = os.path.join(sessions_root, monkey, f"{session_id}_image.nwb")
        nwb_raster_path = os.path.join(sessions_root, monkey, f"{session_id}.nwb")

        if not os.path.exists(nwb_image_path):
            print(f"[!] Missing image file for session {session_id}, skipping.")
            continue
        if not os.path.exists(nwb_raster_path):
            print(f"[!] Missing raster file for session {session_id}, skipping.")
            continue

        # Convert to short session ID
        short_session_id = convert_session_id(session_id)

        # Lookup region from bank_array_regions.csv
        row = bank_df[bank_df['Session'] == short_session_id]
        if row.empty:
            print(f"[!] No region found for {session_id} (short ID: {short_session_id}), skipping.")
            continue
        region = row.iloc[0]['Region']

        if region not in ["CIT", "AIT"]:
            print(f"[!] Region {region} for {session_id} is not CIT or AIT, skipping.")
            continue

        # 1. Extract fixations
        subprocess.run(["python", "fixation_extractions.py", nwb_image_path, session_id], check=True)

        # 2. Extract rasters
        subprocess.run(["python", "raster_extractions.py", nwb_raster_path, session_id], check=True)

        # 3. Map fixations to rasters and save in correct region folder
        fixation_csv = f"../fixations/{session_id}_fixation_data.csv"
        raster_csv = f"../rasters/{session_id}_raster_data.npy"
        region_folder = f"../neuron_maps/{region}"
        os.makedirs(region_folder, exist_ok=True)
        output_csv = os.path.join(region_folder, f"{session_id}_matrix_neurons_by_image.csv")

        if os.path.exists(fixation_csv) and os.path.exists(raster_csv):
            subprocess.run(["python", "raster_fixations.py", fixation_csv, raster_csv, session_id, output_csv], check=True)
        else:
            print(f"[!] Missing extracted CSVs for {session_id}, skipping mapping.")

if __name__ == "__main__":
    process_all_sessions()
