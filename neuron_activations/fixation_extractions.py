import os
import pandas as pd
from pynwb import NWBHDF5IO

def extract_fixations(nwb_file_path, session_id):
    save_folder = "../fixations"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{session_id}_fixation_data.csv")

    with NWBHDF5IO(nwb_file_path, mode='r') as io:
        nwbfile = io.read()

        # --- Extract fixation data ---
        fixations = nwbfile.processing['behavior'].data_interfaces['fixations']
        fixation_df = pd.DataFrame({
            'id': fixations['id'].data[:],
            'ord_in_trial': fixations['ord_in_trial'].data[:],
            'start_time': fixations['start_time'].data[:],
            'stop_time': fixations['stop_time'].data[:],
            'trial_id': fixations['trial_id'].data[:],
            'x': fixations['x'].data[:],
            'y': fixations['y'].data[:]
        })

        # --- Extract image indices and paths ---
        stim = nwbfile.stimulus['presentations']
        image_indices = stim.data[:]  # Each trial's index into external_file
        image_paths = [str(p) for p in stim.indexed_timeseries.external_file]

        # --- Build lookup: trial_id -> image index -> image path ---
        trial_id_to_index = {i: int(image_indices[i]) for i in range(len(image_indices))}
        trial_id_to_path = {i: image_paths[trial_id_to_index[i]] for i in trial_id_to_index}

        # --- Add columns to fixation dataframe ---
        fixation_df['image_index'] = fixation_df['trial_id'].map(trial_id_to_index)
        fixation_df['image_path'] = fixation_df['trial_id'].map(trial_id_to_path)

        # --- Save to proper fixations folder ---
        fixation_df.to_csv(save_path, index=False)

        print(f"✅ Saved fixations with image paths to {save_path}")
        print(fixation_df.head())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python fixation_extractions.py <nwb_file_path> <session_id>")
        sys.exit(1)

    nwb_file_path = sys.argv[1]
    session_id = sys.argv[2]
    extract_fixations(nwb_file_path, session_id)
