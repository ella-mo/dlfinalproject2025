import os
import numpy as np
from pynwb import NWBHDF5IO

def extract_rasters(nwb_path, session_id):
    save_folder = "../rasters"
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, f"{session_id}_raster_data.npy")  # Save as npy file

    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()

        if 'ecephys' not in nwbfile.processing or 'rasters' not in nwbfile.processing['ecephys'].data_interfaces:
            print(f"[!] No rasters found in {session_id}, skipping.")
            return

        rasters = nwbfile.processing['ecephys']['rasters']
        raster_data = rasters.data[:].T  # (units, time)

        np.save(save_path, raster_data)  # Save efficiently
        print(f"✅ Saved rasters to {save_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python raster_extractions.py <nwb_path> <session_id>")
        sys.exit(1)

    nwb_path = sys.argv[1]
    session_id = sys.argv[2]
    extract_rasters(nwb_path, session_id)
