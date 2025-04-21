from pynwb import NWBHDF5IO
import pandas as pd
import numpy as np

# === Load NWB File ===
nwb_file_path = "preprocessing/sub-An_ses-20220105T173929.nwb"

with NWBHDF5IO(nwb_file_path, 'r') as io:
    nwbfile = io.read()
    
    rasters = nwbfile.processing['ecephys']['rasters']
    raster_data = rasters.data[:]  # shape: (time, units)
    starting_time = rasters.starting_time
    rate = rasters.rate
    
    # Build time vector
    num_timepoints = raster_data.shape[0]
    time_vector = np.array([starting_time + i / rate for i in range(num_timepoints)])

    unit_table = nwbfile.processing['ecephys']['unit_names']
    unit_ids = unit_table['id'].data[:]
    unit_names = [
        x.decode('utf-8') if isinstance(x, bytes) else str(x)
        for x in unit_table['unit_name'].data[:]
    ]

# === Preprocessing Criteria ===
halfway = num_timepoints // 2
first_half = raster_data[:halfway, :]
second_half = raster_data[halfway:, :]

mean_first = np.mean(first_half, axis=0)
mean_second = np.mean(second_half, axis=0)

nonzero_in_second_half = mean_second > 0
within_50_percent = np.abs(mean_first - mean_second) / np.maximum(mean_first, 1e-8) <= 0.5
valid_units = nonzero_in_second_half & within_50_percent

# === Filter Rasters and Metadata ===
filtered_rasters = raster_data[:, valid_units]
filtered_ids = unit_ids[valid_units]
filtered_names = np.array(unit_names)[valid_units]

# === Build DataFrame ===
time_cols = [f"time_{t:.3f}s" for t in time_vector]
df = pd.DataFrame(filtered_rasters.T, columns=time_cols)
df.insert(0, "unit_name", filtered_names)
df.insert(0, "unit_id", filtered_ids)

# === Save ===
df.to_csv("preprocessing/preprocessed_raster_data.csv", index=False)
print("Saved to preprocessed_raster_data.csv")
