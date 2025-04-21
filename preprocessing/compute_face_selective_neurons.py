import pandas as pd
import numpy as np

# Load raster data: rows = units, cols = time
raster_df = pd.read_csv("preprocessing/preprocessed_raster_data.csv")

# Load zeroth fixation metadata with face labels
# Assumed columns: trial_id, is_face (bool), unit_id, response_window_start, response_window_end
fixation_meta = pd.read_csv("preprocessing/zeroth_fixation_metadata.csv")  # or generate from raw fixations + face ROI

# Get time column names and convert to seconds
time_cols = [col for col in raster_df.columns if col.startswith("time_")]
time_vals = np.array([float(col.replace("time_", "").replace("s", "")) for col in time_cols])

# Initialize results
fsi_results = []

# For each unit, calculate FSI
for _, unit_row in raster_df.iterrows():
    unit_id = unit_row['unit_id']
    
    # Get all zeroth fixations for this unit
    unit_fixations = fixation_meta[fixation_meta['unit_id'] == unit_id]
    
    face_rates = []
    nonface_rates = []
    
    for _, fx in unit_fixations.iterrows():
        # Get response window for this fixation
        t_start, t_end = fx['response_window_start'], fx['response_window_end']
        time_mask = (time_vals >= t_start) & (time_vals <= t_end)
        response = unit_row[time_cols].values[time_mask]
        mean_fr = np.mean(response)
        
        # Sort by category
        if fx['is_face']:
            face_rates.append(mean_fr)
        else:
            nonface_rates.append(mean_fr)
    
    # Compute FSI
    R_face = np.mean(face_rates) if face_rates else 0
    R_nonface = np.mean(nonface_rates) if nonface_rates else 0
    denom = R_face + R_nonface
    fsi = (R_face - R_nonface) / denom if denom > 0 else 0

    fsi_results.append({
        'unit_id': unit_id,
        'fsi': fsi,
        'face_selective': fsi >= 0.2
    })

# Save FSI results
fsi_df = pd.DataFrame(fsi_results)
fsi_df.to_csv("face_selectivity_index.csv", index=False)
print("Saved face_selectivity_index.csv")
