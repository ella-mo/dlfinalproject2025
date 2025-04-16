from pynwb import NWBHDF5IO
import pandas as pd

nwb_file_path = "preprocessing/sub-An_ses-20220105T173929_image.nwb"

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

    # --- Extract image indices and paths from stimulus presentation ---
    stim = nwbfile.stimulus['presentations']
    image_indices = stim.data[:]  # Each trial's index into external_file
    image_paths = [str(p) for p in stim.indexed_timeseries.external_file]

    # --- Build lookup for trial_id -> image_index & image_path ---
    trial_id_to_index = {i: int(image_indices[i]) for i in range(len(image_indices))}
    trial_id_to_path = {i: image_paths[trial_id_to_index[i]] for i in trial_id_to_index}

    # --- Add columns to fixation dataframe ---
    fixation_df['image_index'] = fixation_df['trial_id'].map(trial_id_to_index)
    fixation_df['image_path'] = fixation_df['trial_id'].map(trial_id_to_path)

    # --- Save to CSV ---
    fixation_df.to_csv("preprocessing/fixations_with_image_index_and_path.csv", index=False)
    print(fixation_df.head())
