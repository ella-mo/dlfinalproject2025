import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def map_rasters_to_fixations(fixation_csv_path, raster_npy_path, session_id, save_path):
    save_folder = os.path.dirname(save_path)
    os.makedirs(save_folder, exist_ok=True)

    # Load data
    fixations = pd.read_csv(fixation_csv_path)
    raster_data = np.load(raster_npy_path)  # shape = (units, time)

    print(f"✅ Loaded raster matrix: {raster_data.shape}")

    num_units, num_time_bins = raster_data.shape
    halfway = num_time_bins // 2

    # === Preprocessing: Remove bad units ===
    keep_units = []

    for i in range(num_units):
        first_half_spikes = raster_data[i, :halfway].sum()
        second_half_spikes = raster_data[i, halfway:].sum()

        if second_half_spikes == 0:
            continue  # Drop units with no activity in second half

        mean_first = first_half_spikes / halfway
        mean_second = second_half_spikes / (num_time_bins - halfway)

        if mean_first == 0:  # Avoid div-by-zero
            continue

        if abs(mean_first - mean_second) / mean_first > 0.5:
            continue  # Drop unstable units

        keep_units.append(i)

    print(f"✅ Units before filtering: {num_units}")
    print(f"✅ Units after filtering: {len(keep_units)}")

    # Filter raster_data
    raster_data = raster_data[keep_units, :]

    # Prepare time bins
    time_bins_ms = np.arange(raster_data.shape[1])  # 0ms, 1ms, 2ms, ...

    # Initialize accumulator
    image_activation = {}

    # Loop over fixations
    for idx, row in tqdm(fixations.iterrows(), total=len(fixations)):
        start_time_ms = int(row['start_time'] * 1000)
        stop_time_ms = int(row['stop_time'] * 1000)
        image_path = row['image_path']

        time_mask = (time_bins_ms >= start_time_ms) & (time_bins_ms < stop_time_ms)
        if not np.any(time_mask):
            continue

        summed_spikes = raster_data[:, time_mask].sum(axis=1)  # (units,)

        if image_path not in image_activation:
            image_activation[image_path] = summed_spikes
        else:
            image_activation[image_path] += summed_spikes

    # Build final DataFrame
    image_paths = []
    activation_data = []

    for img, act in image_activation.items():
        image_paths.append(img)
        activation_data.append(act)

    neuron_matrix = pd.DataFrame(activation_data, columns=[f"unit_{i}" for i in range(raster_data.shape[0])])
    neuron_matrix.insert(0, 'image_path', image_paths)

    neuron_matrix.to_csv(save_path, index=False)
    print(f"✅ Saved neuron mappings to {save_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 5:
        print("Usage: python raster_fixations.py <fixation_csv_path> <raster_npy_path> <session_id> <save_path>")
        sys.exit(1)

    fixation_csv_path = sys.argv[1]
    raster_npy_path = sys.argv[2]
    session_id = sys.argv[3]
    save_path = sys.argv[4]

    map_rasters_to_fixations(fixation_csv_path, raster_npy_path, session_id, save_path)
