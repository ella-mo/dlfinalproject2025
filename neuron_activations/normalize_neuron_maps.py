import os
import pandas as pd
from tqdm import tqdm

def normalize_neuron_maps(neuron_maps_folder, fixations_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for region in os.listdir(neuron_maps_folder):
        region_path = os.path.join(neuron_maps_folder, region)
        if not os.path.isdir(region_path):
            continue

        for matrix_file in tqdm(os.listdir(region_path), desc=f"Processing {region}"):
            if not matrix_file.endswith(".csv"):
                continue

            matrix_path = os.path.join(region_path, matrix_file)

            # Load neuron activation matrix
            df = pd.read_csv(matrix_path)

            # ✅ Drop rows with missing image_path
            df = df.dropna(subset=['image_path'])

            image_paths = df['image_path']
            neuron_data = df.drop(columns=['image_path'])

            # Find corresponding fixation file
            session_id = matrix_file.replace('_matrix_neurons_by_image.csv', '')
            fixation_file = f"../fixations/{session_id}_fixation_data.csv"
            if not os.path.exists(fixation_file):
                print(f"[!] Fixation file not found for {session_id}, skipping normalization.")
                continue

            fixations = pd.read_csv(fixation_file)

            # Build total time mapping per image
            image_total_time = {}
            for idx, row in fixations.iterrows():
                start_time = row['start_time']
                stop_time = row['stop_time']
                img_path = row['image_path']
                duration = stop_time - start_time
                if pd.notna(img_path):  # Make sure path isn't nan
                    image_total_time[img_path] = image_total_time.get(img_path, 0.0) + duration

            # Normalize each image row
            normalized_data = []
            for i, img_path in enumerate(image_paths):
                total_time = image_total_time.get(img_path, None)
                if total_time is None or total_time == 0:
                    print(f"[!] Warning: No valid duration for {img_path}, filling with zeros.")
                    normalized_data.append([0] * neuron_data.shape[1])
                else:
                    normalized_data.append(neuron_data.iloc[i] / total_time)

            # Build and save normalized dataframe
            normalized_df = pd.DataFrame(normalized_data, columns=neuron_data.columns)
            normalized_df.insert(0, 'image_path', image_paths.values)

            region_output_path = os.path.join(output_folder, region)
            os.makedirs(region_output_path, exist_ok=True)

            output_file_path = os.path.join(region_output_path, matrix_file)
            normalized_df.to_csv(output_file_path, index=False)
            print(f"✅ Saved normalized matrix to {output_file_path}")

if __name__ == "__main__":
    neuron_maps_folder = "../neuron_maps"
    fixations_folder = "../fixations"
    output_folder = "../normalized_neuron_maps"

    normalize_neuron_maps(neuron_maps_folder, fixations_folder, output_folder)
