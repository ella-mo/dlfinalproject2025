import os
import numpy as np
import pandas as pd

def combine_fill_and_clean_csvs(folder_path, output_path):
    all_rows = []

    # Step 1: Load all CSV files
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csv_path = os.path.join(folder_path, filename)
            print(f"Loading {csv_path}...")
            df = pd.read_csv(csv_path)
            all_rows.append(df)

    # Step 2: Concatenate all rows
    full_df = pd.concat(all_rows, ignore_index=True)

    # Step 3: Keep only the latest entry for each image_path
    full_df = full_df.drop_duplicates(subset='image_path', keep='last')

    # Step 4: Identify unit columns and convert to numeric
    unit_columns = [col for col in full_df.columns if col.startswith('unit_')]
    full_df[unit_columns] = full_df[unit_columns].apply(pd.to_numeric, errors='coerce')

    # Step 5: Fill NaNs in unit columns with the column mean
    for col in unit_columns:
        mean_value = full_df[col].mean(skipna=True)
        full_df[col] = full_df[col].fillna(mean_value)

    # Step 6: Drop columns that are fully NaN or empty strings
    full_df = full_df.dropna(axis=1, how='all')
    full_df = full_df.loc[:, ~(full_df == '').all()]

    # Step 7: Save the cleaned DataFrame
    full_df.to_csv(output_path, index=False)
    print(f"Final cleaned and combined CSV saved to {output_path}")
    print(f"Final shape: {full_df.shape}")

# === USAGE ===
folder_path = 'neuron_maps/AIT'
output_path = 'combinedAIT.csv'
combine_fill_and_clean_csvs(folder_path, output_path)
