import pandas as pd

def compute_csv_overlap(csv_a_path, csv_b_path):
    # Load both CSV files
    df_a = pd.read_csv(csv_a_path)
    df_b = pd.read_csv(csv_b_path)

    df_b['image_path'] = df_b['image_path'].str.replace('Stimuli/', '', regex=False)

    # Convert to sets for fast lookup
    set_a = set(df_a['Filename'])
    set_b = set(df_b['image_path'])

    # Find how many image paths in A are also in B
    matches = len(set_a.intersection(set_b))
    total = len(set_a)

    percent = (matches / total) * 100

    print(f"{matches} out of {total} ({percent:.2f}%) image paths from CSV A are found in CSV B.")
    return percent


# Example usage:
csv_a_path = r"preprocessing\face.csv"
csv_b_path = r"combinedAIT.csv"
compute_csv_overlap(csv_a_path, csv_b_path)