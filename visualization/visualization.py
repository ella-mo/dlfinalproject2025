import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_normalized_neuron_heatmap(csv_path, output_path=None):
    # Load the neuron activation matrix
    df = pd.read_csv(csv_path)

    # Separate image_path and neuron data
    image_paths = df['image_path']
    neuron_data = df.drop(columns=['image_path'])

    # Normalize: divide each row by its own max value
    normalized_data = neuron_data.div(neuron_data.max(axis=1), axis=0).fillna(0)  # fill NaN (from division by 0) with 0

    # Plot
    plt.figure(figsize=(14, 8))
    plt.imshow(normalized_data, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Normalized Activation (0–1)')
    plt.xlabel('Units')
    plt.ylabel('Images')
    plt.title('Normalized Neuron Activation per Image')

    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"✅ Saved normalized heatmap to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Example usage:
    csv_path = "neuron_maps/V1/sub-An_ses-20220105T173929_matrix_neurons_by_image.csv"  # <-- adjust if needed
    plot_normalized_neuron_heatmap(csv_path)
