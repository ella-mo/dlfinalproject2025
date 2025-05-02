import numpy as np
import tensorflow as tf
import random

from cnn_module import CNNModule
from hmo_model import HMOModel
from train import generate_random_cnn_module, get_best_cnns
from resnet_pre import get_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import seaborn as sns           
from scipy.spatial.distance import pdist, squareform 
from scipy.stats import spearmanr
import pandas as pd

def plot_rdm(rdm, title="Representational Dissimilarity Matrix", figsize=(6,6), cmap="viridis"):
    plt.figure(figsize=figsize)
    sns.heatmap(rdm, cmap=cmap, square=True, cbar=True, 
                xticklabels=False, yticklabels=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def compute_rdm(features, metric="correlation"):
    """
    Compute Representational Dissimilarity Matrix from features.
    - features: (num_images, feature_dim)
    - metric: 'correlation', 'euclidean', etc. (passed to scipy.pdist)
    """
    dists = pdist(features, metric=metric)  # condensed distance matrix
    rdm = squareform(dists)  # make it square
    return rdm

def compare_rdms(rdm1, rdm2):
    # Flatten upper triangle only (avoid redundancy)
    triu_idx = np.triu_indices(rdm1.shape[0], k=1)
    vec1 = rdm1[triu_idx]
    vec2 = rdm2[triu_idx]
    corr, _ = spearmanr(vec1, vec2)
    return corr


# get data
all_inputs, all_labels, all_filenames = get_data("../preprocessing/cifar_batch_graypad.pkl")

# Split into train/test
train_inputs, test_inputs, train_labels, test_labels, train_filenames, test_filenames = train_test_split(
    all_inputs, all_labels, all_filenames, test_size=0.2, stratify=all_labels.argmax(axis=1), random_state=42
)

print("Train shape:", train_inputs.shape, train_labels.shape)
print("Test shape:", test_inputs.shape, test_labels.shape)

# Create tf.data Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# 2: Train CNNs and choose the best ones ------------------------------------------------------

print("getting cnns")
accuracy_list = get_best_cnns(train_dataset, test_dataset)

print("ACC LIST:")
print(accuracy_list)

top_10CNNs = np.argsort(accuracy_list)[-10:][::-1]

# 3: Combine CNNs into HMO model ------------------------------------------------------

cnn_list = []
for i in top_10CNNs:
    restored_model = tf.keras.models.load_model(f'models/module_{i:03d}.keras')
    cnn_list.append(restored_model)

for i in range(20):
    cnn_list.append(generate_random_cnn_module())

model = HMOModel(cnn_list)
print("working")

for cnn in model.layer1:
    cnn.trainable = False

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print("compiled")

model.fit(
    x=train_dataset,
    epochs=5
)

model.evaluate(
    x=test_dataset
)

# Step 1: Load and clean neural data
neural_data_df = pd.read_csv('../combinedAIT.csv')
neural_data_df['image_path'] = neural_data_df['image_path'].str.replace('Stimuli/', '', regex=False)
available_neural_paths = set(neural_data_df['image_path'])

# 2. Find which test images have neural data
available_test = set(test_filenames).intersection(available_neural_paths)

# 3. Build filtered arrays manually
filtered_inputs = []
filtered_labels = []
filtered_filenames = []

for inp, lbl, fname in zip(test_inputs, test_labels, test_filenames):
    if fname in available_test:
        filtered_inputs.append(inp)
        filtered_labels.append(lbl)
        filtered_filenames.append(fname)

filtered_inputs = np.array(filtered_inputs)
filtered_labels = np.array(filtered_labels)
filtered_filenames = np.array(filtered_filenames)

# 4. Build tf.data.Dataset from filtered arrays
test_dataset = tf.data.Dataset.from_tensor_slices((filtered_inputs, filtered_labels, filtered_filenames))
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

features = []
matched_filenames = []

for batch_images, batch_labels, batch_filenames in test_dataset:
    batch_features = model(batch_images, return_features=True)
    features.append(batch_features.numpy())
    batch_filenames = [f.numpy().decode('utf-8') for f in batch_filenames]
    matched_filenames.extend(batch_filenames)

cnn_features_matched = np.vstack(features)
np.save(cnn_features_matched, "cnn_features_matched.npy")

# Filter neural data using matched filenames
filtered_df = neural_data_df.set_index('image_path').loc[matched_filenames].reset_index()
neural_features = filtered_df.drop(columns=['image_path']).to_numpy()

neural_features = np.nan_to_num(neural_features, nan=0.0)

# Compute RDMs
cnn_rdm = compute_rdm(cnn_features_matched, metric="correlation")
plot_rdm(cnn_rdm, title="CNN RDM (matched to neural data)")

neural_rdm = compute_rdm(neural_features, metric="correlation")
plot_rdm(neural_rdm, title="Neural RDM (matched to neural data)")

print("CNN RDM min/max:", np.min(cnn_rdm), np.max(cnn_rdm))
print("Neural RDM min/max:", np.min(neural_rdm), np.max(neural_rdm))

similarity = compare_rdms(cnn_rdm, neural_rdm)

print(f"Matched {len(matched_filenames)} test images with neural data.")
print("Similarity:", similarity)

model.save("models/MAIN_MODEL.keras")

