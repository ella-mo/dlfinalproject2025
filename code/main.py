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

# Get all data
#all_inputs, all_labels = get_data(r'C:\Users\Taher Vahanvaty\Documents\csci1470\dlfinalproject2025\preprocessing\cifar_batch_graypad_trial.pkl')

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



all_inputs, all_labels, all_filenames = get_data("../preprocessing/cifar_batch_graypad_trial.pkl")

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

features = []

for batch_images, _ in test_dataset:
    batch_features = model(batch_images, return_features=True)  # (batch_size, 1250)
    features.append(batch_features.numpy())  # save the batch outputs

# After loop: stack all batches into one big array
cnn_features = np.vstack(features)  # (num_test_images, 1250)

# Now you can compute RDM
cnn_rdm = compute_rdm(cnn_features, metric="correlation")
plot_rdm(cnn_rdm, title="CNN RDM")
print(cnn_rdm)

"""neural_data_df = pd.read_csv('neural_data.csv')

# code to get the images we want based on the filename
filtered_df = neural_data_df[neural_data_df['img_id'].isin(test_filenames)]

# Step 2: Reorder according to test_filenames
# First set img_id as index so we can use .loc
filtered_df = filtered_df.set_index('img_id').loc[test_filenames].reset_index()

# Step 3: Drop the 'img_id' column and get a NumPy array of neural features
neural_features = filtered_df.drop(columns=['img_id']).to_numpy()

neural_rdm = compute_rdm(neural_features, metric="correlation")

similarity = compare_rdms(cnn_rdm, neural_rdm)
print("Spearman correlation CNN-IT:", similarity) """

model.save("models/MAIN_MODEL.keras")