import numpy as np
import tensorflow as tf
import random

from cnn_module import CNNModule
from hmo_model import HMOModel
from train import generate_random_cnn_module, get_best_cnns
from resnet_pre import get_data
from sklearn.model_selection import train_test_split

# # Get all data
all_inputs, all_labels = get_data(r'C:\Users\Taher Vahanvaty\Documents\csci1470\dlfinalproject2025\preprocessing\cifar_batch_graypad_trial.pkl')

# # Split into train/test
train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    all_inputs, all_labels, test_size=0.2, stratify=all_labels.argmax(axis=1), random_state=42
)

# print("Train shape:", train_inputs.shape, train_labels.shape)
# print("Test shape:", test_inputs.shape, test_labels.shape)

# # Create tf.data Datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(32).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels))
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# # 2: Train CNNs and choose the best ones ------------------------------------------------------

# print("getting cnns")
# accuracy_list = get_best_cnns(train_dataset, test_dataset)

# print("ACC LIST:")
# print(accuracy_list)

# top_10CNNs = np.argsort(accuracy_list)[-10:][::-1]

# 3: Combine CNNs into HMO model ------------------------------------------------------

with open("path/to/accuracy_list.txt", "r") as f:
    accuracy_list_string = f.read()

# Parse into list of floats
accuracy_values = np.array([float(x) for x in accuracy_list_string.strip().split(',')])

# Get indices of top 10 highest accuracies
top_10CNNs = np.argsort(accuracy_values)[-10:][::-1]
cnn_list = []

for i in top_10CNNs:
    # Step 1: Load the saved keras model
    saved_model = tf.keras.models.load_model(f'models/module_{i:03d}.keras')
    
    # Step 2: Create a fresh CNNModule
    new_cnn = CNNModule(config={})  # or supply the correct config if needed
    
    # Step 3: Build the new_cnn by calling it once (required before setting weights)
    dummy_input = tf.zeros((1, 224, 224, 3))  # or whatever your input size is
    new_cnn(dummy_input)
    
    # Step 4: Copy weights from saved model to fresh CNNModule
    new_cnn.set_weights(saved_model.get_weights())
    
    # Step 5: Append to list
    cnn_list.append(new_cnn)


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

model.save("models/MAIN_MODEL.keras")