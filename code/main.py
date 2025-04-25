import numpy as np
import tensorflow as tf
import random


from cnn_module import CNNModule
from hmo_model import HMOModel
from train import generate_random_cnn_module, get_best_cnns
from resnet_pre import get_data


# 1: get data ------------------------------------------------------


# images = ...
# image_labels = ...


# TRAIN_FILE = '/Users/pkj/Desktop/resnet_data/train'
# TEST_FILE = '/Users/pkj/Desktop/resnet_data/test'


# classes = [3,5]


# train_inputs, train_labels = get_data(TRAIN_FILE, classes)
# test_inputs, test_labels = get_data(TEST_FILE, classes)


# print(train_labels.shape)

from sklearn.model_selection import train_test_split

# Get all data
all_inputs, all_labels = get_data(r'C:\Users\Taher Vahanvaty\Documents\csci1470\dlfinalproject2025\data\cifar_batch_graypad.pkl')

# Split into train/test
train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    all_inputs, all_labels, test_size=0.2, stratify=all_labels.argmax(axis=1), random_state=42
)

# Optional sanity checks
print("Train shape:", train_inputs.shape, train_labels.shape)
print("Test shape:", test_inputs.shape, test_labels.shape)



# 2: train CNNs and choose the best ones ------------------------------------------------------


print("getting cnns")
accuracy_list = get_best_cnns(train_inputs, train_labels, test_inputs, test_labels)


print("ACC LIST:")
print(acc)


# 3: combine CNNs into HMO model ------------------------------------------------------


cnn_list = []
for i in range(10):
   restored_model = tf.keras.models.load_model(f'models/module_{i:03d}.keras')
   cnn_list.append(restored_model)


for i in range(20):
   cnn_list.append(generate_random_cnn_module())


#inputs = tf.keras.Input(shape=(None, None, 3))
# outputs = model(inputs)
# model = tf.keras.Model(inputs=inputs, outputs=outputs)
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


print(test_inputs.shape)


model.fit(
   x=train_inputs,         # training images, shape (num_samples, height, width, channels)
   y=train_labels,   # labels, shape (num_samples, size of classification problem)
   batch_size=32,
   epochs=5
)


model.evaluate(
   x=test_inputs,
   y=test_labels
)


model.save("models/MAIN_MODEL.keras")
