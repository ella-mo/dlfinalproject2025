import numpy as np
import tensorflow as tf
import random

from cnn_module import CNNModule
from hmo_model import HMOModel
from train import generate_random_cnn_module

# 1: get data ------------------------------------------------------

images = ...
image_labels = ...


# 2: train CNNs and choose the best ones ------------------------------------------------------

cnn_list = []
for i in range(30):
    cnn = generate_random_cnn_module()
    cnn_list.append(cnn)

# 3: combine CNNs into HMO model ------------------------------------------------------

model = HMOModel(cnn_list)


model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.evaluate(
    x=images,         # training images, shape (num_samples, height, width, channels)
    y=image_labels,   # labels, shape (num_samples, size of classification problem)
    batch_size=32,
)