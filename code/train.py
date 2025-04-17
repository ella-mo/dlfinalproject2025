import numpy as np
import tensorflow as tf
import random

from cnn_module import CNNModule


SAMPLE_SPACE_SIZE = 50 # how many CNN modules to generate





def generate_random_cnn_module():
    num_layers = random.choice([1, 2, 3])
    filters = random.choice([16, 32, 64])
    kernel_size = random.choice([3, 5, 7])
    pool_type = random.choice(['max', 'avg', 'soft'])
    norm = random.choice([True, False])
    threshold = random.uniform(0.0, 1.0)

    return CNNModule(config={
        "layers": num_layers,
        "filters": filters,
        "kernel_size": kernel_size,
        "pool": pool_type,
        "normalize": norm,
        "threshold": threshold
    })

def train_module(module, inputs):
    module.forward(inputs)

for i in range(SAMPLE_SPACE_SIZE):

    curr_module = generate_random_cnn_module()
    train_module()