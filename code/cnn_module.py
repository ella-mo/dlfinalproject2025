import numpy as np
import tensorflow as tf
import random

from cnn_module import CNNModule

class CNNModule:
    def __init__(self, config):
        self.layers = config.get("layers", 1)
        self.filters = config.get("filters", 32)
        self.kernel_size = config.get("kernel_size", 3)
        self.pool = config.get("pool", "max")
        self.normalize = config.get("normalize", False)
        self.threshold = config.get("threshold", 0.5)

        self.model = self.build_model()

    def build_model(self, input_shape):
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for _ in range(self.num_layers):
            x = tf.keras.layers.Conv2D(
                self.filters, self.kernel_size, padding='same', activation=None
            )(x)
            x = tf.keras.layers.ReLU(threshold=self.threshold)(x)

            if self.pool_type == 'max':
                x = tf.keras.layers.MaxPooling2D()(x)
            else:
                x = tf.keras.layers.AveragePooling2D()(x)

            if self.normalize:
                x = tf.keras.layers.LayerNormalization()(x)

        # Global average pooling to produce a flat feature vector
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        return tf.keras.Model(inputs, x)
    
    def forward(self, image):
        return self.model(image)



