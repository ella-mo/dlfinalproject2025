import numpy as np
import tensorflow as tf
import random


class CNNModule(tf.keras.Model):
   def __init__(self, config):
       super().__init__()




       self.num_layers = config.get("layers", 1)
       self.filters = config.get("filters", 32)
       self.kernel_size = config.get("kernel_size", 3)
       self.pool = config.get("pool", "max")
       self.normalize = config.get("normalize", False)
       self.threshold = config.get("threshold", 0.5)


       self.conv_layers = []


       self.linear_layer = tf.keras.layers.Dense(64, activation='relu')
       self.classification_layer = tf.keras.layers.Dense(2, activation='softmax')


       for i in range(self.num_layers):
           self.conv_layers.append(tf.keras.layers.Conv2D(self.filters, self.kernel_size, padding='same'))
           self.conv_layers.append(tf.keras.layers.ReLU(threshold=self.threshold))
           if self.pool == 'max':
               self.conv_layers.append(tf.keras.layers.MaxPooling2D())
           else:
               self.conv_layers.append(tf.keras.layers.AveragePooling2D())
           if self.normalize:
               self.conv_layers.append(tf.keras.layers.LayerNormalization())


       self.global_pool = tf.keras.layers.GlobalAveragePooling2D()


   def call(self, inputs, classify=True):
       x = inputs
       for layer in self.conv_layers:
           x = layer(x)


       if classify:
           cnn_out = self.global_pool(x)
           x = self.linear_layer(cnn_out)
           return self.classification_layer(x)
       else:
           return x
  
   def get_conv_layers(self):
       dummy_input = tf.keras.Input(shape=(None, None, 3))  # Flexible shape
       x = dummy_input
       for layer in self.conv_layers:
           x = layer(x)
       return tf.keras.Model(inputs=dummy_input, outputs=x, name="conv_feature_model")

















