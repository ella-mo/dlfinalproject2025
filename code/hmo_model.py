import numpy as np
import tensorflow as tf
import random

from cnn_module import CNNModule



class HMOModel(tf.keras.Model):
    def __init__(self, cnns):
        self.layer1 = cnns[:10]
        self.layer2 = cnns[10:20]
        self.layer3 = cnns[20:30]

        self.dense = tf.keras.layers.Dense(1250)
    
    
    def call(self, image):
        out_tensors = []
        for cnn in self.layer1:
            output = cnn(image)
            out_tensors.append(output)
        
        lay2in = tf.concat(out_tensors, axis=3) # shape of each should be (batch_size, new_height, new_width, filters)

        out_tensors = []
        for cnn in self.layer2:
            output = cnn(lay2in)
            out_tensors.append(output)

        lay3in = tf.concat(out_tensors, axis=3)

        out_tensors = []
        for cnn in self.layer3:
            output = cnn(lay3in)
            out_tensors.append(output)

        lay4in = tf.concat(out_tensors, axis=3)

        return self.dense(lay4in)

