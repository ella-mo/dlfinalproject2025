import numpy as np
import tensorflow as tf
import random


from cnn_module import CNNModule






class HMOModel(tf.keras.Model):
   def __init__(self, cnns):
       super().__init__()
      
       self.layer1 = cnns[:10]
       self.layer2 = cnns[10:20]
       self.layer3 = cnns[20:30]
       self.global_pool = tf.keras.layers.GlobalAveragePooling2D()


       self.dense = tf.keras.layers.Dense(1250, activation='relu')


       self.classify = tf.keras.layers.Dense(2, activation='softmax')
  
       self.image_shape = (8,8)
  
   def call(self, image):
       out_tensors = []
       for cnn in self.layer1:
           output = cnn(image, classify=False)
           output = tf.image.resize(output, self.image_shape)


           out_tensors.append(output)
           #print(output.shape)
      
       lay2in = tf.concat(out_tensors, axis=3) # shape of each should be (batch_size, new_height, new_width, filters)
       #print(lay2in.shape)


       out_tensors = []
       for cnn in self.layer2:
           output = cnn(lay2in, classify=False)
           #print(output.shape)
           output = tf.image.resize(output, self.image_shape)
           out_tensors.append(output)


       lay3in = tf.concat(out_tensors, axis=3)


       out_tensors = []
       for cnn in self.layer3:
           output = cnn(lay3in, classify=False)
           output = tf.image.resize(output, self.image_shape)
           out_tensors.append(output)


       lay4in = tf.concat(out_tensors, axis=3)






       return self.classify(self.dense(self.global_pool(lay4in)))







