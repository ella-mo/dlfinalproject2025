import numpy as np
import tensorflow as tf
import random


from cnn_module import CNNModule




SAMPLE_SPACE_SIZE = 10 # how many CNN modules to generate




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
       "threshold": threshold,
   })


def train_module(module, inputs, input_labels, test_inputs, test_labels):
      
   print("A")
   print(inputs.shape)
   print(input_labels.shape)


   module.compile(
       optimizer='adam',
       loss='categorical_crossentropy',
       metrics=['accuracy'])
  
   module.fit(x = inputs, y = input_labels, batch_size=32, epochs=5)


   loss, accuracy = module.evaluate(x = test_inputs, y = test_labels)


   return accuracy


   # features = module.get_conv_layers()
   #     #features.save("feature_extractor.keras")
   #     # saved_model = curr_module.get_conv_layers()
   # print("saving model.")
   # features.save(f'module_{1:03d}.keras')
   #module.save(f'module_{1:03d}.keras')




def get_best_cnns(inputs, input_labels, test_inputs, test_labels):
   accuracy_list = []
   for i in range(SAMPLE_SPACE_SIZE):


       curr_module = generate_random_cnn_module()
       print("training model " + str(i))
       acc = train_module(curr_module, inputs, input_labels, test_inputs, test_labels)
       accuracy_list.append(acc)


       print("saving model " + str(i) + ".")
       features = curr_module.get_conv_layers()
       features.save(f'models/module_{i:03d}.keras')
       print("model " + str(i) + " saved.")
   return accuracy_list

