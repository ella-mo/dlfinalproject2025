import pickle
import numpy as np
import tensorflow as tf
import os


def unpickle(file) -> dict[str, np.ndarray]:
    """
    CIFAR data contains the files data_batch_1, data_batch_2, ..., 
    as well as test_batch. We have combined all train batches into one
    batch for you. Each of these files is a Python "pickled" 
    object produced with cPickle. The code below will open up a 
    "pickled" object (each file) and return a dictionary.
    NOTE: DO NOT EDIT
    :param file: the file to unpickle
    :return: dictionary of unpickled data
    """
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def get_next_batch(idx, inputs, labels, batch_size=100) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an index, returns the next batch of data and labels. Ex. if batch_size is 5, 
    the data will be a numpy matrix of size 5 * 32 * 32 * 3, and the labels returned will be a numpy matrix of size 5 * 10.
    """
    return (inputs[idx*batch_size:(idx+1)*batch_size], np.array(labels[idx*batch_size:(idx+1)*batch_size]))


# def get_data(file_path, classes) -> tuple[np.ndarray, tf.Tensor]:
#     """
#     Given a file path and a list of class indices, returns an array of 
#     normalized inputs (images) and an array of labels. 
    
#     - **Note** that because you are using tf.one_hot() for your labels, your
#     labels will be a Tensor, hence the mixed output typing for this function. This 
#     is fine because TensorFlow also works with NumPy arrays, which you will
#     see more of in the next assignment. 

#     :param file_path: file path for inputs and labels, something 
#                         like 'CIFAR_data_compressed/train'
#     :param classes: list of class labels (0-9) to include in the dataset

#     :return: normalized NumPy array of inputs and tensor of labels, where 
#                 inputs are of type np.float32 and has size (num_inputs, width, height, num_channels) and 
#                 Tensor of labels with size (num_examples, num_classes)
#     """
#     unpickled_file: dict[str, np.ndarray] = unpickle(file_path)
#     inputs: np.ndarray = np.array(unpickled_file[b'data'])
#     labels: np.ndarray = np.array(unpickled_file[b'labels'])

#     for i,l in enumerate(labels):
#         if(l not in classes):
#             labels[i] = 0
#     label_indices = np.nonzero(labels)
#     inputs = inputs[label_indices]
#     labels = labels[label_indices]

#     inputs = tf.reshape(inputs, shape=[inputs.shape[0], 3, 32, 32]).numpy()
#     inputs = tf.transpose(inputs, perm = [0,2, 3, 1])
#     inputs = inputs/255

#     sorted_classes = np.sort(classes)
#     for i in range(len(sorted_classes)):
#         labels = np.where(labels == classes[i], i, labels)

    
#     labels = tf.one_hot(labels, len(classes)).numpy()
    

#     return (inputs, labels)

def get_data(file_path) -> tuple[np.ndarray, np.ndarray]:
    """
    Loads data from a .pkl file containing 224x224 images and binary labels.
    Returns normalized image data and one-hot encoded labels.
    """
    with open(file_path, 'rb') as fo:
        unpickled_file = pickle.load(fo)

    inputs = np.array(unpickled_file[b'data'])  # shape: (N, 224, 224, 3) or (N, C*H*W)
    labels = np.array(unpickled_file[b'labels'])

    # If images are saved flattened, unflatten them
    if len(inputs.shape) == 2:
        inputs = inputs.reshape(-1, 3, 112, 112).transpose(0, 2, 3, 1)

    inputs = inputs.astype(np.float32) / 255.0
    labels = tf.one_hot(labels, 2).numpy()  # Change '2' if using more than 2 classes

    return inputs, labels
