import numpy as np
import tensorflow as tf
import random

from cnn_module import CNNModule
from hmo_model import HMOModel

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
import numpy as np

import matplotlib.pyplot as plt

# Suppose you have X_train, X_test, Y_train, Y_test


neuron_data = ... # this should be shape (number of neurons x number of test images)
neuron_data_train = ... # this should be shape (number of neurons x number of train images)
model_out_data = ... # this should be shape (number of test images x 1250)
model_out_data_train = ... # this should be shape (number of train images x 1250)

r2_scores = np.zeros(neuron_data.shape[0])

for i, single_neuron in enumerate(neuron_data): # for each row (should be same number of rows in train/test data)
    # 1. Train PLS
    pls = PLSRegression(n_components=25) # 25 components is fixed from the paper
    pls.fit(model_out_data_train, neuron_data_train[i,:])

    # 2. Predict
    Y_pred = pls.predict(model_out_data).flatten()  # flatten if needed

    # 3. Raw model r^2
    r2_model_fit = r2_score(single_neuron, Y_pred)
    r2_scores[i] = r2_model_fit

    # # 4. Neuron split-half reliability
    # # Assume you have repeat trial responses: responses_trials (shape: num_images x num_repeats)

    # half1 = np.mean(responses_trials[:, :num_repeats//2], axis=1)
    # half2 = np.mean(responses_trials[:, num_repeats//2:], axis=1)

    # r_neuron = np.corrcoef(half1, half2)[0, 1]
    # r2_neuron = r_neuron ** 2

    # 5. Normalized explained variance
    #normalized_r2 = r2_model_fit / r2_neuron

# RESULTS:

# Mean R2:
print("The mean r2 value across neurons is " + str(np.mean(r2_scores))) # they actually did median 50.4% ± 2.2%

# VISUALIZATIONS:
#1: histogram of R2 values
plt.hist(r2_scores, bins=10)
plt.xlabel("R2 score")
plt.ylabel("Frequency")
plt.title("Distribution of R2 Scores Across Neurons")

#2: boxplot of r2 values
plt.boxplot(r2_scores)
plt.ylabel("R2 Score")
plt.title("Distribution of R2 Scores Across Neurons")







