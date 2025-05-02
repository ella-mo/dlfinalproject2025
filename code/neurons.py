import numpy as np
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


NEURON_DATA_FILE = 'combinedAIT.csv'
MODEL_DATA_FILE = 'cnn_features_matched.npy'
FILENAMES_FILE = 'ordered_filenames.csv'

filenames = pd.read_csv(FILENAMES_FILE)["filename"].tolist()
neuron_df = pd.read_csv(NEURON_DATA_FILE, index_col = "image_path")

# order the images
desired_order = filenames
print(desired_order)

# Clean index column
neuron_df.index = neuron_df.index.astype(str).str.strip()

desired_order = [str(name).strip() for name in desired_order]
desired_order = [f"Stimuli/{name}" for name in desired_order]

# Filter and reorder
image_names_in_df = [name for name in desired_order if name in neuron_df.index]
neuron_df_ordered = neuron_df.loc[image_names_in_df].reset_index()


neuron_df = neuron_df_ordered.transpose()
print(neuron_df)

neuron_data = neuron_df.iloc[1:].to_numpy()
model_data = np.load(MODEL_DATA_FILE)

r2_scores = np.zeros(neuron_data.shape[0])
neuron_data = neuron_data.transpose()

for i in range(121): # for each row

    X_train, X_test, y_train, y_test = train_test_split(model_data, neuron_data[:, i], test_size=0.25, random_state=42)

    pls = PLSRegression(n_components=5)
    pls.fit(X_train, y_train)

    # Predict
    Y_pred = pls.predict(X_test).flatten()  

    # R2
    r2_model_fit = r2_score(y_test, Y_pred)
    r2_scores[i] = r2_model_fit
    print(r2_model_fit)

# remove any outliers
q1 = np.percentile(r2_scores, 25)
q3 = np.percentile(r2_scores, 75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

filtered_r2 = r2_scores[(r2_scores >= lower_bound) & (r2_scores <= upper_bound)]

# Mean R2:
print("The mean r2 value across neurons is " + str(np.median(filtered_r2))) # they got median 50.4% ± 2.2%

# VISUALIZATIONS:
#1: histogram of R2 values
plt.hist(filtered_r2, bins=20)
plt.xlabel("R2 score")
plt.ylabel("Frequency")
plt.title("Distribution of R2 Scores Across Neurons")
plt.savefig('histogram.png')
#2: boxplot of r2 values
plt.boxplot(filtered_r2)
plt.ylabel("R2 Score")
plt.title("Distribution of R2 Scores Across Neurons")
plt.show()
