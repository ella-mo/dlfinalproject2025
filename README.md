# CSCI1470 Deep Learning Final Project
By Ella, Patrick, and Taha

In this project, we aimed to explore the correlation between layers of CNNs trained on psychophysics behavioral tasks and layers in the visual processing pathway through recorded electrophysiological (EP) data. This research question falls under interpretability of classification models. We were inspired by the paper, “Performance-optimized hierarchical models predict neural responses in higher visual cortex,” by Yamins et al. However, because this paper does not provide code or data, we used the data from “Feature-selective responses in macaque visual cortex follow eye movements during natural vision,” by Xiao et al.  
Pre-existing models were either specialized for optimization of image classification or prediction of area IT activity. In Yamins et al. (2014), the hierarchical modular optimization (HMO) model has both high task performance and neural activity area predictivity.  Specifically, as shown by Figure 1, the HMO model is better at classifying images with high variation, such as different camera positions and lighting. 
However, since neither image nor EP data is available in the paper, we aim to explore whether a similar model can be replicated with different data. In Xiao et al. (2024), we found both task and EP data recorded from six areas in the macaque monkey ventral visual processing pathway. However, the data was collected from free-viewing tasks, so the data are trials of fixations and saccades of several thousand images. We decided to use the data on fixations and the image classification as face versus non-face. Moreover, we used labels of pareidolia stimuli to mimic the high variation of the data used in Yamins et al. (2014).

# Folders and code explanations:
## 000628 
This folder holds a few .nwb files from the data paper in sub-folders, which are the monkeys from which the fixation and raster data was obtained from. 

## AI_use
contains AI use transcripts
## code


## fixations
This folder holds extracted data from the .nwb files from 000628, created from fixation_extractions.py in /neuron_activations.

## models

## neuron_activations
The code creates normalized maps that map the average firing rate of a neuron during an image showing. Note: future use of this code should combine normalize_neuron_maps.py and main.py to calculate the average firing rate of the neuron in a single run. 

1. fixation_extractions.py: extracts the fixations and creates *_fixation_data.csv files.
2. main.py: uses fixation_extractions.py, raster_extractions.py and raster_fixations.py to create number of neurons x  number of images maps. 
3. normalize_neuron_maps.py: divides the neuron x image map by the time the image was shown to calculate the average firing rate of the neuron during the image.
4. raster_extractions.py: extracts and preprocesses the rasters to create *_raster_data.csv files.
5. raster_fixations.py: Maps the extracted data from the *_fixation_data.csv files and *_raster_data.csv files to create number of neurons x number of images maps. Each cell represents the total number of times the neuron fired during the image showing.

## neuron_maps
This folder holds the un-normalized neuron to image maps, created from raster_fixations.py in /neuron_activations.

## preprocessing
The code in this folder was used primarily to explore the data and the code from the data paper; none of the code here is directly relevant to our project.

1. image_extractions.py: extracts Stimuli/image_name.png and creates extracted_external_file.csv that holds png file names
2. raw_images_extractions.py: uses extracted_external_file.csv to create raw_images file, which extracts the images from Stimuli.zip 
3. face_rois_calculation.py: uses retinaface to categorize each of the extracted images as face vs non face and calculate ROIs. the calculations are held in face_rois_from_retinaface.csv
4. compute_face_selective_neurons: computes face selectivity of neurons using face_rois_from_retinaface.csv and preprocessed_raster_data.csv

## rasters
This folder holds extracted data from the .nwb files from 000628, created from raster_extractions.py in /neuron_activations.

## visualization
visualization.py creates heatmaps to visualize the un-normalized neuron maps.

## accuracy_list.txt, accuracy.txt
contains the list of accuracies from running /code with a sample size of 20 and 200 CNN modules.

## bank_array_regions.csv
This is taken directly from the data paper. This was used for /neuron_activations so as to only preprocess .nwb files that had neuronal data from areas CIT or AIT. Note: we ultimately only used data from area AIT. 

## combinedAIT.csv
This is the combined AIT data, as aforementioned in the description for bank_array_regions.csv.
