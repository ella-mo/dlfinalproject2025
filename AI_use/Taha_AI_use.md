I used ChatGPT in a relatively significant way over the course of the project. All of the full AI use transcripts are posted in the enclosing folder.

###
* The first thing I used it for was in the "Face Specific Anaylsis Script".  This was when I was still attempting to understand how the data paper was preprocessing and postprocessing their data and turned out to be mainly useless for our project. I would ask ChatGPT related to their use of .ipynb files. I was using ChatGPT to make sense and get a broad overview of what these files were doing. Some exemplar questions are "okay go through script 0c", "okay so how do i determine how the researchers chose which images to show to the monkey", "so is 1a just looking at which neurons are firing for the monkey during a trial?", "where is it getting the neurons from", "so all of this analysis data is after the data from the monkeys have been collected?".

* The next file "Load CNN Models HMO" was to do with reloading our HMO model after it had been trained. We had siginifciant issues delaing with loading and saving models in the assignment due to the weird architecture of the models. Since we were using the CNNModule class and the HMOModel class, these had specific architectures which would not be recorded in our savefile with the model. It would simply save the files and Keras model, losing much of the structure that we had built up. To figure out how to deal with this we asked some exemplar transcript questions like "we have an accuracy list of comma separated values, can you load the corresponding models and put them in top10_CNNS", "its in a txt file could you laod it in too", "this causes an issue because it saves it as a keras module not a cnnmodule, could you createa  new cnn module, clone the old one and the load the weights from the old model?" "config={} what is the config file suppsoed to specify the weght shape?"

* The next file "Balance face non-face images" was to do with balancing the dataset. We had issues with the dataset having 19000 non-face images and 1000 face images. We had already extracted the data but I was unsure about how to balance it so that we had a 50/50 split between face and non-face images. Without this split our model was doing effectively nothing, since it attained a high accuracy just by guessing face every time. Some example transcripts are "so say this has 1000 face images and 10000 non-face images, the total result will be 1000 face and 1000 nonface, all of the face images will be present?".

* The next file "Binary Classification Labelling" was to do with the extraction of the dataset itself. We wanted to replicate the CIFAR style dataset since that was what our CNN's were working with and it would have reduced the amount of friction when it came to putting all of the pieces together. I did not know the specifics about how .pkl files worked and the labelling methodology for the CIFAR style dataset so I asked ChatGPT questions like: "how would I split my file in to train/test splits". This file also attempted to get more context with regard to the face.csv data. I asked questions like "what are these csv labels doing".

 * The next file "How LIME Works" was a relatively early file in our project when we were still considering using LIME for interpretability of the CNN data. We realized quickly that this would be difficult because of the finnickyness of the neurons but I asked questions like: "can you tell me how LIME works", "Say I have a 4-layer CNN that is classifying images based on whether it contains a face or it doesn't contain a face, would I be able to interpret ONLY the first layer of a CNN and then ONLY the second two layers and so on and so forth, effectively am i able to separate the different layers of the CNN in order to see how the model 'learned'", "okay so I have free-viewing monkey fixatation data and I have a CNN architechure that is interpretable on a layer-by-layer basiss would it be possible to go through the similarity of the fixation data and the cnn lime data?"

 * The final file RDM from CNN Layer2 worked through my understanding of RDM's. I asked questions like: 
    * can you give me an example of what an rdm is calculating
    * wait so what happens if the size of the matrices arent the same?
    * what size would the output matrices be?











