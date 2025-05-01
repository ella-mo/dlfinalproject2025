This folder contains the transcripts of AI. The transcripts can seen by opening the .mhtml files in your preferred browser.

Ella:

Accessing Data in a Repository - Claude.mhtml: This script was used to download and explore the data. Due to the size of the data, locally downloading the zip files was sub-optimal. Moreover, the type of file, .nwb, was a type that I had never encountered before, so I needed help exporting the data contained in these files to .csv files.

Extract fixations from NWB.mhtml: This script was used after trying for several days to understand the structure of the files and how data from the same trial mapped into different files. I had never worked with .nwb files, so exporting the data was something that I needed help with. 

Deep Learning Final 2025 - Dimensionality Reduction for Grouping.mhtml: This script was supposed to help me map the fixation data to the raster data, due to the sheer size of the data. However, there was a discrepancy of the interpretations of the units, so this transcript didn't end up very useful. 

Deep Learning Final 2025 - Dynamic NWB File Access.mhtml: This transcript is an extension of Dimensionality Reduction for Grouping. The transcript wrote neuron_activations/main.py so as to take in the bank_array_regions.csv and do the necessary preprocessing for the files. The size of the files made the task difficult. 

Deep Learning Final 2025 - Extract rasters from NWB.mhtml: This transcript was used to extract data from .nwb files, which I had never worked with before. The majority of the transcript contains attempts at finding ROIs and FSIs, which we ultimately pivoted away from. 

Deep Learning Final 2025 - Path Navigation Correction.mhtml: After the general logic of the code in /neuron_activations was written, in order to run the code in OSCAR for the AIT/CIT recorded areas, I needed to figure out what the path navigations were. This also helped me make sure that the logic for data extraction from the .nwb files was sound. Moreover, due to the size of the files, especially the raster data, I needed a way to make the extraction and preprocessing faster, such as using .npy files instead.

Deep Learning Final 2025 - Raster Extraction and Normalization.mhtml: This transcript helped me get the original logic fo the data extraction from the .nwb files before ensuring the logic and path navigation was correct in Path Navigation Correction. 

Dynamic CSV Path Generation.mhtml: I wanted to save the extracted data in a dynamic way. Then, after running it in OSCAR, we realized we needed to re-do our logic to load the top 10 modules into the HMO model. 

NWB file contents.mhtml: This was for original data exploration. I didn't realize both Stimuli.zip and Stimuli.z01 were needed to open the files. Moreover, I didn't actually want to unzip them due to the size of the files. I used this to access the data without overloading my computer's memory.