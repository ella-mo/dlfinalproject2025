# CSCI1470 Deep Learning Final Project
By Ella, Patrick, and Taha

# Preprocessing
1. image_extractions.py: extracts Stimuli/image_name.png and creates extracted_external_file.csv that holds png file names
2. raw_images_extractions.py: uses extracted_external_file.csv to create raw_images file, which extracts the images from Stimuli.zip 
3. fixation_extractions.py: extracts the fixations and creates fixations_with_image_index_and_path.csv (id,ord_in_trial,start_time,stop_time,trial_id,x,y,image_index,image_path)
4. raster_extractions.py: extracts and preprocesses the rasters to create preprocessed_raster_data.csv
5. face_rois_calculation.py: uses retinaface to categorize each of the extracted images as face vs non face and calculate ROIs. the calculations are held in face_rois_from_retinaface.csv
6. compute_face_selective_neurons: computes face selectivity of neurons using face_rois_from_retinaface.csv and preprocessed_raster_data.csv