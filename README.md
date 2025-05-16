# HDsEMG

python version 3.9.12

pre: download data from https://data.4tu.nl/articles/Raw_data_collected_for_the_study_of_Characterization_of_forearm_high-density_electromyograms_during_wrist-hand_tasks_in_individuals_with_Duchenne_Muscular_Dystrophy/12718697
place Data_KN at root folder.

1) remove HS8 from dataset and run data_preprocess.py. 

It cleans and normalizes the raw signals by fixing bad channels, applying band-pass (20–450 Hz) and notch (50 Hz) filters, and scaling most values to within ±20. The cleaned signals are saved as data_for_training/processed_data.pkl.

2) run detect_onset.py. This script reads the filtered data, detects when each gesture starts and stops using the method mentioned in literature, and marks every timepoint as active or static. The resulting labels are stored in data_for_training/labels_data.pkl.

3) run get_training_data.py. It extracts 1s signal segments for each gesture (using the center of long activations,2.5s, and one static window per subject), stacks them into 64×(2048×N) arrays, and reconstructs them in two ways:

    Dictionary learning (dict.pkl): 128 atoms with 16 nonzero coefficients.

    Discrete wavelet transform (dwt.pkl): coif1 wavelet with the 16 largest coefficients.

    locality-constrained linear coding (llc.pkl): 16 knn.


    not important data:
    It also creates overlapping 200 ms windows (with 50 ms hops) from those one‑second segments and saves them in segmented_200ms.pkl. 
    
4) run Example_data_extraction.py to see the result. All output files live in the data_for_training/ directory. 







