# HDsEMG

python version 3.9.12

1) run data_preprocess.py. 

It cleans and normalizes the raw signals by fixing bad channels, applying band-pass (20–450 Hz) and notch (50 Hz) filters, and scaling most values to within ±20. The cleaned signals are saved as data_for_training/processed_data.pkl.

2) run detect_onset.py. This script reads the filtered data, detects when each gesture starts and stops using the method mentioned in literature, and marks every timepoint as active or static. The resulting labels are stored in data_for_training/labels_data.pkl.

3) run get_training_data.py. It extracts 1s signal segments for each gesture (using the center of long activations,2.5s, and one static window per subject), stacks them into 64×(2048×N) arrays, and reconstructs them in two ways:

    Dictionary learning (dict.pkl): 256 atoms with 20 nonzero coefficients.

    Wavelet thresholding (dwt.pkl): db1 wavelet with the 20 largest coefficients.

    It also creates overlapping 200 ms windows (with 50 ms hops) from those one‑second segments and saves them in segmented_200ms.pkl. 
    
4) run Example_data_extraction.py to see the result. All output files live in the data_for_training/ directory. 







