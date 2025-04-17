# HDsEMG

python version 3.12.7

we have segmented_data.pkl contained 8x8 sEMG arrays 200ms windows with 50ms update for each gesture.
The sEMG layout follows the same in paper https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2020.00231/full the screenshot of the layout in paper is provided in layout.png

See DataExtraction.ipynb for example extraction and format of the data. probably need careful tune of hyperparameters.

Unfortunately, failed to reproduce the segmentation method mentioned in papers. Right now, assume the mid-1s window in 3s are the stead state of rest and the corresponding gesture.

# For TCN only

'''
conda create -n emg_tcn python=3.11 \
    cudatoolkit=11.8 cudnn=8.9 -c conda-forge
conda activate emg_tcn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
'''
## Preprocessing
Use dataset_preprocessing.py first, then use dataset_forming.py

## Network training
Run train.py





