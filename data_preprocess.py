# thie file just normalizes the data, filters it, and saves it to a pickle file

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.hongj_preprocess import (
    bandpass_filter, teager_kaiser_energy, lowpass_filter, load_semg_data, interpolate_bad_channels, notch_filter
)


# ─────────── Configuration ───────────
data_folder   = 'Data_KN'                  # root folder of HS*/<gesture>.mat
output_root   = 'data_for_training'     # base output folder
fs            = 2048
window_ma     = int(0.5 * fs)  # 500 ms smoothing for MVC




# Specify bad electrodes per subject and gesture
# Gesture keys must match filenames (without .mat)
bad_electrode_map = {
    'HS1': {
        'closehand': [8],
        'openhand' : [8],
        'point'    : [8],
        'thumb_flex': [8],
        'thumb_ext': [8],
        'wrist_flex': [8],
        'wrist_ext': [8]
    },
    'HS2': {
        'closehand': [],
        'openhand' : [],
        'point'    : [],
        'thumb_flex': [],
        'thumb_ext': [],
        'wrist_flex': [],
        'wrist_ext': []
    },
    'HS3': {
        'closehand': [],
        'openhand' : [],
        'point'    : [],
        'thumb_flex': [],
        'thumb_ext': [],
        'wrist_flex': [],
        'wrist_ext': []
    },
    'HS4': {
        'closehand': [9],
        'openhand' : [],
        'point'    : [9],
        'thumb_flex': [9],
        'thumb_ext': [9],
        'wrist_flex': [9],
        'wrist_ext': []
    },
    'HS5': {
        'closehand': [],
        'openhand' : [],
        'point'    : [],
        'thumb_flex': [9],
        'thumb_ext': [9],
        'wrist_flex': [9],
        'wrist_ext': []
    },
    'HS6': {
        'closehand': [9],
        'openhand' : [],
        'point'    : [9],
        'thumb_flex': [9],
        'thumb_ext': [9],
        'wrist_flex': [9],
        'wrist_ext': []
    },
    'HS7': {
        'closehand': [],
        'openhand' : [],
        'point'    : [],
        'thumb_flex': [],
        'thumb_ext': [],
        'wrist_flex': [],
        'wrist_ext': []
    },
    'HS8': {
        'closehand': [],
        'openhand' : [],
        'point'    : [],
        'thumb_flex': [57],
        'thumb_ext': [],
        'wrist_flex': [],
        'wrist_ext': []
    },
    # add more subjects as needed
}

# ─────────── Helper: moving‐average smoothing ───────────
def moving_average(x, w=window_ma):
    return np.convolve(x, np.ones(w)/w, mode='same')

# ─────────── Load & process ───────────
semg_data     = load_semg_data(data_folder)
processed_data = {}

for subject, gestures in semg_data.items():
    processed_data[subject] = {}

    # --- 1) Process MVC to get per‐channel max ---
    mvc_key = next((g for g in gestures if g.lower().startswith('mvc')), None)
    mvc_max = None
    if mvc_key:
        mvc_raw = gestures[mvc_key]
        bad_ch  = bad_electrode_map.get(subject, {}).get(mvc_key, [])
        if bad_ch:
            mvc_raw = interpolate_bad_channels(mvc_raw, bad_ch)

        # bandpass → notch
        mvc_filt = bandpass_filter(mvc_raw, lowcut=20, highcut=450, fs=fs)
        mvc_filt = notch_filter(mvc_filt, fs=fs, notch_freq=50, q=50)

        # smooth & find max per channel
        mvc_sm = np.vstack([
            moving_average(mvc_filt[ch, :])
            for ch in range(mvc_filt.shape[0])
        ])
        mvc_max = mvc_sm.max(axis=1)
    
    # --- 2) Process & normalize every other gesture ---
    for gesture, raw in gestures.items():
        if gesture.lower().startswith('mvc'):
            continue
        
        bad_ch = bad_electrode_map.get(subject, {}).get(gesture, [])
        emg    = raw.copy()
        if bad_ch:
            emg = interpolate_bad_channels(emg, bad_ch)

        filt  = bandpass_filter(emg, lowcut=20, highcut=450, fs=fs)
        filt  = notch_filter(filt, fs=fs, notch_freq=50, q=50)

        if mvc_max is not None:
            normed = np.zeros_like(filt)
            for ch in range(filt.shape[0]):
                normed[ch, :] = filt[ch, :] / mvc_max[ch] if mvc_max[ch] != 0 else 0
            processed_data[subject][gesture] = normed
        else:
            processed_data[subject][gesture] = filt

# ─────────── 3) Save to disk ───────────
out_path = os.path.join(output_root, 'processed_data.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(processed_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved processed data → {out_path}")