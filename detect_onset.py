import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from utils.hongj_preprocess import (
    bandpass_filter, teager_kaiser_energy, lowpass_filter, load_semg_data, interpolate_bad_channels, notch_filter
)


# ─────────── Configuration ───────────
data_folder   = 'Data_KN'                  # root folder of HS*/<gesture>.mat
output_root   = 'segmentation_result'     # base output folder

fs            = 2048
ts_trim       = int(1.0*fs)                    # samples to trim at start/end
ts_trim_end   = int(1.0*fs)
 # Default threshold factors if not specified
default_k_onset, default_k_offset = 10, 2

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

# Specify threshold factors per subject and gesture\ n# Format: 'Subject': { 'gesture': (k_onset, k_offset), ... }
# threshold_map = {
#     'HS2': {
#         'closehand': (12, 3),
#         'openhand' : (10, 2),
#         'point'    : (11, 2),
#         'thumb_flex': (10, 2),
#         'thumb_ext': (10, 2),
#         'wrist_flex': (9,  2),
#         'wrist_ext': (10, 2)
#     },
#     # ... add entries for other subjects
# }
threshold_map = {
    'HS1': {
        'closehand': (20, 10),
        'pointer' :(20, 10),
        'wrist_flex': (20,  5),
    },
    'HS2': {
        'thumb_ext': (3, 1),
        'thumb_flex': (1.75, 0.25),####
        'wrist_ext': (10, 5),
        'wrist_flex': (20, 5),
    },
    'HS3': {
        'closehand': (100, 50), ####
        'pointer': (2.5, 0.75), ####
        
        'thumb_flex': (2.75, 0.75),
        'wrist_ext': (40,10)
    },
    'HS4': {
        'closehand': (10, 6),
        'pointer': (4, 2),
        'thumb_ext': (0.75,0.1), ####
        'thumb_flex': (2,0.5), #########
        
        
    },
    'HS5': {
        'closehand': (3, 1),
        'pointer': (25, 15),
        'thumb_ext': (5,2),
        'thumb_flex': (20,4),
        'wrist_ext': (25,10),
        'wrist_flex': (25,10)
    },
    'HS6': {
        'openhand':(40, 20), ####
        'pointer': (5, 1),
        'thumb_flex': (0.4,-0.5) #####
    },
    'HS7':{
        'closehand': (3, 1),
        'openhand': (15, 5),
        'thumb_flex': (5, 2)
    },
    'HS8': {
        'closehand': (1.2, 0), ######
        'pointer'    : (1.25, 0.5), ########
        'thumb_ext': (3, 0.5), ####
        'thumb_flex': (2.5, 1.75), #abondan
        'wrist_flex': (4,  1),
        'wrist_ext': (4, 0.5)
    },
    # ... add entries for other subjects
}

# ─────────── Main Loop ───────────
semg_data   = load_semg_data(data_folder)
labels_data = {}

# make sure output folders exist
os.makedirs(output_root,         exist_ok=True)
os.makedirs('data_for_training', exist_ok=True)

for subject, gestures in semg_data.items():
    plot_dir   = os.path.join(output_root, subject)
    os.makedirs(plot_dir, exist_ok=True)

    label_dict = {}

    for gesture, raw in gestures.items():
        # skip MVC trials if desired
        if gesture.lower().startswith('mvc'):
            continue

        # raw data shape
        n_chan, orig_len = raw.shape

        # 1) Trim edges for processing
        emg = raw[:, ts_trim:-ts_trim_end]

        # 2) Interpolate bad channels if needed
        bad_list = bad_electrode_map.get(subject, {}).get(gesture, [])
        if bad_list:
            emg = interpolate_bad_channels(emg, bad_list)

        # 3) Filtering & envelope
        filtered = bandpass_filter(emg, lowcut=20, highcut=450, fs=fs)
        filtered = notch_filter(filtered, fs=fs, notch_freq=50, q=50)
        tkeo     = teager_kaiser_energy(filtered)
        rect     = np.abs(tkeo)
        envelope = lowpass_filter(rect, cutoff=10, fs=fs, order=4)
        env_avg  = envelope.mean(axis=0)

        # 4) Compute thresholds
        k_on, k_off = threshold_map.get(subject, {}).get(
            gesture, (default_k_onset, default_k_offset)
        )
        static_mean = env_avg[:2*fs].mean()
        static_std  = env_avg[:2*fs].std()
        thr_on  = static_mean + k_on  * static_std
        thr_off = static_mean + k_off * static_std

        # 5) Hysteresis segmentation
        active_idx = np.nonzero(env_avg > thr_on)[0]
        segments   = []
        if active_idx.size:
            # split into groups of consecutive indices
            breaks = np.where(np.diff(active_idx) > 1)[0] + 1
            for grp in np.split(active_idx, breaks):
                onset = grp[0]
                start = grp[-1] + 1
                offs  = np.nonzero(env_avg[start:] < thr_off)[0]
                offset = (start + offs[0]) if offs.size else len(env_avg)
                segments.append((onset, offset))

        # 6) Filter by duration & merge
        min_dur = int(0.5 * fs)
        max_dur = int(6.0 * fs)
        valid  = [(a, b) for a, b in segments if min_dur <= (b - a) <= max_dur]
        merged = []
        if valid:
            cs, ce = valid[0]
            for a, b in valid[1:]:
                if a - ce <= int(0.75 * fs):
                    ce = b
                else:
                    merged.append((cs, ce))
                    cs, ce = a, b
            merged.append((cs, ce))

        # 7) Build full-length label array
        lab1 = np.zeros(orig_len, dtype=np.uint8)
        for a, b in merged:
            lab1[ts_trim + a : ts_trim + b] = 1

        # tile to 2D to match raw.shape
        lab2 = np.tile(lab1, (n_chan, 1))
        label_dict[gesture] = lab2

        # 8) Diagnostic plot (optional)
        t = np.arange(env_avg.size) / fs
        plt.figure(figsize=(10, 3))
        plt.plot(t, env_avg, label='Envelope (mean TKEO)')
        plt.hlines([thr_on, thr_off], t[0], t[-1],
                   linestyles='--', colors=['r', 'orange'],
                   label=['Onset thr', 'Offset thr'])
        for a, b in merged:
            plt.axvspan(a / fs, b / fs, color='green', alpha=0.25)
        plt.title(f"{subject} → {gesture}")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy (a.u.)")
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(plot_dir, f"{gesture}.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {os.path.join(plot_dir, f'{gesture}.png')}")

    # store this subject’s labels
    labels_data[subject] = label_dict

# ─────────── Save combined labels_data ───────────
out_path = os.path.join('data_for_training', 'labels_data.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(labels_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved full-label dict → {out_path}")