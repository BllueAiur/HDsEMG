import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Paths to the PKL files
processed_path = os.path.join('data_for_training', 'processed_data.pkl')
labels_path = os.path.join('data_for_training', 'labels_data.pkl')

# Load the data
with open(processed_path, 'rb') as f:
    processed_data = pickle.load(f)
with open(labels_path, 'rb') as f:
    labels_data = pickle.load(f)

# Compare dimensions for every subject and gesture
records = []
for subj, gestures in processed_data.items():
    for gesture_name, proc_arr in gestures.items():
        label_arr = labels_data.get(subj, {}).get(gesture_name)
        records.append({
            'Subject': subj,
            'Gesture': gesture_name,
            'Processed Shape': proc_arr.shape,
            'Labels Shape': label_arr.shape if label_arr is not None else None
        })

df = pd.DataFrame(records)
print("Dimension comparison (Processed vs. Labels):")
print(df.to_string(index=False))

# Select first subject and gesture to plot channel 0
sel_subject = df.loc[2, 'Subject']
sel_gesture = df.loc[2, 'Gesture']
channel_idx = 10

proc_signal = processed_data[sel_subject][sel_gesture][channel_idx]
label_mask  = labels_data[sel_subject][sel_gesture][channel_idx]

# Plot
t = np.arange(proc_signal.size)
plt.figure(figsize=(12, 4))
plt.plot(t, proc_signal, label='Processed Signal (ch 0)')
plt.plot(t, label_mask * np.max(proc_signal), linestyle='--', label='Label Mask (scaled)')
plt.title(f"{sel_subject} - {sel_gesture} (Channel {channel_idx})")
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()
plt.show()



import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

# ───────── CONFIG ─────────
DATA_DIR = 'data_for_training'
OUTPUT_IMG_DIR = 'image_examples'
NUM_SAMPLES = 5
FS = 2048
WIN1 = FS  # 1 second

# ───────── HELPERS ─────────
def vector_to_image(vec):
    img = np.zeros((8, 8), dtype=vec.dtype)
    for i, v in enumerate(vec):
        r = 7 - (i % 8)
        c = 7 - (i // 8)
        img[r, c] = v
    return img

# ───────── LOAD DATA ─────────
with open(os.path.join(DATA_DIR, 'segmented_200ms.pkl'), 'rb') as f:
    seg200 = pickle.load(f)
with open(os.path.join(DATA_DIR, 'actual.pkl'), 'rb') as f:
    actual_ds = pickle.load(f)
with open(os.path.join(DATA_DIR, 'dwt.pkl'), 'rb') as f:
    dwt_ds = pickle.load(f)
with open(os.path.join(DATA_DIR, 'dict.pkl'), 'rb') as f:
    dict_ds = pickle.load(f)

# Select a subject and gesture for demonstration
subj0 = sorted(actual_ds.keys())[0]
gest0 = sorted(actual_ds[subj0].keys())[0]
    
# ───────── FIGURE 1: 200ms Windows & 6s Time Series ─────────
fig = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])

# Top row: three 200ms heatmaps
for i in range(3):
    ax = fig.add_subplot(gs[0, i])
    data = seg200[subj0][gest0][i]
    im = ax.imshow(data, aspect='auto')
    ax.set_title(f"{subj0}/{gest0}\n200ms Window #{i+1}", fontsize=10)
    ax.set_xlabel("Sample", fontsize=8)
    ax.set_ylabel("Channel", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)

# Bottom row: 6s channel-0 comparison across all columns
ax_ts = fig.add_subplot(gs[1, :])
t = np.arange(6*FS) / FS
ch_act  = actual_ds[subj0][gest0][0, :6*FS]
ch_dwt  = dwt_ds[subj0][gest0][0, :6*FS]
ch_dict = dict_ds[subj0][gest0][0, :6*FS]

ax_ts.plot(t, ch_act, label='Actual', linewidth=1)
ax_ts.plot(t, ch_dwt, linestyle='--', label='DWT recon', linewidth=1)
ax_ts.plot(t, ch_dict, linestyle=':', label='Dict recon', linewidth=1)
ax_ts.set_title(f"{subj0}/{gest0} Channel 0: First 6 Seconds", fontsize=11)
ax_ts.set_xlabel("Time (s)", fontsize=9)
ax_ts.set_ylabel("Amplitude", fontsize=9)
ax_ts.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
ax_ts.legend(fontsize=8, loc='upper right')

plt.suptitle("Segmentation & Reconstruction Overview", fontsize=12)
plt.show()

# ───────── FIGURE 2: Mid-Point Images for All Gestures ─────────
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# Select subject and gestures
subj0 = sorted(actual_ds.keys())[0]
gestures = sorted(actual_ds[subj0].keys())

# Determine minimum common length
min_len = min(mat.shape[1] for mat in actual_ds[subj0].values())

# Sample frame indices
rng = np.random.default_rng(42)
sample_idxs = rng.choice(min_len, size=NUM_SAMPLES, replace=False)

# ───────── COMPUTE GLOBAL PERCENTILE SCALE ─────────
all_vals = []
for t_idx in sample_idxs:
    for gest in gestures:
        for ds in (actual_ds, dwt_ds, dict_ds):
            all_vals.append(ds[subj0][gest][:, t_idx])
all_vals = np.hstack(all_vals)

vmin = np.percentile(all_vals, 1)   
vmax = np.percentile(all_vals, 99)  

# ───────── PLOT & SAVE WITH CONSISTENT PERCENTILE SCALE ─────────
for t_idx in sample_idxs:
    fig, axes = plt.subplots(len(gestures), 3, 
                             figsize=(10, 3*len(gestures)), 
                             constrained_layout=True)
    
    for row, gest in enumerate(gestures):
        imgs = [
            vector_to_image(actual_ds[subj0][gest][:, t_idx]),
            vector_to_image(dwt_ds[subj0][gest][:, t_idx]),
            vector_to_image(dict_ds[subj0][gest][:, t_idx])
        ]
        titles = ['Actual', 'DWT Recon', 'Dict Recon']
        for col, (img, title) in enumerate(zip(imgs, titles)):
            ax = axes[row, col]
            im = ax.imshow(img, origin='lower', aspect='equal',
                           vmin=vmin, vmax=vmax)
            if row == 0:
                ax.set_title(title, fontsize=11)
            if col == 0:
                ax.set_ylabel(gest, fontsize=9, rotation=0, 
                              labelpad=40, va='center')
            ax.set_xticks([])
            ax.set_yticks([])
            cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                                fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=6)
    
    fig.suptitle(f"{subj0}: Frame {t_idx} ", fontsize=14)
    save_path = os.path.join(OUTPUT_IMG_DIR, f"{subj0}_frame_{t_idx}.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

print(f"Saved {NUM_SAMPLES} composite images with 5th–95th percentile scale to '{OUTPUT_IMG_DIR}/'.")