import os
import pickle

# 1) Paths
seg_data_path = "/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp4/project/FATE/CS_HW/datasets/segmented_data.pkl"
output_base  = "/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp4/project/FATE/CS_HW/datasets/segment_files"

# 2) Load the segmented_data dict
with open(seg_data_path, "rb") as f:
    segmented_data = pickle.load(f)

# 3) Build a global label map
all_gestures = sorted({
    gesture
    for subject, gestures in segmented_data.items()
    for gesture in gestures.keys()
})
label_map = {g: idx for idx, g in enumerate(all_gestures)}
print("Label map:", label_map)

# 4) Prepare output directories
os.makedirs(output_base, exist_ok=True)
for gesture in all_gestures:
    os.makedirs(os.path.join(output_base, gesture), exist_ok=True)

# 5) Iterate and save each window into its gesture folder
for subject, gestures in segmented_data.items():
    for gesture, windows in gestures.items():
        lbl = label_map[gesture]
        gesture_dir = os.path.join(output_base, gesture)
        for i, win in enumerate(windows):
            fname = f"{subject}_{gesture}_{i}.pkl"
            outpath = os.path.join(gesture_dir, fname)
            with open(outpath, "wb") as fout:
                # each file contains both the array and its label
                pickle.dump({
                    "data": win,
                    "label": lbl
                }, fout)

print("All segments saved under:", output_base)
