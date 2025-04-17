import os
import numpy as np
import scipy.io
import scipy.signal as signal
import matplotlib.pyplot as plt
from utils import preprocess as pp
import pickle

# Example usage:
data_folder_path = '/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp4/project/FATE/CS_HW/datasets/data/data' 
structured_data = pp.load_semg_data(data_folder_path)

# Now data_structure is a dictionary where each key is a subject folder name and each value is another dictionary
# mapping gesture names to the corresponding sEMG data (64 x N numpy array).
print(structured_data.keys())  # list all subject folder names

del structured_data['HS2']['closehand']

fs = 2048              # Sampling frequency (Hz)
bp_low = 20.0          # Bandpass low cutoff (Hz)
bp_high = 450.0        # Bandpass high cutoff (Hz)
order_bp = 4           # Filter order

nyq = 0.5 * fs
b_bp, a_bp = signal.butter(order_bp, [bp_low/nyq, bp_high/nyq], btype='bandpass')

notch_freq = 50.0      # Notch filter center frequency (Hz)
Q = 50.0               # Quality factor
w0 = notch_freq / nyq
b_notch, a_notch = signal.iirnotch(w0, Q)

window_size = int(fs * 0.2)
window = np.ones(window_size) / window_size

processed_data = {}

for subject, gestures in structured_data.items():
    print(f"Processing subject: {subject} ...")
    processed_data[subject] = {}
    for gesture, data in gestures.items():
        if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == 64:
            processed = np.zeros_like(data)
            for ch in range(data.shape[0]):
                processed[ch, :] = pp.preprocess_signal(data[ch, :],b_bp=b_bp,a_bp=a_bp,b_notch=b_notch,a_notch=a_notch)
            processed_data[subject][gesture] = processed
        else:
            print(f"Skipping {subject} - {gesture}: Data format unexpected.")
    
    if "MVC" in processed_data[subject]:
        mvc_data = processed_data[subject]["MVC"]
        mvc_smoothed = np.zeros_like(mvc_data)
        mvc_max = np.zeros(mvc_data.shape[0])
        for ch in range(mvc_data.shape[0]):
            mvc_smoothed[ch, :] = pp.moving_average(mvc_data[ch, :],window=window)
            mvc_max[ch] = np.max(mvc_smoothed[ch, :])
        processed_data[subject]["MVC_max"] = mvc_max
        for gesture, data in processed_data[subject].items():
            if gesture in ["MVC", "MVC_max"]:
                continue
            for ch in range(data.shape[0]):
                processed_data[subject][gesture][ch, :] = data[ch, :] / mvc_max[ch] if mvc_max[ch] != 0 else 0  
    else:
        print(f"Subject {subject} does not have an 'MVC' gesture. Skipping normalization.")

print('preprocessing finished ')

#### check if data are retrieved correctly
# Select a subject and gesture for demonstration
subject_names = list(processed_data.keys())
selected_subject = subject_names[0]

# Get the list of gesture names for the selected subject
gesture_names = list(processed_data[selected_subject].keys())
selected_gesture = gesture_names[0]

# Retrieve the sEMG data for the selected subject and gesture
semg_data = processed_data[selected_subject][selected_gesture]

# Print details about the subject and its sEMG data
print("Selected Subject:", selected_subject)
print("Available Gestures:", gesture_names)
print("Selected Gesture:", selected_gesture)
print("Shape of sEMG Data (Channels x Time Steps):", semg_data.shape)


# ==========================
# Segmentation Parameters
# ==========================
active_start_offset = 4    # Active segment start time (s)
active_end_offset   = 5    # Active segment end time (s)
rest_start_offset   = 1    # Rest segment start time (s)
rest_end_offset     = 2    # Rest segment end time (s)
cycle_duration      = 6    # Cycle duration (s)

window_duration = 0.2      # Window duration (s)
step_duration   = 0.05     # Step duration (s)
window_samples  = int(window_duration * fs)
step_size       = int(step_duration * fs)
import random
import matplotlib.pyplot as plt

fig_saving_dir = '/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp4/project/FATE/CS_HW/datasets/segment_fig'
os.makedirs(fig_saving_dir, exist_ok=True)
segmented_data = {}
fix_seg_strategy = False
if fix_seg_strategy:
    for subject, gestures in processed_data.items():
        print(f"Segmenting subject: {subject} ...")
        segmented_data[subject] = {}
        segmented_data[subject]["rest"] = []
        
        for gesture, data in gestures.items():
            if gesture in ["MVC", "MVC_max"]:
                continue
            segmented_data[subject].setdefault(gesture, [])
            num_channels, N_samples = data.shape
            n = 0
            
            # segment each cycle (populate segmented_data as before)
            while True:
                active_start = int((active_start_offset + cycle_duration * n) * fs)
                active_end   = int((active_end_offset   + cycle_duration * n) * fs)
                rest_start   = int((rest_start_offset   + cycle_duration * n) * fs)
                rest_end     = int((rest_end_offset     + cycle_duration * n) * fs)
                if active_end > N_samples or rest_end > N_samples:
                    break

                active_segment = data[:, active_start:active_end]
                rest_segment   = data[:, rest_start:rest_end]
                active_windows = pp.segment_window(active_segment, window_samples, step_size)
                rest_windows   = pp.segment_window(rest_segment, window_samples, step_size)
                segmented_data[subject][gesture].extend(active_windows)
                segmented_data[subject]["rest"].extend(rest_windows)

                n += 1

            # if no full cycles, skip plotting
            if n == 0:
                continue

            # --- Plot full-length signal with shaded active/rest regions ---
            sample_chs = random.sample(range(num_channels), 4)
            t_full = np.arange(N_samples) / fs

            fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
            fig.suptitle(f'{subject} | {gesture} — full signal with active/rest highlighted', fontsize=14)

            for i, ch in enumerate(sample_chs):
                axs[i].plot(t_full, data[ch, :], color='black', linewidth=0.8)

                # shade each cycle's active/rest
                for cycle_idx in range(n):
                    a_s = (active_start_offset + cycle_idx * cycle_duration) * fs
                    a_e = (active_end_offset   + cycle_idx * cycle_duration) * fs
                    r_s = (rest_start_offset   + cycle_idx * cycle_duration) * fs
                    r_e = (rest_end_offset     + cycle_idx * cycle_duration) * fs

                    t_a_s, t_a_e = a_s / fs, a_e / fs
                    t_r_s, t_r_e = r_s / fs, r_e / fs

                    axs[i].axvspan(t_a_s, t_a_e, color='red',   alpha=0.3,
                                label='Active' if cycle_idx == 0 else "")
                    axs[i].axvspan(t_r_s, t_r_e, color='blue',  alpha=0.3,
                                label='Rest'   if cycle_idx == 0 else "")

                axs[i].set_ylabel(f'Ch {ch}')
                if i == 0:
                    axs[i].legend(loc='upper right')

            axs[-1].set_xlabel('Time (s)')
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            fig_path = os.path.join(fig_saving_dir, f'{subject}_{gesture}_highlighted.png')
            plt.savefig(fig_path)
            plt.close(fig)
else:
    for subject, gestures in processed_data.items():
        print(f"Segmenting subject: {subject} ...")
        segmented_data[subject] = {}
        segmented_data[subject]["rest"] = []

        for gesture, data in gestures.items():
            if gesture in ["MVC", "MVC_max"]:
                continue
            segmented_data[subject].setdefault(gesture, [])

            # --- New segmentation starts here ---
            emg_data        = data[:, 5*fs:-200]
            filtered_emg    = pp.bandpass_filter(emg_data)
            tkeo_emg        = pp.teager_kaiser_energy(filtered_emg)
            window_size     = int(0.05 * fs)
            smoothed_emg    = pp.moving_average2(tkeo_emg, window_size)
            rectified_emg   = np.abs(smoothed_emg)
            envelope_emg    = pp.lowpass_filter2(rectified_emg)
            envelope_avg    = np.mean(envelope_emg, axis=0)

            # local stats for hysteresis thresholds
            local_window    = int(5.0 * fs)
            kernel          = np.ones(local_window) / local_window
            local_mean      = np.convolve(envelope_avg, kernel, mode='same')
            local_mean_sq   = np.convolve(envelope_avg**2, kernel, mode='same')
            local_std       = np.sqrt(np.maximum(local_mean_sq - local_mean**2, 0))
            k_onset, k_offset = 1.0, 0.5
            thr_on_l        = local_mean + k_onset * local_std
            thr_off_l       = local_mean - k_offset * local_std

            # find active‐onset indices
            on_idx = np.nonzero(envelope_avg > thr_on_l)[0]
            segments = []
            if on_idx.size:
                splits = np.split(on_idx, np.where(np.diff(on_idx) > 1)[0] + 1)
                for grp in splits:
                    onset = grp[0]
                    search_start = grp[-1] + 1
                    offs = np.nonzero(envelope_avg[search_start:] < thr_off_l[search_start:])[0]
                    offset = search_start + (offs[0] if offs.size else len(envelope_avg))
                    segments.append((onset, offset))

            # filter by duration
            min_dur   = int(0.5 * fs)
            max_dur   = int(6.0 * fs)
            filt_segs = [(s, e) for s, e in segments if min_dur <= (e - s) <= max_dur]

            # merge nearby segments
            merge_gap = int(0.75 * fs)
            merged = []
            if filt_segs:
                cur_s, cur_e = filt_segs[0]
                for s, e in filt_segs[1:]:
                    if s - cur_e <= merge_gap:
                        cur_e = e
                    else:
                        merged.append((cur_s, cur_e))
                        cur_s, cur_e = s, e
                merged.append((cur_s, cur_e))
            else:
                merged = filt_segs

            # keep only the longest 10 active segments
            if len(merged) > 10:
                merged = sorted(merged, key=lambda se: se[1] - se[0], reverse=True)[:10]
            # re-sort by onset time
            merged = sorted(merged, key=lambda se: se[0])

            # define rest segments as intervals between active segments
            rest_segs = []
            for i in range(len(merged) - 1):
                r_start = merged[i][1]
                r_end   = merged[i + 1][0]
                if r_end > r_start:
                    rest_segs.append((r_start, r_end))

            # segment and store active parts
            for s, e in merged:
                active_windows = pp.segment_window(emg_data[:, s:e], window_samples, step_size)
                segmented_data[subject][gesture].extend(active_windows)

            # segment and store rest parts
            for s, e in rest_segs:
                rest_windows = pp.segment_window(emg_data[:, s:e], window_samples, step_size)
                segmented_data[subject]["rest"].extend(rest_windows)

            print(f"{gesture}: kept {len(merged)} active segments, {len(rest_segs)} rest segments")

            # optional: visualize envelope with active/rest highlights
            t = np.arange(envelope_avg.shape[0]) / fs
            plt.figure(figsize=(12, 4))
            plt.plot(t, envelope_avg, label='Envelope')
            plt.plot(t, thr_on_l, 'r--', label='Onset thr')
            plt.plot(t, thr_off_l, 'orange', linestyle='--', label='Offset thr')
            for s, e in merged:
                plt.axvspan(s/fs, e/fs, color='green', alpha=0.3, label='Active' if s == merged[0][0] else "")
            for s, e in rest_segs:
                plt.axvspan(s/fs, e/fs, color='blue', alpha=0.2, label='Rest' if s == rest_segs[0][0] else "")
            plt.title(f"{subject} | {gesture}")
            plt.xlabel("Time (s)")
            plt.ylabel("Energy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(fig_saving_dir, f"{subject}_{gesture}_hyst10.png"))
            plt.close()

print("Segmentation complete.")

with open("/home/kasm-user/Desktop/volumes/nfs-hdd-storage/chronos/exp4/project/FATE/CS_HW/datasets/segmented_data.pkl", "wb") as f:
    pickle.dump(segmented_data, f)

print("segmented_data has been saved to 'segmented_data.pkl'.")
for subject, gestures in segmented_data.items():
    print(f"Subject: {subject}")
    for gesture, windows in gestures.items():
        num_windows = len(windows)
        print(f"  Gesture: {gesture} - Number of windows: {num_windows}")
        if num_windows > 0:
            print(f"    Example window shape: {windows[0].shape}")

