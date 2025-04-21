import os
import scipy
import numpy as np
import scipy.io
import scipy.signal as signal
import h5py


def load_semg_data(data_folder):
    """
    Load sEMG data from the given folder into a layered dictionary.
    Structure: {subject_name: {gesture_name: semg_array, ...}, ...}
    
    Args:
        data_folder (str): The path to the Data folder containing subject folders.
        
    Returns:
        dict: A nested dictionary with the sEMG data.
    """
    semg_data = {}
    
    for subject in os.listdir(data_folder):
        subject_path = os.path.join(data_folder, subject)
        if os.path.isdir(subject_path) and subject.startswith('HS'): # for now, only HS subjects
            semg_data[subject] = {}
            
            # each .mat file (gesture)
            for filename in os.listdir(subject_path):
                if filename.endswith('.mat'):
                    gesture_name = os.path.splitext(filename)[0]  # Remove .mat extension
                    file_path = os.path.join(subject_path, filename)
                    try:
                        mat_contents = scipy.io.loadmat(file_path)
                    except:
                        print(file_path, ' has been corrupted')
                        
                    # Each .mat file is assumed to have only one variable (ignore __header__, __version__, __globals__)
                    for key in mat_contents:
                        if not key.startswith('__'):
                            semg_array = mat_contents[key]
                            break
                            
                    # Store the sEMG data under the gesture name for the subject
                    semg_data[subject][gesture_name] = semg_array
                    
    return semg_data

# Specify bad electrodes per subject and gesture
# Gesture keys must match filenames (without .mat)
# bad_electrode_map = {
#     'HS1': {
#         'closehand': [9],
#         'openhand' : [],
#         'point'    : [9],
#         'thumb_flex': [9],
#         'thumb_ext': [9],
#         'wrist_flex': [9],
#         'wrist_ext': []
#     },
#     # add more subjects as needed
# }


def interpolate_bad_channels(emg, bad_idxs):
    """
    emg: (64, T) array
    bad_idxs: list of 1-based channel indices
    """
    n_ch, T = emg.shape
    for idx in bad_idxs:
        # map 1-based idx to (row, col)
        row = 7 - ((idx-1) % 8)
        col = 7 - ((idx-1) // 8)
        neigh_vals = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = row + dr
                if nr < 0 or nr > 7:
                    continue
                nc = (col + dc) % 8
                # invert mapping
                neighbor_idx = (7-nc)*8 + (7-nr) + 1
                if neighbor_idx in bad_idxs:
                    continue
                # collect neighbor's time series
                neigh_vals.append(emg[neighbor_idx-1, :])
        if not neigh_vals:
            continue
        # average neighbors
        emg[idx-1, :] = np.mean(neigh_vals, axis=0)
    return emg


def bandpass_filter(data, lowcut=20, highcut=450, fs=2048, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered

def notch_filter(data, notch_freq=50, q=50, fs=2048):
    # Normalize the notch frequency
    nyq = 0.5 * fs
    w0 = notch_freq / nyq
    # Design IIR notch filter
    b, a = signal.iirnotch(w0, q)
    # Apply filter forward and backward
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered

def lowpass_filter(data, cutoff=5, fs=2048, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low')
    envelope = signal.filtfilt(b, a, data, axis=-1)
    return envelope


def teager_kaiser_energy(data):
    """
    Compute the Teager-Kaiser Energy Operator for each channel.
    For a discrete signal x[n], TKEO is defined as:
        Psi[x(n)] = x[n]^2 - x[n-1]*x[n+1]
    """
    tkeo = np.empty_like(data)
    for i in range(data.shape[0]):
        channel = data[i, :]
        tkeo[i, 1:-1] = channel[1:-1]**2 - channel[:-2] * channel[2:]
        # Handle boundaries by replicating the adjacent value
        tkeo[i, 0] = tkeo[i, 1]
        tkeo[i, -1] = tkeo[i, -2]
    return tkeo

def moving_average(data, window_size):
    """
    Apply a moving average filter along the time axis for each channel.
    """
    kernel = np.ones(window_size) / window_size
    # Apply convolution along the last axis
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=-1, arr=data)