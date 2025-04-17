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

def preprocess_signal(sig,b_bp,a_bp,b_notch,a_notch):
    """Apply bandpass, notch filtering, then full-wave rectification."""
    filtered = signal.filtfilt(b_bp, a_bp, sig)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    return np.abs(filtered)

def moving_average(sig,window):
    """Apply moving average with a 200 ms window."""
    return np.convolve(sig, window, mode='same')

def bandpass_and_notch(sig, fs,bp_low,bp_high,bp_order,notch_freq,notch_Q):
    """
    Apply a bandpass filter (4th order Butterworth, 20–450 Hz) 
    and a notch filter (2nd order IIR at 50 Hz, Q=50) to the signal.
    """
    nyq = 0.5 * fs
    lowcut = bp_low / nyq
    highcut = bp_high / nyq
    b_bp, a_bp = signal.butter(bp_order, [lowcut, highcut], btype='bandpass')
    filtered = signal.filtfilt(b_bp, a_bp, sig)
    
    # Notch filter design
    w0 = notch_freq / nyq
    b_notch, a_notch = signal.iirnotch(w0, notch_Q)
    filtered = signal.filtfilt(b_notch, a_notch, filtered)
    return filtered

def apply_TKEO(sig):
    """
    Apply the Teager–Kaiser Energy Operator (TKEO) on a 1D signal.
    Uses zero padding at the boundaries.
    """
    padded = np.concatenate(([0], sig, [0]))
    tkeo = padded[1:-1]**2 - padded[0:-2]*padded[2:]
    return tkeo

lp_order = 3           
lp_cutoff = 50.0

def lowpass_filter(sig, fs, cutoff=lp_cutoff, order=lp_order):
    """Apply a low-pass Butterworth filter."""
    nyq = 0.5 * fs
    normalized_cutoff = cutoff / nyq
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    return signal.filtfilt(b, a, sig)

def process_channel(sig, fs, h,baseline_duration):
    """
    Process a single channel signal by:
      1. Filtering with bandpass and notch filters.
      2. Applying TKEO.
      3. Full-wave rectification.
      4. Low-pass filtering.
      5. Computing threshold T from the baseline (first 2 s).
    Returns the processed signal and its threshold.
    """
    filtered = bandpass_and_notch(sig, fs)
    tkeo = apply_TKEO(filtered)
    rectified = np.abs(tkeo)
    processed = lowpass_filter(rectified, fs)
    
    baseline_samples = int(baseline_duration * fs)
    if len(processed) >= 3 * baseline_samples:
        baseline = processed[-2*baseline_samples:-baseline_samples]
    else:
        baseline = processed
    mu = np.mean(baseline)
    std = np.std(baseline)
    T = mu + h * std
    return processed, T

window_length = 50     
channel_percent = 0.05

def aggregated_activity(processed_data, thresholds, window_length=window_length, channel_percent=channel_percent):
    """
    Compute an aggregated binary time sequence for a gesture.
    At each time index, if at least ceil(channel_percent*num_channels) channels have
    50 consecutive samples (starting at that index, padded if necessary) above their threshold,
    mark that time index as active (1); otherwise, inactive (0).
    """
    num_channels, N = processed_data.shape
    required_channels = int(np.ceil(channel_percent * num_channels))
    
    binary_seq = np.zeros(N, dtype=int)
    for i in range(N):
        # Determine the window: if not enough samples, pad with the last sample's value for each channel
        if i + window_length <= N:
            window_data = processed_data[:, i:i+window_length]
        else:
            pad_length = i + window_length - N
            window_data = np.concatenate((processed_data[:, i:], 
                                          np.tile(processed_data[:, -1:], (1, pad_length))), axis=1)
        # For each channel, check if all values in the window exceed the channel's threshold
        active_channels = np.sum(np.all(window_data > thresholds[:, None], axis=1))
        if active_channels >= required_channels:
            binary_seq[i] = 1
    return binary_seq

def consecutive_segments(binary_signal):
    """
    Compute consecutive segment lengths for a binary signal.
    Returns a list of tuples (segment_length, value).
    """
    segments = []
    if len(binary_signal) == 0:
        return segments
    current_val = binary_signal[0]
    count = 1
    for val in binary_signal[1:]:
        if val == current_val:
            count += 1
        else:
            segments.append((count, current_val))
            current_val = val
            count = 1
    segments.append((count, current_val))
    return segments

def segment_window(segment, window_samples, step):
    """Segment a 2D array (channels x L) into overlapping windows."""
    windows = []
    L = segment.shape[1]
    for start in range(0, L - window_samples + 1, step):
        windows.append(segment[:, start:start + window_samples])
    return windows

def bandpass_filter(data, lowcut=20, highcut=450, fs=2048, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered

def lowpass_filter2(data, cutoff=5, fs=2048, order=4):
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

def moving_average2(data, window_size):
    """
    Apply a moving average filter along the time axis for each channel.
    """
    kernel = np.ones(window_size) / window_size
    # Apply convolution along the last axis
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=-1, arr=data)