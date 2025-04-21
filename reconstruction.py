#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pywt
from scipy.fftpack import dct, idct
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from utils.hongj_preprocess import bandpass_filter, notch_filter, load_semg_data
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ------------------------- Configuration -------------------------
CONFIG = {
    'DATA_PATH':           'Data_KN',
    'PICKLE_PATH':         'processed_data.pkl',
    'FS':                  2048,
    'BP_CUTOFF':           (20, 450),
    'NOTCH_FREQ':          50,
    'NOTCH_Q':             50,
    'TRAIN_FRAC':          0.01,
    'TEST_SAMPLES':        2000,
    'RNG_SEED':            42,
    'SPARSE_KS':           [8, 12, 16, 20,24, 28,32],
    'WAVELET_FAMILIES':    ['db1', 'coif1', 'bior1.3'],
    'DICT_ATOMS_LIST':     [128, 256],
    'BATCH_SIZE':          512,
    'DICT_INITS':          ['random'],
    'PLOT_SIZE':           (14, 6)
}

# ------------------------- Output Directory -------------------------
OUTPUT_DIR = "image_reconstruction"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ------------------------- Utility Functions -------------------------
def vector_to_image(vec):
    img = np.zeros((8, 8))
    for idx in range(64):
        row = 7 - (idx % 8)
        col = 7 - (idx // 8)
        img[row, col] = vec[idx]
    return img

def dct2d(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def idct2d(coeffs):
    return idct(idct(coeffs.T, norm='ortho').T, norm='ortho')

def compute_nmse(original, reconstructed):
    return np.sum((original - reconstructed)**2) / np.sum(original**2)

def compute_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed)**2)
    if mse == 0:
        return float('inf')
    max_val = np.max(original)
    return 10 * np.log10((max_val**2) / mse)

# ------------------------- Data Loading & Preprocessing -------------------------
def load_and_preprocess_data():
    print("1/4 ▶ Loading & preprocessing data...")
    raw_data = load_semg_data(CONFIG['DATA_PATH'])
    processed = {}
    for subject, gestures in raw_data.items():
        processed[subject] = {}
        for gesture, data in gestures.items():
            filtered_data = np.zeros_like(data)
            for channel in range(data.shape[0]):
                bp = bandpass_filter(data[channel], CONFIG['BP_CUTOFF'][0], CONFIG['BP_CUTOFF'][1], CONFIG['FS'])
                filtered_data[channel] = notch_filter(bp, CONFIG['NOTCH_FREQ'], CONFIG['NOTCH_Q'], CONFIG['FS'])
            processed[subject][gesture] = filtered_data
    return processed

# ------------------------- Dictionary Training -------------------------
def train_dictionaries(training_data, n_atoms):
    print(f"\n▶ Training dictionary with {n_atoms} atoms...")
    dictionaries = {}
    for init_method in CONFIG['DICT_INITS']:
        print(f"Initialization: {init_method}")
        if init_method == 'dct':
            basis = []
            for u in range(8):
                for v in range(8):
                    block = np.eye(8)[u][:, None] @ np.eye(8)[v][None, :]
                    basis.append(idct2d(block).flatten())
            init_dict = np.array(basis)
            if n_atoms > 64:
                init_dict = np.vstack([init_dict, np.random.randn(n_atoms-64, 64)])
        else:
            init_dict = np.random.randn(n_atoms, 64)
        mdl = MiniBatchDictionaryLearning(
            n_components=n_atoms,
            dict_init=init_dict,
            transform_algorithm='omp',
            transform_n_nonzero_coefs=max(CONFIG['SPARSE_KS']),
            batch_size=CONFIG['BATCH_SIZE'],
            fit_algorithm='lars',
            random_state=CONFIG['RNG_SEED'],
            n_jobs=-1
        )
        total = len(training_data)
        batches = int(np.ceil(total / CONFIG['BATCH_SIZE']))
        for b in range(batches):
            start = b * CONFIG['BATCH_SIZE']
            end = start + CONFIG['BATCH_SIZE']
            mdl.partial_fit(training_data[start:end])
            if (b+1) % 10 == 0 or (b+1) == batches:
                print(f"  Batch {b+1}/{batches} complete")
        key = f"DL_{init_method}_{n_atoms}"
        dictionaries[key] = mdl.components_
    return dictionaries

# ------------------------- Evaluation -------------------------
def evaluate_methods(test_frames, dictionaries):
    print("3/4 ▶ Evaluating reconstruction methods...")
    methods = [f"{m}_k{k}" for m in ['DCT']+CONFIG['WAVELET_FAMILIES'] for k in CONFIG['SPARSE_KS']]
    methods += [f"{d}_k{k}" for d in dictionaries.keys() for k in CONFIG['SPARSE_KS']]
    results = {m: {'nmse': [], 'psnr': []} for m in methods}
    for idx, frame in enumerate(test_frames, 1):
        if idx % 100 == 0:
            print(f"   ▸ Processed {idx}/{len(test_frames)} frames")
        # Traditional methods
        for method in ['DCT'] + CONFIG['WAVELET_FAMILIES']:
            if method == 'DCT':
                coeffs = dct2d(frame).flatten()
                recon = lambda c: idct2d(c.reshape(8,8))
            else:
                w = pywt.Wavelet(method)
                lvl = pywt.dwt_max_level(min(frame.shape), w.dec_len)
                clist = pywt.wavedec2(frame, w, level=lvl)
                carr, sl = pywt.coeffs_to_array(clist)
                coeffs = carr.flatten()
                recon = lambda c: pywt.waverec2(pywt.array_to_coeffs(c.reshape(carr.shape), sl, output_format='wavedec2'), w)
            for k in CONFIG['SPARSE_KS']:
                m_key = f"{method}_k{k}"
                idxs = np.argsort(np.abs(coeffs))[::-1][:k]
                mask = np.zeros_like(coeffs, bool)
                mask[idxs] = True
                rec_img = recon(coeffs * mask)
                results[m_key]['nmse'].append(compute_nmse(frame, rec_img))
                results[m_key]['psnr'].append(compute_psnr(frame, rec_img))
        # Dictionary methods
        for dname, atoms in dictionaries.items():
            omp = OrthogonalMatchingPursuit(n_nonzero_coefs=max(CONFIG['SPARSE_KS']))
            omp.fit(atoms.T, frame.flatten())
            full = omp.coef_
            for k in CONFIG['SPARSE_KS']:
                key = f"{dname}_k{k}"
                idxs = np.argsort(np.abs(full))[::-1][:k]
                sp = np.zeros_like(full)
                sp[idxs] = full[idxs]
                rec_img = (atoms.T @ sp).reshape(frame.shape)
                results[key]['nmse'].append(compute_nmse(frame, rec_img))
                results[key]['psnr'].append(compute_psnr(frame, rec_img))
    final = {}
    for m, v in results.items():
        final[m] = {
            'nmse_mean': np.mean(v['nmse']),
            'nmse_std': np.std(v['nmse']),
            'psnr_mean': np.mean(v['psnr']),
            'psnr_std': np.std(v['psnr'])
        }
    return final

# ------------------------- Plotting -------------------------
def plot_performance_comparison(results):
    print("\nPerformance comparison values:")
    grouping = defaultdict(list)
    for key, vals in results.items():
        base, k_str = key.rsplit('_k', 1)
        grouping[base].append((int(k_str), vals['nmse_mean'], vals['psnr_mean']))

    for base, lst in grouping.items():
        lst = sorted(lst, key=lambda x: x[0])
        ks  = [x[0] for x in lst]
        nm  = [round(x[1],4) for x in lst]
        ps  = [round(x[2],2) for x in lst]
        print(f"{base}: ks={ks}\n  NMSE means={nm}\n  PSNR means={ps}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=CONFIG['PLOT_SIZE'])
    styles = {
        'DCT': '--', 'db1': '-.', 'coif1': ':', 'bior1.3': (0,(3,1,1,1)),
        **{f"DL_random_{n}": '-' for n in CONFIG['DICT_ATOMS_LIST']}
    }
    markers = ['o','s','D','^','v']
    colors  = ['#1f77b4','#2ca02c','#d62728','#9467bd','#8c564b']

    for idx, (mth, style) in enumerate(styles.items()):
        if mth not in grouping:
            continue
        lst = sorted(grouping[mth], key=lambda x: x[0])
        ks  = [x[0] for x in lst]
        nm  = [x[1] for x in lst]
        ps  = [x[2] for x in lst]
        ax1.plot(ks, nm, label=mth, linestyle=style, marker=markers[idx%len(markers)], color=colors[idx%len(colors)])
        ax2.plot(ks, ps, label=mth, linestyle=style, marker=markers[idx%len(markers)], color=colors[idx%len(colors)])

    ax1.set(xlabel='Sparsity Level (k)', ylabel='NMSE')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='best')

    ax2.set(xlabel='Sparsity Level (k)', ylabel='PSNR (dB)')
    ax2.grid(alpha=0.3)
    ax2.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'linear_performance_comparison.png'), bbox_inches='tight')
    plt.show()



def plot_all_reconstructions(test_frame, dictionaries):
    methods = ['DCT'] + CONFIG['WAVELET_FAMILIES'] + list(dictionaries.keys())
    n_rows = len(methods) + 1
    n_cols = len(CONFIG['SPARSE_KS']) + 1
    fig = plt.figure(figsize=(2*n_cols, 2*n_rows))
    plt.suptitle("Reconstruction Comparison Across Methods and Sparsity Levels", y=1.02)

    # Original
    ax = fig.add_subplot(n_rows, n_cols, 1)
    ax.imshow(test_frame, cmap='viridis')
    ax.set_title("Original")
    ax.axis('off')

    # Column headers
    for i, k in enumerate(CONFIG['SPARSE_KS'], start=1):
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.set_title(f"k={k}", fontsize=8)
        ax.axis('off')

    # Rows
    for r, method in enumerate(methods, start=1):
        # Method label
        ax = fig.add_subplot(n_rows, n_cols, r*n_cols + 1)
        ax.text(0.5, 0.5, method.replace('_', '\n'), ha='center', va='center', fontsize=8)
        ax.axis('off')
        for c, k in enumerate(CONFIG['SPARSE_KS'], start=1):
            ax = fig.add_subplot(n_rows, n_cols, r*n_cols + c + 1)
            # Compute coefficients
            if method == 'DCT':
                coeffs = dct2d(test_frame).flatten()
                recon_fn = lambda v: idct2d(v.reshape(8,8))
            elif method in CONFIG['WAVELET_FAMILIES']:
                w = pywt.Wavelet(method)
                lvl = pywt.dwt_max_level(min(test_frame.shape), w.dec_len)
                clist = pywt.wavedec2(test_frame, w, level=lvl)
                carr, sl = pywt.coeffs_to_array(clist)
                coeffs = carr.flatten()
                recon_fn = lambda v: pywt.waverec2(
                    pywt.array_to_coeffs(v.reshape(carr.shape), sl, output_format='wavedec2'), w)
            else:
                atoms = dictionaries[method]
                omp  = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
                omp.fit(atoms.T, test_frame.flatten())
                coeffs = omp.coef_
                # CORRECT: reshape the flattened image back to 8×8
                recon_fn = lambda v, atoms=atoms: (atoms.T @ v).reshape(8, 8)
                
            # Threshold
            idxs = np.argsort(np.abs(coeffs))[::-1][:k]
            mask = np.zeros_like(coeffs, bool)
            mask[idxs] = True
            rec = recon_fn(coeffs * mask)

            ax.imshow(rec, cmap='viridis')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'full_reconstruction_comparison.png'), bbox_inches='tight')
    plt.show()

# ------------------------- Main -------------------------
if __name__ == "__main__":
    # Load or preprocess data
    if os.path.exists(CONFIG['PICKLE_PATH']):
        with open(CONFIG['PICKLE_PATH'], 'rb') as f:
            data = pickle.load(f)
    else:
        data = load_and_preprocess_data()
        with open(CONFIG['PICKLE_PATH'], 'wb') as f:
            pickle.dump(data, f)

    # Prepare vectors
    rng = np.random.default_rng(CONFIG['RNG_SEED'])
    all_vecs = []
    for sub in data.values():
        for mat in sub.values():
            all_vecs.extend(mat.T)
    idxs = rng.permutation(len(all_vecs))
    n_train = int(CONFIG['TRAIN_FRAC'] * len(all_vecs))

    X_train = np.vstack([vector_to_image(all_vecs[i]).flatten() for i in idxs[:n_train]])
    test_frames = [vector_to_image(all_vecs[i]) for i in idxs[n_train:n_train+CONFIG['TEST_SAMPLES']]]

    # Train dictionaries
    dicts = {}
    for na in CONFIG['DICT_ATOMS_LIST']:
        dicts.update(train_dictionaries(X_train, na))

    # Evaluate
    results = evaluate_methods(test_frames, dicts)
    plot_performance_comparison(results)
    plot_all_reconstructions(test_frames[0], dicts)


