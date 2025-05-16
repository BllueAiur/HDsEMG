#!/usr/bin/env python3
"""
HD-sEMG Reconstruction & Separability Benchmark

This script compares the following methods for 8×8 block reconstruction and
class separability on HD-sEMG data:
  - DCT-based sparsification
  - Wavelet-based sparsification (bior1.3, db1, coif1)
  - MiniBatchDictionaryLearning + OMP sparse coding
  - Locality-constrained Linear Coding (LLC)

The single sparsity parameter `k` (from CONFIG['SPARSE_KS']) controls:
  • number of retained coefficients in DCT and wavelet transforms
  • number of nonzero atoms in OMP codes
  • number of nearest atoms (knn) and nonzeros in LLC

Outputs:
  • Reconstruction quality (NMSE, PSNR) vs k plots
  • Global Fisher separability vs k plots
  • Full grid of reconstructed blocks
"""
import os
import pickle
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pywt
from scipy.fftpack import dct, idct
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.preprocessing import StandardScaler

from utils.hongj_preprocess import bandpass_filter, notch_filter, load_semg_data
from utils.LLC import MultiClassLLC

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ───────────────────────── Configuration ──────────────────────────
CONFIG = {
    "DATA_PATH": "Data_KN",
    "PICKLE_PATH": "processed_data.pkl",
    "FS": 2048,
    "BP_CUTOFF": (20, 450),
    "NOTCH_FREQ": 50,
    "NOTCH_Q": 50,
    "TRAIN_FRAC": 0.02,
    "TEST_SAMPLES": 2500,
    "RNG_SEED": 19,
    "SPARSE_KS": [8,10,12,14,16,18,20,22,24],
    "WAVELET_FAMILIES": ["bior1.3", "db1", "coif1"],
    "DICT_ATOMS_LIST": [128],
    "BATCH_SIZE": 1024,
    "PLOT_SIZE": (14, 6),
    # LLC hyperparameters
    "LLC_BETA": 1e-3,
    "LLC_WORDS_PER_CLASS": 16
}

OUTPUT_DIR = "image_reconstruction"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ───────────────────────── Helper Functions ───────────────────────

def vector_to_image(vec: np.ndarray) -> np.ndarray:
    """Convert a 64-element vector to an 8×8 image."""
    img = np.zeros((8, 8))
    for i in range(64):
        row = 7 - (i % 8)
        col = 7 - (i // 8)
        img[row, col] = vec[i]
    return img


def dct2d(img: np.ndarray) -> np.ndarray:
    """2D DCT with orthonormal normalization."""
    return dct(dct(img.T, norm="ortho").T, norm="ortho")


def idct2d(coefs: np.ndarray) -> np.ndarray:
    """2D inverse DCT with orthonormal normalization."""
    return idct(idct(coefs.T, norm="ortho").T, norm="ortho")


def compute_nmse(orig: np.ndarray, rec: np.ndarray) -> float:
    """Normalized mean squared error."""
    return np.sum((orig - rec) ** 2) / np.sum(orig ** 2)


def compute_psnr(orig: np.ndarray, rec: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB."""
    mse = np.mean((orig - rec) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(np.max(orig) ** 2 / mse)


def global_fisher(X: np.ndarray, y: np.ndarray) -> float:
    """
    Multivariate Fisher ratio = trace(S_B) / trace(S_W).  S_B: between-class,
    S_W: within-class scatter.
    """
    mu = X.mean(axis=0)
    SB = np.zeros((X.shape[1], X.shape[1]))
    SW = np.zeros_like(SB)
    for label in np.unique(y):
        Xc = X[y == label]
        muc = Xc.mean(axis=0)
        SB += Xc.shape[0] * np.outer(muc - mu, muc - mu)
        SW += (Xc - muc).T @ (Xc - muc)
    return np.trace(SB) / np.trace(SW)


# ───────────────────────── Data Loading ──────────────────────────

def load_and_preprocess_data() -> dict:
    """
    Load raw HD-sEMG data, apply bandpass and notch filters per channel.
    Returns dict of dicts: subject → gesture → filtered matrix.
    """
    raw = load_semg_data(CONFIG["DATA_PATH"])
    processed = {}
    for subj, gests in raw.items():
        processed[subj] = {}
        for gest, mat in gests.items():
            filt = np.zeros_like(mat)
            for ch in range(mat.shape[0]):
                bp = bandpass_filter(
                    mat[ch], CONFIG["BP_CUTOFF"][0], CONFIG["BP_CUTOFF"][1], CONFIG["FS"]
                )
                filt[ch] = notch_filter(
                    bp, CONFIG["NOTCH_FREQ"], CONFIG["NOTCH_Q"], CONFIG["FS"]
                )
            processed[subj][gest] = filt
    return processed


# ───────────────────────── Reconstruction Methods ─────────────────

def reconstruct_dct_block(block: np.ndarray, k: int) -> np.ndarray:
    coeffs = dct2d(block).flatten()
    idxs = np.argsort(np.abs(coeffs))[::-1][:k]
    mask = np.zeros_like(coeffs, bool)
    mask[idxs] = True
    return idct2d((coeffs * mask).reshape(8, 8))


def reconstruct_wavelet_block(
    block: np.ndarray, family: str, k: int
) -> np.ndarray:
    w = pywt.Wavelet(family)
    lvl = pywt.dwt_max_level(8, w.dec_len)
    coeffs_list = pywt.wavedec2(block, w, level=lvl)
    arr, slices = pywt.coeffs_to_array(coeffs_list)
    flat = arr.flatten()
    idxs = np.argsort(np.abs(flat))[::-1][:k]
    mask = np.zeros_like(flat, bool)
    mask[idxs] = True
    filtered = (flat * mask).reshape(arr.shape)
    return pywt.waverec2(pywt.array_to_coeffs(filtered, slices, output_format='wavedec2'), w)


def reconstruct_dict_block(
    block: np.ndarray, atoms: np.ndarray, k: int
) -> np.ndarray:
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k)
    omp.fit(atoms.T, block.flatten())
    coef = omp.coef_
    idxs = np.argsort(np.abs(coef))[::-1][:k]
    mask = np.zeros_like(coef, bool)
    mask[idxs] = True
    return (atoms.T @ (coef * mask)).reshape(8, 8)


def reconstruct_llc_block(
    block: np.ndarray, atoms: np.ndarray, k: int, beta: float
) -> tuple:
    """
    LLC reconstruction: solves weights over k nearest atoms.
    Returns reconstructed block and full weight vector.
    """
    f = block.flatten()[None, :]
    dist2 = (np.sum(f**2) - 2 * f @ atoms.T + np.sum(atoms**2, axis=1)).ravel()
    neigh = np.argsort(dist2)[:k]
    Z = atoms[neigh] - f  # shape (k, D)
    C = Z @ Z.T
    C += beta * max(np.trace(C), 1e-12) * np.eye(k)
    w_nn, *_ = np.linalg.lstsq(C, np.ones(k), rcond=None)
    w_nn /= w_nn.sum() + 1e-12
    full_w = np.zeros(atoms.shape[0])
    full_w[neigh] = w_nn
    rec = (atoms.T @ full_w).reshape(8, 8)
    return rec, full_w


# ───────────────────────── Training Routines ──────────────────────

def train_dictionary(
    X_train: np.ndarray, n_atoms: int
) -> dict:
    """Train a MiniBatchDictionaryLearning model."""
    mdl = MiniBatchDictionaryLearning(
        n_components=n_atoms,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=max(CONFIG['SPARSE_KS']),
        batch_size=CONFIG['BATCH_SIZE'],
        random_state=CONFIG['RNG_SEED'],
        n_jobs=-1
    )
    mdl.fit(X_train)
    return {f"DL_random_{n_atoms}": mdl.components_}


def train_llc_vocab(
    X_train: np.ndarray, y_train: np.ndarray
) -> tuple:
    """Train per-class KMeans codebooks for LLC and stack them."""
    feats_by_cls = defaultdict(list)
    for feat, label in zip(X_train, y_train):
        feats_by_cls[label].append(feat)
    feature_list = [np.vstack(v) for v in feats_by_cls.values()]
    class_list = list(feats_by_cls.keys())

    llc_model = MultiClassLLC(
        knn=max(CONFIG['SPARSE_KS']),
        beta=CONFIG['LLC_BETA'],
        per_class_words=CONFIG['LLC_WORDS_PER_CLASS']
    )
    llc_model.fit(feature_list, class_list)
    atoms = np.vstack([llc_model.dictionaries[cls] for cls in class_list])
    return llc_model, atoms


# ───────────────────────── Evaluation: Reconstruction ─────────────

def evaluate_reconstruction(
    test_frames: list, dictionaries: dict, llc_atoms: np.ndarray
) -> dict:
    stats = defaultdict(lambda: {'nmse': [], 'psnr': []})
    methods = ['DCT', *CONFIG['WAVELET_FAMILIES'], *dictionaries.keys(), 'LLC']
    for frame in test_frames:
        for method in methods:
            for k in CONFIG['SPARSE_KS']:
                key = f"{method}_k{k}"
                if method == 'DCT':
                    rec = reconstruct_dct_block(frame, k)
                elif method in CONFIG['WAVELET_FAMILIES']:
                    rec = reconstruct_wavelet_block(frame, method, k)
                elif method == 'LLC':
                    rec, _ = reconstruct_llc_block(frame, llc_atoms, k, CONFIG['LLC_BETA'])
                else:
                    rec = reconstruct_dict_block(frame, dictionaries[method], k)

                stats[key]['nmse'].append(compute_nmse(frame, rec))
                stats[key]['psnr'].append(compute_psnr(frame, rec))

    return {
        m: {
            'nmse_mean': np.mean(v['nmse']),
            'psnr_mean': np.mean(v['psnr'])
        }
        for m, v in stats.items()
    }


# ───────────────────────── Evaluation: Separability ──────────────

def evaluate_separability(
    data: dict, dictionaries: dict, llc_atoms: np.ndarray
) -> dict:
    rng = np.random.default_rng(CONFIG['RNG_SEED'])
    gestures = list(next(iter(data.values())).keys())

    frames, labels = [], []
    for label, gesture in enumerate(gestures):
        for subj in data.values():
            mat = subj[gesture]
            idxs = rng.choice(mat.shape[1], 1000, replace=False)
            frames.append(mat[:, idxs].T)
            labels.extend([label] * 1000)

    X_orig = np.vstack(frames)
    y = np.array(labels)

    abs_sep = defaultdict(dict)
    methods = ['Original', 'DCT', *CONFIG['WAVELET_FAMILIES'], *dictionaries.keys(), 'LLC']

    total = len(methods) * len(CONFIG['SPARSE_KS'])
    counter = 0
    print(f"▶ Separability tasks: {total}")

    for method in methods:
        for k in CONFIG['SPARSE_KS']:
            counter += 1
            print(f" [{counter}/{total}] {method}, k={k}")

            if method == 'Original':
                Xr = X_orig.copy()
            elif method == 'DCT':
                Xr = np.array([
                    reconstruct_dct_block(f.reshape(8, 8), k).flatten()
                    for f in X_orig
                ])
            elif method in CONFIG['WAVELET_FAMILIES']:
                Xr = np.array([
                    reconstruct_wavelet_block(f.reshape(8, 8), method, k).flatten()
                    for f in X_orig
                ])
            elif method == 'LLC':
                Xr = np.array([
                    reconstruct_llc_block(f.reshape(8, 8), llc_atoms, k, CONFIG['LLC_BETA'])[0].flatten()
                    for f in X_orig
                ])
            else:
                Xr = np.array([
                    reconstruct_dict_block(f.reshape(8, 8), dictionaries[method], k).flatten()
                    for f in X_orig
                ])

            Xs = StandardScaler().fit_transform(Xr)
            abs_sep[method][k] = global_fisher(abs(Xs), y)

    print("Original Fisher scores:")
    for k in CONFIG['SPARSE_KS']:
        print(f" k={k}: {abs_sep['Original'][k]}")

    rel = {
        m: {
            k: abs_sep[m][k] / abs_sep['Original'][k]
            for k in CONFIG['SPARSE_KS']
        }
        for m in methods
    }
    return rel


# ───────────────────────── Plotting Functions ─────────────────────

def plot_performance_comparison(results: dict) -> None:
    """Plot NMSE and PSNR means vs k for all methods, with discrete k, larger text, and legend at bottom."""
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import os

    grouping = defaultdict(list)
    for key, vals in results.items():
        method, k_str = key.split('_k')
        grouping[method].append((int(k_str), vals['nmse_mean'], vals['psnr_mean']))

    print("Performance (NMSE / PSNR means):")
    for method, lst in grouping.items():
        lst = sorted(lst)
        ks = [x[0] for x in lst]
        nm = [x[1] for x in lst]
        ps = [x[2] for x in lst]
        print(f"{method}: ks={ks} NMSE={nm} PSNR={ps}")

    # Use a slightly taller figure to better fit big fonts and the legend
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8) if CONFIG['PLOT_SIZE'][1] < 7 else CONFIG['PLOT_SIZE'])

    styles = {'DCT': '--', 'bior1.3': '-.', 'db1': ':', 'coif1': (0, (3, 1, 1, 1))}
    styles.update({f'DL_random_{n}': '-' for n in CONFIG['DICT_ATOMS_LIST']})
    styles['LLC'] = '--'
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p']
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    legend_handles = []
    legend_labels = []

    for i, (method, lst) in enumerate(grouping.items()):
        lst = sorted(lst)
        ks = [x[0] for x in lst]
        nm = [x[1] for x in lst]
        ps = [x[2] for x in lst]

        handle1, = ax1.plot(
            ks, nm,
            label=method,
            linestyle=styles.get(method, '-'),
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            markersize=7.5
        )
        ax2.plot(
            ks, ps,
            label=method,
            linestyle=styles.get(method, '-'),
            marker=markers[i % len(markers)],
            color=colors[i % len(colors)],
            markersize=7.5
        )
        legend_handles.append(handle1)
        legend_labels.append(method)

    ax1.set(xlabel='Sparsity k', ylabel='NMSE')
    ax2.set(xlabel='Sparsity k', ylabel='PSNR (dB)')

    for ax in (ax1, ax2):
        ax.tick_params(axis='both', which='major', labelsize=22)
        ax.xaxis.label.set_size(22)
        ax.yaxis.label.set_size(22)
        ax.grid(alpha=0.3)
        ks_all = sorted(set(k for lst in grouping.values() for k, _, _ in lst))
        ax.set_xticks(ks_all)

    # Adjust layout to reserve extra space at bottom for large legend
    plt.subplots_adjust(bottom=0.32)
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(len(legend_labels), 5), fontsize=20, frameon=False
    )

    plt.tight_layout(rect=[0, 0.18, 1, 1])  # more room for legend
    plt.savefig(os.path.join(OUTPUT_DIR, 'linear_performance_comparison.png'))
    plt.show()


def plot_separability(sep: dict, normalize: bool = True) -> None:
    """Plot Fisher separability vs k, with discrete k, larger text, and legend at bottom."""
    import matplotlib.pyplot as plt
    import os

    ref = sep['Original']
    fig, ax = plt.subplots(figsize=(14, 8) if CONFIG['PLOT_SIZE'][1] < 7 else CONFIG['PLOT_SIZE'])

    legend_handles = []
    legend_labels = []

    for i, (method, scores) in enumerate(sep.items()):
        ks = sorted(scores)
        vals = [scores[k] / ref[k] if normalize else scores[k] for k in ks]
        handle, = ax.plot(ks, vals, marker='o', label=method, markersize=7.5)
        legend_handles.append(handle)
        legend_labels.append(method)

    ylabel = 'Relative Fisher (Original=1)' if normalize else 'Fisher Global'
    ax.set(xlabel='Sparsity k', ylabel=ylabel)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.xaxis.label.set_size(22)
    ax.yaxis.label.set_size(22)
    ax.grid(alpha=0.3)
    ks_all = sorted(set(k for scores in sep.values() for k in scores))
    ax.set_xticks(ks_all)

    plt.subplots_adjust(bottom=0.28)
    fig.legend(
        legend_handles, legend_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(len(legend_labels), 5), fontsize=20, frameon=False
    )

    plt.tight_layout(rect=[0, 0.16, 1, 1])
    fn = 'sep_rel.png' if normalize else 'sep_abs.png'
    plt.savefig(os.path.join(OUTPUT_DIR, fn))
    plt.show()


def plot_all_reconstructions(
    test_frame: np.ndarray,
    dictionaries: dict,
    llc_atoms: np.ndarray
) -> None:
    """Display a grid of original + reconstructions for each method and k."""
    methods = ['Original', 'DCT'] + CONFIG['WAVELET_FAMILIES'] + list(dictionaries.keys()) + ['LLC']
    n_rows = len(methods)
    n_cols = len(CONFIG['SPARSE_KS'])
    fig, axes = plt.subplots(
        n_rows, n_cols + 1,
        figsize=(2 * (n_cols + 1), 2 * n_rows)
    )
    plt.suptitle("Reconstruction Comparison Across Methods and Sparsity Levels", y=1.02)

    # Labels and original
    for i, method in enumerate(methods):
        ax = axes[i, 0]
        if method == 'Original':
            ax.imshow(test_frame, cmap='viridis')
            ax.set_title('Original')
        else:
            ax.axis('off')
            ax.text(0.5, 0.5, method, ha='center', va='center', fontsize=8)
        ax.axis('off')

    # Column titles k
    for j, k in enumerate(CONFIG['SPARSE_KS'], start=1):
        axes[0, j].set_title(f'k={k}', fontsize=8)

    # Reconstruction panels
    for i, method in enumerate(methods):
        for j, k in enumerate(CONFIG['SPARSE_KS'], start=1):
            ax = axes[i, j]
            if method == 'Original':
                rec = test_frame
            elif method == 'DCT':
                rec = reconstruct_dct_block(test_frame, k)
            elif method in CONFIG['WAVELET_FAMILIES']:
                rec = reconstruct_wavelet_block(test_frame, method, k)
            elif method == 'LLC':
                rec, _ = reconstruct_llc_block(test_frame, llc_atoms, k, CONFIG['LLC_BETA'])
            else:
                rec = reconstruct_dict_block(test_frame, dictionaries[method], k)

            ax.imshow(rec, cmap='viridis')
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, 'full_reconstruction_comparison.png'),
        bbox_inches='tight'
    )
    plt.show()


# ───────────────────────── Main Pipeline ──────────────────────────
if __name__ == '__main__':
    # Load or preprocess data
    if os.path.exists(CONFIG['PICKLE_PATH']):
        with open(CONFIG['PICKLE_PATH'], 'rb') as f:
            data = pickle.load(f)
    else:
        data = load_and_preprocess_data()
        with open(CONFIG['PICKLE_PATH'], 'wb') as f:
            pickle.dump(data, f)

    # Prepare samples for train/test split
    samples = []
    for gesture in next(iter(data.values())).keys():
        for subj in data.values():
            mats = subj[gesture]
            for vec in mats.T:
                samples.append((vec, gesture))

    rng = np.random.default_rng(CONFIG['RNG_SEED'])
    perm = rng.permutation(len(samples))
    n_train = int(CONFIG['TRAIN_FRAC'] * len(samples))
    train_idx, test_idx = perm[:n_train], perm[n_train:n_train + CONFIG['TEST_SAMPLES']]

    X_train = np.vstack([vector_to_image(samples[i][0]).flatten() for i in train_idx])
    y_train = np.array([samples[i][1] for i in train_idx])
    test_frames = [vector_to_image(samples[i][0]) for i in test_idx]

    # Train dictionaries and LLC vocab
    dicts = train_dictionary(X_train, CONFIG['DICT_ATOMS_LIST'][0])
    llc_model, llc_atoms = train_llc_vocab(X_train, y_train)

    # Evaluate reconstruction quality
    rec_results = evaluate_reconstruction(test_frames, dicts, llc_atoms)
    plot_performance_comparison(rec_results)

    # Evaluate and plot separability
    sep_results = evaluate_separability(data, dicts, llc_atoms)
    plot_separability(sep_results, normalize=False)

    # Plot full reconstructions for one test frame
    plot_all_reconstructions(test_frames[0], dicts, llc_atoms)

    print("Done.")