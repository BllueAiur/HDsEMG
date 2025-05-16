#!/usr/bin/env python3
"""
Full pipeline with LLC added, simplified indexing, and proper static handling
"""
import os
import pickle
import numpy as np
import pywt
from collections import defaultdict
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit
from utils.LLC import MultiClassLLC

# ───────── CONFIG ─────────
FS = 2048
WIN1 = FS
THRESH = int(2.5 * FS)
WIN200 = int(0.2 * FS)
HOP50 = int(0.05 * FS)
DICT_AT = 128
SPARSITY = 16
DATA_DIR = "data_for_training"
LLC_BETA = 1e-3
LLC_WORDS_PER_CLASS = 16

# ───────── HELPERS ─────────
def extract_blocks(mask):
    idx = np.nonzero(mask)[0]
    return [] if idx.size == 0 else [(g[0], g[-1]+1) for g in np.split(idx, np.where(np.diff(idx) > 1)[0]+1)]

# ───────── DATA LOADING ─────────
with open(os.path.join(DATA_DIR, "processed_data.pkl"), "rb") as f:
    processed = pickle.load(f)
with open(os.path.join(DATA_DIR, "labels_data.pkl"), "rb") as f:
    labels = pickle.load(f)

# ───────── 1s WINDOW EXTRACTION ─────────
windows_1s = defaultdict(lambda: defaultdict(list))
for subj, gests in processed.items():
    for gest, emg in gests.items():
        lab = labels.get(subj, {}).get(gest)
        if lab is None: continue
        
        mask = lab[0] > 0
        # Active segments
        for a, b in extract_blocks(mask):
            if (length := b-a) >= WIN1:
                start = a + max(0, (length-WIN1)//2) if length > THRESH else a
                windows_1s[subj][gest].append(emg[:, start:start+WIN1])
        
        # Static segments (include in LLC training)
        if not (subj == 'HS8' and gest == 'thumb_flex'):
            inv = ~mask
            blocks = [(a,b) for a,b in extract_blocks(inv) if (b-a) >= WIN1]
            if blocks:
                a_s, _ = max(blocks, key=lambda x: x[1]-x[0])
                windows_1s[subj]["static"].append(emg[:, a_s:a_s+WIN1])

# ───────── DATA COLLECTION ─────────
all_vectors = []
class_features = defaultdict(list)
wavelet = 'coif1'
lvl = pywt.dwt_max_level(8, pywt.Wavelet(wavelet).dec_len)

for subj, gests in windows_1s.items():
    for gest, mats in gests.items():
        for mat in mats:
            # Collect raw vectors directly (no index transformation)
            for t in range(WIN1):
                vec = mat[:, t].copy()
                all_vectors.append(vec)
                class_features[gest].append(vec)

# ───────── DICTIONARY TRAINING ─────────
X = np.vstack(all_vectors)
print(f"\n▶ Training dictionary on {X.shape[0]} frames...")
mdl = MiniBatchDictionaryLearning(
    n_components=DICT_AT,
    transform_algorithm='omp',
    transform_n_nonzero_coefs=SPARSITY,
    batch_size=512,
    random_state=191,
    n_jobs=-1,
).fit(X)
atoms = mdl.components_.T
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=SPARSITY)

# ───────── LLC TRAINING (INCLUDES STATIC) ─────────
print("▶ Training LLC vocabularies...")
feature_list, class_list = [], []
for gest, feats in class_features.items():
    if len(feats) < LLC_WORDS_PER_CLASS: 
        continue  # Skip classes with insufficient samples
    feature_list.append(np.vstack(feats))
    class_list.append(gest)

llc = MultiClassLLC(
    knn=SPARSITY,
    beta=LLC_BETA,
    per_class_words=LLC_WORDS_PER_CLASS
).fit(feature_list, class_list)

# ───────── RECONSTRUCTION PIPELINE ─────────
actual_ds = defaultdict(lambda: defaultdict(list))
dwt_ds = defaultdict(lambda: defaultdict(list))
dict_ds = defaultdict(lambda: defaultdict(list))
llc_ds = defaultdict(lambda: defaultdict(list))
seg200 = defaultdict(lambda: defaultdict(list))

total = sum(len(lst) for gests in windows_1s.values() for lst in gests.values())
count = 0  # Initialize counter

for subj, gests in windows_1s.items():
    for gest, wlist in gests.items():
        for mat in wlist:
            count += 1  # Increment per matrix
            print(f"[{count}/{total}] Processing {subj}/{gest}", end="\r")
            actual_ds[subj][gest].append(mat)
            R_dwt, R_dict, R_llc = [np.zeros_like(mat) for _ in range(3)]
            
            for t in range(WIN1):
                vec = mat[:, t]
                
                # DWT Reconstruction
                img = vec.reshape(8, 8)
                coeffs = pywt.wavedec2(img, wavelet, level=lvl)
                arr, slices = pywt.coeffs_to_array(coeffs)
                idxs = np.argsort(np.abs(arr.ravel()))[::-1][:SPARSITY]
                mask = np.zeros(arr.size, bool)
                mask[idxs] = True
                arr_filtered = (arr.ravel() * mask).reshape(arr.shape)
                rec_dwt = pywt.waverec2(pywt.array_to_coeffs(arr_filtered, slices, 'wavedec2'), wavelet)
                R_dwt[:,t] = rec_dwt.ravel()
                
                # Dictionary Reconstruction
                omp.fit(atoms, vec)
                rec_dict = atoms @ omp.coef_
                R_dict[:,t] = rec_dict
                
                # LLC Reconstruction
                if gest in llc.dictionaries:
                    cls_atoms = llc.dictionaries[gest]
                    dists = np.sum((cls_atoms - vec)**2, axis=1)
                    neigh = np.argsort(dists)[:SPARSITY]
                    Z = cls_atoms[neigh] - vec
                    C = Z @ Z.T + LLC_BETA * np.eye(SPARSITY)
                    w = np.linalg.solve(C, np.ones(SPARSITY))
                    w /= w.sum()
                    rec_llc = (cls_atoms[neigh].T @ w).ravel()
                    R_llc[:,t] = rec_llc
            
            dwt_ds[subj][gest].append(R_dwt)
            dict_ds[subj][gest].append(R_dict)
            llc_ds[subj][gest].append(R_llc)
            
            # 200ms segmentation
            for s in range(0, WIN1-WIN200+1, HOP50):
                seg200[subj][gest].append(mat[:, s:s+WIN200])

# ───────── POST-PROCESSING ─────────
def stack(ds):
    for subj, gests in ds.items():
        for gest in gests:
            ds[subj][gest] = np.hstack(ds[subj][gest]) if ds[subj][gest] else np.empty((64,0))

for ds in [actual_ds, dwt_ds, dict_ds, llc_ds]:
    stack(ds)

# ───────── SAVING RESULTS ─────────
def to_plain(d):
    return {k: to_plain(v) for k,v in d.items()} if isinstance(d, defaultdict) else d

os.makedirs(DATA_DIR, exist_ok=True)
for name, ds in [
    ("segmented_200ms", seg200),
    ("actual", actual_ds),
    ("dwt", dwt_ds),
    ("dict", dict_ds),
    ("llc", llc_ds)
]:
    with open(os.path.join(DATA_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(to_plain(ds), f)
    print(f"Saved {name}.pkl")

print(" All processing complete!")