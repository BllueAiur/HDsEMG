#!/usr/bin/env python3
"""
Full pipeline to:
 1. Load processed_data.pkl and labels_data.pkl
 2. Extract all 1s windows per subject/gesture
 3. Train a 256-atom dictionary (sparsity=20) on all frames
 4. Reconstruct each 1s window via DWT (db1) and Dictionary Learning
 5. Extract 200ms windows with 50ms hop from each 1s segment
 6. Stack all 1s windows into single 64×(2048×N) arrays
 7. Combine all static segments into a single 'static' set per subject
 8. Save four pickles without '_1s' suffix
"""
import os
import pickle
import numpy as np
import pywt
from collections import defaultdict
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.linear_model import OrthogonalMatchingPursuit

# ───────── CONFIG ─────────
FS       = 2048
WIN1     = FS                   # 1 second = 2048 samples
THRESH   = int(2.5 * FS)        # 2.5 seconds threshold
WIN200   = int(0.2 * FS)        # 200 ms window
HOP50    = int(0.05 * FS)       # 50 ms hop
DICT_AT  = 256                  # dictionary atoms
SPARSITY = 20                   # non-zero coefs
DATA_DIR = "data_for_training"

# ───────── HELPERS ─────────
_FLAT_IDX = {i: (7 - (i % 8)) * 8 + (7 - (i // 8)) for i in range(64)}

def vector_to_image(vec):
    img = np.zeros((8, 8), dtype=vec.dtype)
    for i, v in enumerate(vec):
        r = 7 - (i % 8)
        c = 7 - (i // 8)
        img[r, c] = v
    return img

def extract_blocks(mask):
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    splits = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    return [(g[0], g[-1] + 1) for g in splits]

# ───────── LOAD DATA ─────────
with open(os.path.join(DATA_DIR, "processed_data.pkl"), "rb") as f:
    processed = pickle.load(f)
with open(os.path.join(DATA_DIR, "labels_data.pkl"), "rb") as f:
    labels = pickle.load(f)

# ───────── EXTRACT 1s WINDOWS ─────────
windows_1s = defaultdict(lambda: defaultdict(list))
for subj, gests in processed.items():
    for gest, emg in gests.items():
        lab2 = labels.get(subj, {}).get(gest)
        if lab2 is None:
            continue
        mask = lab2[0] > 0
        # active segments
        for a, b in extract_blocks(mask):
            length = b - a
            if length < WIN1:
                continue
            start = a + max(0, (length - WIN1) // 2) if length > THRESH else a
            windows_1s[subj][gest].append(emg[:, start:start+WIN1])
        # static segment
        if not (subj == 'HS8' and gest == 'thumb_flex'):
            inv = ~mask
            blocks = [(a,b) for a,b in extract_blocks(inv) if (b-a) >= WIN1]
            if blocks:
                a_s, _ = max(blocks, key=lambda x: x[1]-x[0])
                windows_1s[subj]["static"].append(emg[:, a_s:a_s+WIN1])

# ───────── PRINT SHAPES ─────────
print("Found 1s windows:")
for subj, gests in windows_1s.items():
    for gest, wlist in gests.items():
        if wlist:
            print(f"  {subj}/{gest}: {len(wlist)} windows, each {wlist[0].shape}")

# ───────── TRAIN DICTIONARY WITH PROGRESS ─────────
all_imgs = []
wavelet = 'db1'
lvl = pywt.dwt_max_level(8, pywt.Wavelet(wavelet).dec_len)
for gests in windows_1s.values():
    for mats in gests.values():
        for mat in mats:
            for t in range(WIN1):
                all_imgs.append(vector_to_image(mat[:, t]).flatten())
X = np.vstack(all_imgs)
print(f"\n▶ Training dictionary on {X.shape[0]} frames with {DICT_AT} atoms…")
mdl = MiniBatchDictionaryLearning(
    n_components=DICT_AT,
    transform_algorithm='omp',
    transform_n_nonzero_coefs=SPARSITY,
    batch_size=512,
    random_state=0,
    n_jobs=-1,
    verbose=True
)
mdl.fit(X)
atoms = mdl.components_.T
print("✅ Dictionary training complete.\n")
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=SPARSITY)

# ───────── RECONSTRUCT & SLICE ─────────
actual_ds = defaultdict(lambda: defaultdict(list))
dwt_ds    = defaultdict(lambda: defaultdict(list))
dict_ds   = defaultdict(lambda: defaultdict(list))
seg200    = defaultdict(lambda: defaultdict(list))
total = sum(len(lst) for gests in windows_1s.values() for lst in gests.values())
count = 0
for subj, gests in windows_1s.items():
    for gest, wlist in gests.items():
        for mat in wlist:
            count += 1
            print(f"[{count}/{total}] Reconstructing {subj}/{gest}...", end="\r")
            actual_ds[subj][gest].append(mat)
            R_dwt  = np.zeros_like(mat)
            R_dict = np.zeros_like(mat)
            for t in range(WIN1):
                v = mat[:, t]; img = vector_to_image(v)
                coeffs, sl = pywt.coeffs_to_array(pywt.wavedec2(img, wavelet, lvl))
                flat = coeffs.ravel()
                idxs = np.argsort(np.abs(flat))[-SPARSITY:]
                mask = np.zeros_like(flat, bool); mask[idxs] = True
                thr = (flat*mask).reshape(coeffs.shape)
                rec_img = pywt.waverec2(pywt.array_to_coeffs(thr, sl, 'wavedec2'), wavelet)
                rec_flat = rec_img.ravel()
                for ch in range(64): R_dwt[ch,t] = rec_flat[_FLAT_IDX[ch]]
                omp.fit(atoms, img.flatten())
                coef = omp.coef_; rec = (atoms @ coef).ravel()
                for ch in range(64): R_dict[ch,t] = rec[_FLAT_IDX[ch]]
            dwt_ds[subj][gest].append(R_dwt)
            dict_ds[subj][gest].append(R_dict)
            for s in range(0, WIN1-WIN200+1, HOP50):
                seg200[subj][gest].append(mat[:, s:s+WIN200])
print("\n✅ Reconstructions complete.\n")

# ───────── STACK 1s WINDOWS ─────────
for name, ds in (("actual", actual_ds), ("dwt", dwt_ds), ("dict", dict_ds)):
    for subj, gests in ds.items():
        for gest, mats in gests.items():
            ds[subj][gest] = np.hstack(mats) if mats else np.empty((64,0))
    print(f"[{name}] stacked all 1s windows.")

# ───────── COMBINE ALL STATIC ─────────
for name, ds in (("actual", actual_ds), ("dwt", dwt_ds), ("dict", dict_ds), ("segmented_200ms", seg200)):
    for subj, gests in ds.items():
        static_keys = [g for g in gests if g == 'static']
        if static_keys:
            # already merged under 'static' during extraction
            continue
    print(f"[{name}] static combined.")

# ───────── SAVE PICKLES ─────────
# Recursively convert defaultdicts to plain dicts
from collections import defaultdict as _ddict

def to_plain(d):
    if isinstance(d, _ddict):
        d = dict(d)
    if isinstance(d, dict):
        return {k: to_plain(v) for k,v in d.items()}
    return d

os.makedirs(DATA_DIR, exist_ok=True)
for name, ds in (
    ("segmented_200ms", seg200),
    ("actual",           actual_ds),
    ("dwt",              dwt_ds),
    ("dict",             dict_ds),
):
    plain = to_plain(ds)
    path = os.path.join(DATA_DIR, f"{name}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(plain, f)
    print(f"Saved {name}.pkl")
print("All pickles saved.")
