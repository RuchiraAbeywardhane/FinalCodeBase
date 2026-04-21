import re

path = r"e:\FInal Year Project\LDACode\WorkingNow.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

# FIX 1: Add N_FEAT_SEF constant and update N_FEATURES_RAW
old = 'N_FEAT_FAA  = 14   # Frontal Alpha Asymmetry + hemispheric asymmetries\nN_FEAT_RIEM = 10   # Riemannian tangent-space upper triangle of 4x4 log-cov\nN_FEATURES_RAW = (N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP +\n                  N_FEAT_HR  + N_FEAT_PPI  + N_FEAT_FAA + N_FEAT_RIEM)'
new = 'N_FEAT_FAA  = 14   # Frontal Alpha Asymmetry + hemispheric asymmetries\nN_FEAT_SEF  = 8    # SEF90 + median freq, per 4 EEG channels (2 x 4)\nN_FEAT_RIEM = 10   # Riemannian tangent-space upper triangle of 4x4 log-cov\nN_FEATURES_RAW = (N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP +\n                  N_FEAT_HR  + N_FEAT_PPI  + N_FEAT_FAA + N_FEAT_SEF + N_FEAT_RIEM)'
assert old in src, "FIX1 anchor not found"
src = src.replace(old, new, 1)
print("FIX1 done")

# FIX 2: Add extract_sef_features function before Riemannian section
SEF_FUNC = (
    "\n"
    "# ===============================================================\n"
    "# SPECTRAL EDGE FREQUENCY  (8 features)\n"
    "# SEF90 = freq below which 90% of PSD power lies\n"
    "# MedFreq = median frequency (50% power)\n"
    "# Per channel: 4 channels x 2 = 8 features\n"
    "# ===============================================================\n"
    "def extract_sef_features(eeg_4ch, sr=EEG_SR):\n"
    "    feats = []\n"
    "    for ch in range(4):\n"
    "        sig = eeg_4ch[ch].astype(np.float64)\n"
    "        freqs, psd = _compute_psd(sig, sr)\n"
    "        cumsum = np.cumsum(psd)\n"
    "        total  = cumsum[-1]\n"
    "        if total < 1e-12:\n"
    "            feats.extend([0.0, 0.0])\n"
    "            continue\n"
    "        sef90 = float(freqs[np.searchsorted(cumsum, 0.90 * total)])\n"
    "        med_f = float(freqs[np.searchsorted(cumsum, 0.50 * total)])\n"
    "        feats.extend([sef90, med_f])\n"
    "    return safe_array(np.array(feats, dtype=np.float32))\n"
    "\n"
    "\n"
)
riem_anchor = "# ===============================================================\n# RIEMANNIAN GEOMETRY"
assert riem_anchor in src, "RIEM anchor not found"
src = src.replace(riem_anchor, SEF_FUNC + riem_anchor, 1)
print("FIX2 done")

# FIX 3: Add f_sef to window extraction and concatenation
old3 = (
    "        f_faa  = extract_faa_features(ew)          # 14 FAA features\n"
    "        f_riem = extract_riemannian_features(ew)   # 10 Riemannian (pre-EA)\n"
    "\n"
    "        feat = np.concatenate(\n"
    "            [f_eeg, f_muse, f_bvp, f_hr, f_ppi, f_faa, f_riem]\n"
    "        ).astype(np.float32)"
)
new3 = (
    "        f_faa  = extract_faa_features(ew)          # 14 FAA features\n"
    "        f_sef  = extract_sef_features(ew)           # 8 SEF features\n"
    "        f_riem = extract_riemannian_features(ew)   # 10 Riemannian (pre-EA)\n"
    "\n"
    "        feat = np.concatenate(\n"
    "            [f_eeg, f_muse, f_bvp, f_hr, f_ppi, f_faa, f_sef, f_riem]\n"
    "        ).astype(np.float32)"
)
assert old3 in src, "FIX3 extract anchor not found"
src = src.replace(old3, new3, 1)
print("FIX3 done")

# FIX 4: Update RIEM_START to include N_FEAT_SEF
old4 = "RIEM_START = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA"
new4 = "RIEM_START = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA + N_FEAT_SEF"
assert old4 in src, "FIX4 RIEM_START anchor not found"
src = src.replace(old4, new4, 1)
print("FIX4 done")

# FIX 5: Widen K_GRID
old5 = "K_GRID = [80, 120, N_FEATURES]"
new5 = "K_GRID = [60, 80, 100, 120, 150, 200, N_FEATURES]"
assert old5 in src, "FIX5 K_GRID anchor not found"
src = src.replace(old5, new5, 1)
print("FIX5 done")

# FIX 6: MI sub-sample 2000 -> 5000 (all occurrences)
src, n1 = re.subn(r"(choice\(tr_idx_mi,\s*)2000(\s*,\s*replace=False\))", r"\g<1>5000\2", src)
src, n2 = re.subn(r"(choice\(len\(trF_final\),\s*)2000(\s*,\s*replace=False\))", r"\g<1>5000\2", src)
src, n3 = re.subn(r"(choice\(len\(trF_l\),\s*)2000(\s*,\s*replace=False\))", r"\g<1>5000\2", src)
print(f"FIX6 done: mi_fold={n1}, final={n2}, loso={n3}")

# FIX 7: CORAL on fold-based test split
old7 = (
    "# PCA: fit on train windows only, decorrelates before LDA\n"
    "pca_final = PCA(n_components=0.95, random_state=42)\n"
    "trF_pca   = pca_final.fit_transform(trF_final[:, final_feature_idx])\n"
    "teF_pca   = pca_final.transform(teF_final[:, final_feature_idx])"
)
new7 = (
    "# CORAL: align test-fold feature covariance to training distribution (no leakage)\n"
    "print('Applying CORAL to test-fold features ...')\n"
    "teF_coral = np.clip(coral_align(trF_final[:, final_feature_idx],\n"
    "                                teF_final[:, final_feature_idx]), -10, 10)\n"
    "\n"
    "# PCA: fit on train windows only, decorrelates before LDA\n"
    "pca_final = PCA(n_components=0.95, random_state=42)\n"
    "trF_pca   = pca_final.fit_transform(trF_final[:, final_feature_idx])\n"
    "teF_pca   = pca_final.transform(teF_coral)"
)
assert old7 in src, "FIX7 CORAL anchor not found"
src = src.replace(old7, new7, 1)
print("FIX7 done")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)

print("\nAll fixes applied.")
print("N_FEAT_SEF present      :", "N_FEAT_SEF" in src)
print("extract_sef_features    :", "def extract_sef_features" in src)
print("f_sef in concatenation  :", "f_sef, f_riem" in src)
print("RIEM_START with SEF     :", "N_FEAT_SEF" in [l for l in src.splitlines() if "RIEM_START" in l][0])
print("K_GRID                  :", [l.strip() for l in src.splitlines() if "K_GRID" in l and "=" in l][0])
print("MI sub-sample = 5000    :", "5000" in src)
print("CORAL test-split        :", "teF_coral" in src)
