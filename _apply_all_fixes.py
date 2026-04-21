import re

path = r"e:\FInal Year Project\LDACode\WorkingNow.py"
with open(path, "r", encoding="utf-8") as f:
    src = f.read()

changes = 0

# Fix 1: Add N_FEAT_SEF + update N_FEATURES_RAW
old = "N_FEAT_FAA  = 14   # Frontal Alpha Asymmetry + hemispheric asymmetries\nN_FEAT_RIEM = 10   # Riemannian tangent-space upper triangle of 4x4 log-cov\nN_FEATURES_RAW = (N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP +\n                  N_FEAT_HR  + N_FEAT_PPI  + N_FEAT_FAA + N_FEAT_RIEM)"
new = "N_FEAT_FAA  = 14   # Frontal Alpha Asymmetry + hemispheric asymmetries\nN_FEAT_SEF  = 8    # SEF90 + median freq, per 4 EEG channels (2 x 4)\nN_FEAT_RIEM = 10   # Riemannian tangent-space upper triangle of 4x4 log-cov\nN_FEATURES_RAW = (N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP +\n                  N_FEAT_HR  + N_FEAT_PPI  + N_FEAT_FAA + N_FEAT_SEF + N_FEAT_RIEM)"
if old in src:
    src = src.replace(old, new, 1); changes += 1; print("Fix1 OK")
else:
    print("Fix1 MISS")

# Fix 2: Insert extract_sef_features before Riemannian section
sef_func = "\n# ===============================================================\n# SPECTRAL EDGE FREQUENCY  (8 features)\n# SEF90 = freq below which 90% of power resides; MedFreq = 50%\n# 4 channels x 2 features = 8\n# ===============================================================\ndef extract_sef_features(eeg_4ch, sr=EEG_SR):\n    feats = []\n    for ch in range(4):\n        sig = eeg_4ch[ch].astype(np.float64)\n        freqs, psd = _compute_psd(sig, sr)\n        cum = np.cumsum(psd)\n        total = cum[-1]\n        if total < 1e-12:\n            feats.extend([0.0, 0.0])\n            continue\n        sef90 = float(freqs[np.searchsorted(cum, 0.90 * total)])\n        med   = float(freqs[np.searchsorted(cum, 0.50 * total)])\n        feats.extend([sef90, med])\n    return safe_array(np.array(feats, dtype=np.float32))\n\n"
marker = "# ===============================================================\n# RIEMANNIAN GEOMETRY"
if marker in src:
    src = src.replace(marker, sef_func + marker, 1); changes += 1; print("Fix2 OK")
else:
    print("Fix2 MISS")

# Fix 3: Add f_sef to window concat
old3 = "        f_faa  = extract_faa_features(ew)          # 14 FAA features\n        f_riem = extract_riemannian_features(ew)   # 10 Riemannian (pre-EA)\n\n        feat = np.concatenate(\n            [f_eeg, f_muse, f_bvp, f_hr, f_ppi, f_faa, f_riem]\n        ).astype(np.float32)"
new3 = "        f_faa  = extract_faa_features(ew)          # 14 FAA features\n        f_sef  = extract_sef_features(ew)           # 8 SEF features\n        f_riem = extract_riemannian_features(ew)   # 10 Riemannian (pre-EA)\n\n        feat = np.concatenate(\n            [f_eeg, f_muse, f_bvp, f_hr, f_ppi, f_faa, f_sef, f_riem]\n        ).astype(np.float32)"
if old3 in src:
    src = src.replace(old3, new3, 1); changes += 1; print("Fix3 OK")
else:
    print("Fix3 MISS")

# Fix 4: Update RIEM_START
old4 = "RIEM_START = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA\n"
new4 = "RIEM_START = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA + N_FEAT_SEF\n"
if old4 in src:
    src = src.replace(old4, new4, 1); changes += 1; print("Fix4 OK")
else:
    print("Fix4 MISS")

# Fix 5: Widen K_GRID
old5 = "K_GRID = [80, 120, N_FEATURES]"
new5 = "K_GRID = [60, 80, 100, 120, 150, 200, N_FEATURES]"
if old5 in src:
    src = src.replace(old5, new5, 1); changes += 1; print("Fix5 OK")
else:
    print("Fix5 MISS")

# Fix 6: MI sub-sample 2000 -> 5000
n = src.count("2000, replace=False")
src = src.replace("2000, replace=False", "5000, replace=False")
changes += n; print(f"Fix6 OK: {n} occurrences")

# Fix 7: CORAL on fold-based test split
old7 = "# PCA: fit on train windows only, decorrelates before LDA\npca_final = PCA(n_components=0.95, random_state=42)\ntrF_pca   = pca_final.fit_transform(trF_final[:, final_feature_idx])\nteF_pca   = pca_final.transform(teF_final[:, final_feature_idx])"
new7 = "# CORAL: align test-fold covariance to training distribution\nteF_coral = np.clip(\n    coral_align(trF_final[:, final_feature_idx], teF_final[:, final_feature_idx]),\n    -10, 10)\ntrF_sel   = trF_final[:, final_feature_idx]\n\n# PCA: fit on train windows only, decorrelates before LDA\npca_final = PCA(n_components=0.95, random_state=42)\ntrF_pca   = pca_final.fit_transform(trF_sel)\nteF_pca   = pca_final.transform(teF_coral)"
if old7 in src:
    src = src.replace(old7, new7, 1); changes += 1; print("Fix7 OK")
else:
    print("Fix7 MISS")

with open(path, "w", encoding="utf-8") as f:
    f.write(src)
print(f"Total: {changes}")
