"""
Fixes two data leakages in WorkingNow.py:
  1. MixUp was sampling partners from ALL folds (including test fold).
     Fix: build cls_fold_idx keyed by (class, fold) so MixUp only
          pairs windows within the same fold.
  2. Euclidean Alignment was computing R from ALL subject windows
     (including test-fold ones).
     Fix: only accumulate train-fold windows into _sid_to_train_ew
          when estimating R; apply the resulting Rinv to every window.

Both fixes require fold assignment to happen BEFORE augmentation and EA,
so this script also reorders those three sections.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("WorkingNow.py", encoding="utf-8") as f:
    content = f.read()

# ── locate section boundaries ───────────────────────────────────
AUG_HDR  = "# FEATURE-SPACE DATA AUGMENTATION  (training windows only)"
EA_HDR   = "# EUCLIDEAN ALIGNMENT -- per-subject Riemannian re-centring"
FOLD_HDR = "# BALANCED FOLD ASSIGNMENT  (must happen before preprocessing)"
PRE_HDR  = "# PREPROCESS  -  fit on non-test-fold windows only (no leakage)"

aug_start  = content.index(AUG_HDR)
ea_start   = content.index(EA_HDR)
fold_start = content.index(FOLD_HDR)
pre_start  = content.index(PRE_HDR)

before = content[:aug_start]   # everything up to (not including) augmentation
after  = content[pre_start:]   # PREPROCESS onwards — unchanged

# ── 1. FOLD ASSIGNMENT block (moved first, no logic changes) ────
new_fold = '''\
# BALANCED FOLD ASSIGNMENT  moved BEFORE augmentation & EA
# Fold assignment must precede both so they can be fold-aware.
print("\\nBalanced fold assignment ...")

labels_arr = np.array([tr["label"] for tr in train_trials])

subj_emo_trial = {}
for ti, tr in enumerate(train_trials):
    subj_emo_trial.setdefault(tr["sid"], {})[tr["label"]] = ti

rng_fold = np.random.RandomState(123)
shuffled_subs = sorted(subj_emo_trial.keys())
rng_fold.shuffle(shuffled_subs)

trial_pos = {}
for g_idx, sid in enumerate(shuffled_subs):
    group = g_idx % 4
    for emo_idx in range(4):
        if emo_idx in subj_emo_trial[sid]:
            trial_pos[subj_emo_trial[sid][emo_idx]] = (group + emo_idx) % 4

# Assign fold to every ORIGINAL window now, before augmentation or EA
win_pos_orig = np.array([trial_pos[ti] for ti in allTI])

for fk in range(4):
    te_t = [ti for ti, p in trial_pos.items() if p == fk]
    c = Counter(labels_arr[te_t].tolist())
    print(f"Fold {fk}: test clips={len(te_t)} emotions={dict(sorted(c.items()))}")

'''

# ── 2. AUGMENTATION block (MixUp restricted to same fold) ───────
new_aug = '''\
# FEATURE-SPACE DATA AUGMENTATION  (training windows only)
# FIX: MixUp partner is drawn from the SAME fold only, so synthetic
#      training samples never contain information from test-fold windows.
AUG_NOISE_COPIES = 2
AUG_MIXUP_COPIES = 1
AUG_NOISE_STD    = 0.05
AUG_MIXUP_ALPHA  = 0.4

print("Augmenting training windows ...")
rng_aug  = np.random.RandomState(7)
feat_std = np.std(allF_raw, axis=0) + 1e-8

# Key change: index by (class, fold) so MixUp never crosses fold boundaries
cls_fold_idx = {
    (c, fk): np.where((allY == c) & (win_pos_orig == fk))[0]
    for c in range(NUM_CLASSES) for fk in range(4)
}

aug_feats, aug_labels, aug_tkeys, aug_sids, aug_tidxs, aug_winpos = [], [], [], [], [], []

for orig_idx in range(len(allY)):
    lbl  = allY[orig_idx];   feat = allF_raw[orig_idx]
    sid  = allSID[orig_idx]; tkey = allTK[orig_idx]
    ti   = allTI[orig_idx];  fk   = int(win_pos_orig[orig_idx])

    for _ in range(AUG_NOISE_COPIES):
        noise = rng_aug.randn(feat.shape[0]).astype(np.float32) * feat_std * AUG_NOISE_STD
        aug_feats.append(feat + noise)
        aug_labels.append(lbl); aug_tkeys.append(tkey)
        aug_sids.append(sid);   aug_tidxs.append(ti); aug_winpos.append(fk)

    # MixUp: only pool partners from the SAME fold and class
    pool = cls_fold_idx[(lbl, fk)]
    diff = pool[allSID[pool] != sid]
    pool = diff if len(diff) > 0 else pool

    for _ in range(AUG_MIXUP_COPIES):
        if len(pool) == 0:
            continue
        j   = rng_aug.choice(pool)
        lam = float(rng_aug.beta(AUG_MIXUP_ALPHA, AUG_MIXUP_ALPHA))
        mixed = (lam * feat + (1 - lam) * allF_raw[j]).astype(np.float32)
        aug_feats.append(mixed)
        aug_labels.append(lbl); aug_tkeys.append(tkey)
        aug_sids.append(sid);   aug_tidxs.append(ti); aug_winpos.append(fk)

if aug_feats:
    allF_raw  = np.vstack([allF_raw, np.array(aug_feats, dtype=np.float32)])
    allY      = np.concatenate([allY,  np.array(aug_labels)])
    allTK     = np.concatenate([allTK, np.array(aug_tkeys)])
    allSID    = np.concatenate([allSID, np.array(aug_sids)])
    allTI     = np.concatenate([allTI,  np.array(aug_tidxs)])
    allIS_AUG = np.concatenate([allIS_AUG, np.ones(len(aug_feats), dtype=bool)])
    # Augmented windows inherit the fold of their source (no cross-fold bleed)
    win_pos = np.concatenate([win_pos_orig, np.array(aug_winpos)])
    print(f"After augmentation: {len(allY)} total windows ({len(aug_feats)} synthetic)")
else:
    win_pos = win_pos_orig.copy()

'''

# ── 3. EA block (R fitted on train-fold windows only) ───────────
new_ea = '''\
# EUCLIDEAN ALIGNMENT -- per-subject Riemannian re-centring
# FIX: R (mean covariance) is estimated from TRAINING-FOLD windows only.
#      The derived Rinv is then applied to ALL windows of the subject
#      so test windows are correctly aligned without their EEG
#      contributing to the alignment matrix.
print("Applying Euclidean Alignment (EA) per subject (train-fold only for R) ...")
RIEM_START = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA

_sid_to_train_ew = defaultdict(list)  # train-fold windows -> fit R
_sid_to_all_ew   = defaultdict(list)  # all windows        -> aligned with train-derived R
_sid_to_gidx     = defaultdict(list)  # global row indices in allF_raw

for ti, tr in enumerate(train_trials):
    sid = tr["sid"]; eeg = tr["eeg"]; bvp = tr["bvp"]; band = tr["band"]
    dur2  = min(eeg.shape[1]/EEG_SR, band.shape[1]/BAND_SR, len(bvp)/TRAIN_BVP_SR)
    step2 = WINDOW_SEC * (1 - OVERLAP_FRAC)
    is_train_trial = (trial_pos.get(ti, TEST_FOLD) != TEST_FOLD)
    for wi in range(int((dur2 - WINDOW_SEC) / step2) + 1):
        e_s = int(wi * step2 * EEG_SR); e_e = e_s + EEG_WIN
        if e_e > eeg.shape[1]:
            break
        window = eeg[:, e_s:e_e]
        _sid_to_all_ew[sid].append(window)
        if is_train_trial:
            _sid_to_train_ew[sid].append(window)  # only train-fold for R

for gidx, sid in enumerate(allSID):
    _sid_to_gidx[sid].append(gidx)

for sid in sorted(_sid_to_all_ew.keys()):
    train_ew = _sid_to_train_ew.get(sid, [])
    if len(train_ew) < 2:
        continue  # need >= 2 windows to estimate a stable R

    # Compute R from training-fold windows ONLY -- no test leakage
    covs = [_regularised_cov(w.astype(np.float64)) for w in train_ew]
    R    = np.stack(covs, axis=0).mean(axis=0)
    Rinv = _matrix_sqrt_inv(R)

    # Apply the train-derived Rinv to ALL windows of this subject
    all_ew = _sid_to_all_ew[sid]
    gidxs  = _sid_to_gidx[sid]
    for k, window in enumerate(all_ew):
        if k < len(gidxs):
            aligned = Rinv @ window.astype(np.float64)
            allF_raw[gidxs[k], RIEM_START:RIEM_START+N_FEAT_RIEM] = \
                extract_riemannian_features(np.array(aligned, dtype=np.float32))

print("EA done -- R fitted on train-fold only, applied to all windows (no test leakage).")

'''

new_content = before + new_fold + new_aug + new_ea + after

with open("WorkingNow.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print(f"WorkingNow.py rewritten successfully ({new_content.count(chr(10))} lines).")

# Quick sanity check: fold assignment must appear before augmentation
fa = new_content.index("win_pos_orig = np.array")
ag = new_content.index("cls_fold_idx = {")
ea = new_content.index("_sid_to_train_ew = defaultdict")
assert fa < ag < ea, "Order wrong!"
print("Order check passed: fold_assign < augmentation < EA")
