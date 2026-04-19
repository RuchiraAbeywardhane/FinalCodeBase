import ast

with open('train.py', 'r', encoding='utf-8') as f:
    src = f.read()

# ── 1. Add coral_align function after euclidean_alignment ──
coral_func = '''
# ===============================================================
# CORAL -- CORrelation ALignment  (Sun & Saenko 2016)
# Aligns the covariance of source (train) features to match the
# target (test subject) feature covariance.
# Steps:
#   1. Whiten source:   X_src @ C_src^{-1/2}
#   2. Re-colour:       result  @ C_tgt^{+1/2}
# No labels needed -- purely unsupervised.
# ===============================================================
def coral_align(X_src, X_tgt, reg=1e-6):
    """
    X_src : (N_train, F)  z-scored training features
    X_tgt : (N_test,  F)  z-scored test features
    Returns X_src_aligned with same shape as X_src.
    X_tgt is unchanged -- it is already the target distribution.
    """
    from scipy.linalg import sqrtm
    X_src = np.array(X_src, dtype=np.float64)
    X_tgt = np.array(X_tgt, dtype=np.float64)
    F = X_src.shape[1]

    C_src = np.cov(X_src.T) + reg * np.eye(F)
    C_tgt = np.cov(X_tgt.T) + reg * np.eye(F)

    # matrix square roots -- sqrtm can return complex due to numerics, take real part
    A_src = np.real(sqrtm(np.linalg.inv(C_src)))   # C_src^{-1/2}
    A_tgt = np.real(sqrtm(C_tgt))                   # C_tgt^{+1/2}

    X_aligned = X_src @ A_src @ A_tgt
    return np.clip(X_aligned, -10, 10).astype(np.float32)

'''

src = src.replace(
    'def euclidean_alignment(eeg_windows_list):',
    coral_func + 'def euclidean_alignment(eeg_windows_list):'
)

# ── 2. Apply CORAL in the final test split (after z-scoring teF_final) ──
old_final = (
    'mi_global         = mutual_info_classif(trF_final[sub_idx], trY_final[sub_idx], random_state=42, n_neighbors=5)\n'
    'final_feature_idx = np.argsort(-mi_global)[:FINAL_K]'
)
new_final = (
    '# CORAL: align train features towards test-subject distribution\n'
    'print("Applying CORAL alignment on final test split ...")\n'
    'teF_final = coral_align(teF_final, teF_final)  # identity -- target stays as-is\n'
    'trF_final_c = coral_align(trF_final, teF_final)\n'
    '\n'
    'mi_global         = mutual_info_classif(trF_final_c[sub_idx], trY_final[sub_idx], random_state=42, n_neighbors=5)\n'
    'final_feature_idx = np.argsort(-mi_global)[:FINAL_K]'
)
src = src.replace(old_final, new_final)

# update the fit call to use trF_final_c
src = src.replace(
    'final_clf.fit(trF_final[:, final_feature_idx], trY_final)',
    'final_clf.fit(trF_final_c[:, final_feature_idx], trY_final)'
)
# update the training sanity check predict call
src = src.replace(
    'final_clf.predict(trF_final[:, final_feature_idx])',
    'final_clf.predict(trF_final_c[:, final_feature_idx])'
)

# ── 3. Apply CORAL in the LOSO loop (after z-scoring, before MI) ──
old_loso_mi = (
    '    trY_l = allY[tr_m]; teY_l = allY[te_m]; teTK_l = allTK[te_m]\n'
    '\n'
    '    sub_l = (np.random.RandomState(42).choice(len(trF_l), 2000, replace=False)\n'
    '             if len(trF_l) > 2000 else np.arange(len(trF_l)))\n'
    '    mi_l  = mutual_info_classif(trF_l[sub_l], trY_l[sub_l], random_state=42, n_neighbors=5)\n'
    '    fi_l  = np.argsort(-mi_l)[:FINAL_K]'
)
new_loso_mi = (
    '    trY_l = allY[tr_m]; teY_l = allY[te_m]; teTK_l = allTK[te_m]\n'
    '\n'
    '    # CORAL: align training feature distribution towards this test subject\n'
    '    trF_l = coral_align(trF_l, teF_l)\n'
    '\n'
    '    sub_l = (np.random.RandomState(42).choice(len(trF_l), 2000, replace=False)\n'
    '             if len(trF_l) > 2000 else np.arange(len(trF_l)))\n'
    '    mi_l  = mutual_info_classif(trF_l[sub_l], trY_l[sub_l], random_state=42, n_neighbors=5)\n'
    '    fi_l  = np.argsort(-mi_l)[:FINAL_K]'
)
src = src.replace(old_loso_mi, new_loso_mi)

with open('train.py', 'w', encoding='utf-8') as f:
    f.write(src)

# Verify
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")

checks = [
    'def coral_align(',
    'from scipy.linalg import sqrtm',
    'C_src^{-1/2}',
    'trF_l = coral_align(trF_l, teF_l)',
    'trF_final_c = coral_align(',
    'final_clf.fit(trF_final_c[:,',
]
for c in checks:
    print('OK' if c in src else 'MISSING', ':', c)
