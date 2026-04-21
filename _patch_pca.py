import ast

with open('WorkingNow.py', encoding='utf-8') as f:
    src = f.read()

# 1. Add PCA import
old = 'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis'
new = old + '\nfrom sklearn.decomposition import PCA'
assert old in src, "import anchor not found"
src = src.replace(old, new, 1)

# 2. Final model block: replace LDA fit section with PCA -> LDA
old2 = (
    "final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=FINAL_SH)\n"
    "final_clf.fit(trF_final[:, final_feature_idx], trY_final)\n"
    "print(\"Final model trained.\")\n"
    "print(\"\\nTraining sanity-check report:\")\n"
    "print(classification_report(trY_final, final_clf.predict(trF_final[:, final_feature_idx]),\n"
    "    target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))"
)
new2 = (
    "# PCA: fit on train windows only, decorrelates before LDA\n"
    "pca_final = PCA(n_components=0.95, random_state=42)\n"
    "trF_pca   = pca_final.fit_transform(trF_final[:, final_feature_idx])\n"
    "teF_pca   = pca_final.transform(teF_final[:, final_feature_idx])\n"
    "print('PCA: {} -> {} components (95% var)'.format(len(final_feature_idx), trF_pca.shape[1]))\n"
    "\n"
    "final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=FINAL_SH)\n"
    "final_clf.fit(trF_pca, trY_final)\n"
    "print(\"Final model trained.\")\n"
    "print(\"\\nTraining sanity-check report:\")\n"
    "print(classification_report(trY_final, final_clf.predict(trF_pca),\n"
    "    target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))"
)
assert old2 in src, "final_clf anchor not found"
src = src.replace(old2, new2, 1)

# 3. Fix predict_proba call to use PCA-transformed test features
old3 = 'te_probs = final_clf.predict_proba(teF_final[:, final_feature_idx])'
new3 = 'te_probs = final_clf.predict_proba(teF_pca)'
assert old3 in src, "te_probs anchor not found"
src = src.replace(old3, new3, 1)

# 4. LOSO: insert PCA after MI selection, before LDA fit
old4 = (
    "    try:\n"
    "        clf_l = LinearDiscriminantAnalysis(solver=\"lsqr\", shrinkage=FINAL_SH)\n"
    "        clf_l.fit(trF_l[:, fi_l], trY_l)\n"
    "        probs_l = clf_l.predict_proba(teF_l[:, fi_l])"
)
new4 = (
    "    # PCA per LOSO fold: fit on training subjects only\n"
    "    pca_l  = PCA(n_components=0.95, random_state=42)\n"
    "    trF_lp = pca_l.fit_transform(trF_l[:, fi_l])\n"
    "    teF_lp = pca_l.transform(teF_l[:, fi_l])\n"
    "\n"
    "    try:\n"
    "        clf_l = LinearDiscriminantAnalysis(solver=\"lsqr\", shrinkage=FINAL_SH)\n"
    "        clf_l.fit(trF_lp, trY_l)\n"
    "        probs_l = clf_l.predict_proba(teF_lp)"
)
assert old4 in src, "LOSO clf anchor not found"
src = src.replace(old4, new4, 1)

with open('WorkingNow.py', 'w', encoding='utf-8') as f:
    f.write(src)

ast.parse(src)
print("Syntax OK")

# Print all PCA-related lines to confirm
for i, line in enumerate(src.splitlines()):
    if any(x in line for x in ['from sklearn.decomposition', 'pca_final', 'pca_l', 'trF_pca', 'teF_pca', 'trF_lp', 'teF_lp']):
        print(f'{i+1:4d}: {line.rstrip()}')
