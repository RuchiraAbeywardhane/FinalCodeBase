import re

with open('train.py', 'r', encoding='utf-8') as f:
    src = f.read()

# 1. Add SVC import
src = src.replace(
    'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis',
    'from sklearn.discriminant_analysis import LinearDiscriminantAnalysis\nfrom sklearn.svm import SVC'
)

# 2. Replace grid search header + body
src = re.sub(
    r'# SELECT FINAL LDA HYPERPARAMETERS\n',
    '# SELECT FINAL SVM HYPERPARAMETERS  (no 3-component ceiling unlike LDA)\n',
    src
)
src = src.replace(
    'print("\\nSelecting final LDA hyperparameters ...")',
    'print("\\nSelecting final SVM hyperparameters ...")'
)
src = src.replace(
    "K_GRID = [80, 120, N_FEATURES]\nSH_GRID = ['auto', 0.05, 0.1, 0.3, 0.5, 0.7, 0.9]\n\ngrid_scores = []\n\nfor K in K_GRID:\n    for sh in SH_GRID:\n        fold_scores = []\n\n        for fk in range(4):\n            te_mask = (win_pos == fk)\n            vl_mask = (win_pos == (fk + 1) % 4)\n            tr_mask = ~te_mask & ~vl_mask\n\n            trF = allF_n[tr_mask]\n            trY = allY[tr_mask]\n            vlF = allF_n[vl_mask]\n            vlY = allY[vl_mask]\n\n            mi_ranked = np.argsort(-mi_per_fold[fk])\n            fi = mi_ranked[:K]\n\n            try:\n                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=sh)\n                clf.fit(trF[:, fi], trY)\n                sc = float((clf.predict(vlF[:, fi]) == vlY).mean())\n                fold_scores.append(sc)\n            except Exception:\n                fold_scores.append(-1.0)\n\n        mean_sc = np.mean(fold_scores)\n        grid_scores.append((mean_sc, K, sh))\n\ngrid_scores.sort(reverse=True, key=lambda x: x[0])\nbest_mean_val, FINAL_K, FINAL_SH = grid_scores[0]\n\nprint(f\"Chosen final params: K={FINAL_K}, shrinkage={FINAL_SH}, mean_val={best_mean_val:.4f}\")",
    "K_GRID = [60, 80, 120, N_FEATURES]\nC_GRID = [0.1, 1.0, 10.0, 100.0]\n\ngrid_scores = []\n\nfor K in K_GRID:\n    for C in C_GRID:\n        fold_scores = []\n\n        for fk in range(4):\n            te_mask = (win_pos == fk)\n            vl_mask = (win_pos == (fk + 1) % 4)\n            tr_mask = ~te_mask & ~vl_mask\n\n            trF = allF_n[tr_mask]\n            trY = allY[tr_mask]\n            vlF = allF_n[vl_mask]\n            vlY = allY[vl_mask]\n\n            mi_ranked = np.argsort(-mi_per_fold[fk])\n            fi = mi_ranked[:K]\n\n            try:\n                clf = SVC(kernel='rbf', C=C, gamma='scale',\n                          class_weight='balanced', probability=True,\n                          random_state=42)\n                clf.fit(trF[:, fi], trY)\n                sc = float((clf.predict(vlF[:, fi]) == vlY).mean())\n                fold_scores.append(sc)\n            except Exception:\n                fold_scores.append(-1.0)\n\n        mean_sc = np.mean(fold_scores)\n        grid_scores.append((mean_sc, K, C))\n        print(f'  K={K:4d}  C={C:6.1f}  val_acc={mean_sc:.4f}')\n\ngrid_scores.sort(reverse=True, key=lambda x: x[0])\nbest_mean_val, FINAL_K, FINAL_C = grid_scores[0]\n\nprint(f\"Chosen final params: K={FINAL_K}, C={FINAL_C}, mean_val={best_mean_val:.4f}\")"
)

# 3. Replace final model training
src = src.replace(
    "print(f\"\\nTraining final LDA on folds != TEST_FOLD ({TEST_FOLD}) ...\")",
    "print(f\"\\nTraining final SVM on folds != TEST_FOLD ({TEST_FOLD}) ...\")"
)
src = src.replace(
    "final_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=FINAL_SH)\nfinal_clf.fit(trF_final[:, final_feature_idx], trY_final)",
    "final_clf = SVC(kernel='rbf', C=FINAL_C, gamma='scale',\n                class_weight='balanced', probability=True, random_state=42)\nfinal_clf.fit(trF_final[:, final_feature_idx], trY_final)"
)

# 4. Replace LOSO classifier
src = src.replace(
    "        clf_l = LinearDiscriminantAnalysis(solver=\"lsqr\", shrinkage=FINAL_SH)\n        clf_l.fit(trF_l[:, fi_l], trY_l)",
    "        clf_l = SVC(kernel='rbf', C=FINAL_C, gamma='scale',\n                    class_weight='balanced', probability=True, random_state=42)\n        clf_l.fit(trF_l[:, fi_l], trY_l)"
)

with open('train.py', 'w', encoding='utf-8') as f:
    f.write(src)

# Verify syntax
import ast
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")

# Verify key strings are present
checks = ['from sklearn.svm import SVC', 'C_GRID', 'FINAL_C', "kernel='rbf'"]
for c in checks:
    print(f"  {'OK' if c in src else 'MISSING'}: {c}")
