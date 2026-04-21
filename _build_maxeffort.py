"""
Builds MaxEffort.py by:
1. Taking everything up to (not including) PER-FOLD MI from WorkingNow.py
2. Appending a new section with every untried technique:
   - SMOTE (fix class imbalance / FEAR starvation)
   - PCA decorrelation (95% variance)
   - Stacking ensemble: LDA + HistGradientBoosting + XGBoost + RBF-SVM + KNN
   - Temporal EMA smoothing of window probabilities
   - Test-time subject re-centring in LOSO (unsupervised domain shift fix)
   - Hyperparameter search restricted to non-TEST_FOLD folds only
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open("WorkingNow.py", encoding="utf-8") as f:
    lines = f.readlines()

# Find where PER-FOLD MI starts — we keep everything before it
cutoff = next(i for i, l in enumerate(lines) if "PER-FOLD MI" in l)
preprocess_block = "".join(lines[:cutoff])

new_section = r"""
# ================================================================
# MAX EFFORT  -  EXTRA IMPORTS
# ================================================================
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# ================================================================
# HELPERS
# ================================================================
def smooth_probs_ema(probs, alpha=0.4):
    """Causal exponential moving average over consecutive windows."""
    out = probs.copy().astype(np.float64)
    for t in range(1, len(out)):
        out[t] = alpha * probs[t] + (1 - alpha) * out[t - 1]
    return out

def subject_recentre(trF, teF):
    """Shift test features so their mean matches training mean (unsupervised)."""
    return teF - (teF.mean(axis=0) - trF.mean(axis=0))

def make_lda(sh=0.3):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage=sh)

def make_hgb():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, max_depth=4,
        min_samples_leaf=20, l2_regularization=0.1,
        class_weight="balanced", random_state=42)

def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42,
        verbosity=0)

def make_rbf_svm():
    return CalibratedClassifierCV(
        SVC(kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=False),
        cv=3, method="isotonic")

def make_knn():
    return KNeighborsClassifier(
        n_neighbors=7, metric="euclidean",
        weights="distance", n_jobs=-1)

def build_stack(sh=0.3):
    return StackingClassifier(
        estimators=[
            ("lda", make_lda(sh)),
            ("hgb", make_hgb()),
            ("xgb", make_xgb()),
            ("svm", make_rbf_svm()),
            ("knn", make_knn()),
        ],
        final_estimator=make_lda(0.5),
        stack_method="predict_proba",
        cv=3, passthrough=False, n_jobs=-1)

# ================================================================
# PER-FOLD MI  (restricted to non-TEST_FOLD folds)
# ================================================================
print("\nComputing per-fold MI ...")
mi_per_fold = {}
for fk in range(4):
    tr_mask_mi = (win_pos != fk) & (win_pos != TEST_FOLD)
    tr_idx_mi  = np.where(tr_mask_mi)[0]
    sub = (np.random.RandomState(42).choice(tr_idx_mi, 2000, replace=False)
           if len(tr_idx_mi) > 2000 else tr_idx_mi)
    mi_per_fold[fk] = mutual_info_classif(
        allF_n[sub], allY[sub], random_state=42, n_neighbors=5)

# ================================================================
# HYPERPARAMETER SEARCH  (non-TEST_FOLD folds only)
# ================================================================
print("\nHyperparameter search (non-test folds only) ...")
K_GRID  = [60, 100, 160, N_FEATURES]
SH_GRID = [0.05, 0.1, 0.3, 0.5, 0.7]
non_test_folds = [fk for fk in range(4) if fk != TEST_FOLD]

best_val, FINAL_K, FINAL_SH = -1, 100, 0.3
for K in K_GRID:
    for sh in SH_GRID:
        scores = []
        for fk in non_test_folds:
            vl_m = (win_pos == fk)
            tr_m = (win_pos != fk) & (win_pos != TEST_FOLD)
            fi   = np.argsort(-mi_per_fold[fk])[:K]
            try:
                clf = make_lda(sh)
                clf.fit(allF_n[tr_m][:, fi], allY[tr_m])
                scores.append(float((clf.predict(allF_n[vl_m][:, fi]) == allY[vl_m]).mean()))
            except Exception:
                scores.append(-1.0)
        sc = float(np.mean(scores))
        if sc > best_val:
            best_val, FINAL_K, FINAL_SH = sc, K, sh

print(f"Best val={best_val:.4f}  K={FINAL_K}  shrinkage={FINAL_SH}")

# ================================================================
# TRAIN / TEST SPLIT
# ================================================================
train_mask = (win_pos != TEST_FOLD)
test_mask  = (win_pos == TEST_FOLD)
trF_all = allF_n[train_mask];  trY_all = allY[train_mask]
teF_all = allF_n[test_mask];   teY_all = allY[test_mask]
teTK    = allTK[test_mask];    teSID   = allSID[test_mask]
print(f"  Train windows: {len(trY_all)}   Test windows: {len(teY_all)}")

# Global MI on full train set
sub_g  = (np.random.RandomState(42).choice(len(trF_all), 2000, replace=False)
          if len(trF_all) > 2000 else np.arange(len(trF_all)))
mi_g   = mutual_info_classif(trF_all[sub_g], trY_all[sub_g], random_state=42, n_neighbors=5)
fi_g   = np.argsort(-mi_g)[:FINAL_K]

trF = trF_all[:, fi_g]
teF = teF_all[:, fi_g]

# ── SMOTE: oversample minority classes (train only) ────────────
print("Applying SMOTE ...")
counts = np.bincount(trY_all)
print(f"  Before SMOTE: {dict(enumerate(counts))}")
try:
    sm = SMOTE(random_state=42, k_neighbors=5)
    trF_sm, trY_sm = sm.fit_resample(trF, trY_all)
    print(f"  After  SMOTE: {dict(enumerate(np.bincount(trY_sm)))}")
except Exception as e:
    trF_sm, trY_sm = trF, trY_all
    print(f"  SMOTE skipped: {e}")

# ── PCA: decorrelate, keep 95% variance ────────────────────────
pca = PCA(n_components=0.95, random_state=42)
trF_pca = pca.fit_transform(trF_sm)
teF_pca = pca.transform(teF)
print(f"  PCA: {trF_sm.shape[1]} -> {trF_pca.shape[1]} components")

# ── Stacking ensemble ──────────────────────────────────────────
print("\nFitting stacking ensemble ...")
stack_clf = build_stack(FINAL_SH)
stack_clf.fit(trF_pca, trY_sm)
tr_hat = stack_clf.predict(trF_pca)
print(f"  Train accuracy (sanity): {(tr_hat == trY_sm).mean():.4f}")

# ================================================================
# EVALUATE: TEST SPLIT
# ================================================================
print("\nEvaluating on test split ...")
te_probs_raw = stack_clf.predict_proba(teF_pca)

trial_rows, window_rows = [], []
for tkey in sorted(set(teTK)):
    m        = (teTK == tkey)
    probs_t  = smooth_probs_ema(te_probs_raw[m], alpha=0.4)
    preds_t  = np.argmax(probs_t, axis=1)
    true_lbl = int(teY_all[m][0])
    for r in range(len(preds_t)):
        window_rows.append({
            "fold": TEST_FOLD, "trial_key": tkey, "window_idx": r,
            "true_idx": true_lbl, "true_label": IDX_TO_LABEL[true_lbl],
            "pred_idx": int(preds_t[r]), "pred_label": IDX_TO_LABEL[int(preds_t[r])],
            "prob_NEUTRAL":    float(probs_t[r, 0]),
            "prob_ENTHUSIASM": float(probs_t[r, 1]),
            "prob_SADNESS":    float(probs_t[r, 2]),
            "prob_FEAR":       float(probs_t[r, 3]),
        })
    pi, conf, mp = trial_vote_from_probs(probs_t)
    trial_rows.append({
        "fold": TEST_FOLD, "trial_key": tkey, "n_windows": int(m.sum()),
        "true_idx": true_lbl, "true_label": IDX_TO_LABEL[true_lbl],
        "trial_pred_idx": pi, "trial_pred_label": IDX_TO_LABEL[pi],
        "trial_confidence": conf,
        "prob_NEUTRAL":    float(mp[0]), "prob_ENTHUSIASM": float(mp[1]),
        "prob_SADNESS":    float(mp[2]), "prob_FEAR":       float(mp[3]),
    })

trial_df  = pd.DataFrame(trial_rows)
window_df = pd.DataFrame(window_rows)
trial_df.to_csv("/kaggle/working/maxeffort_trial_predictions.csv",   index=False)
window_df.to_csv("/kaggle/working/maxeffort_window_predictions.csv", index=False)

y_true_w = window_df["true_idx"].astype(int).values
y_pred_w = window_df["pred_idx"].astype(int).values
print("\n" + "=" * 60)
print("MAX EFFORT  --  TEST WINDOW-LEVEL")
print("=" * 60)
print(f"Window accuracy: {(y_true_w == y_pred_w).mean():.4f}")
print(classification_report(y_true_w, y_pred_w,
      target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))
print(confusion_matrix(y_true_w, y_pred_w, labels=list(range(NUM_CLASSES))))

y_true_t = trial_df["true_idx"].astype(int).values
y_pred_t = trial_df["trial_pred_idx"].astype(int).values
print("\n" + "=" * 60)
print("MAX EFFORT  --  TEST TRIAL-LEVEL")
print("=" * 60)
print(f"Trial accuracy: {(y_true_t == y_pred_t).mean():.4f}")
print(classification_report(y_true_t, y_pred_t,
      target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))
print(confusion_matrix(y_true_t, y_pred_t, labels=list(range(NUM_CLASSES))))

# ================================================================
# LOSO  -- per-subject: SMOTE + PCA + stack + re-centring + EMA
# ================================================================
print("\n" + "=" * 60)
print("MAX EFFORT  --  LOSO")
print("=" * 60)

loso_win_rows, loso_trial_rows, loso_accs = [], [], []

for loso_sid in sorted(set(allSID)):
    te_m = (allSID == loso_sid) & ~allIS_AUG
    tr_m = ~(allSID == loso_sid)
    if te_m.sum() == 0 or tr_m.sum() == 0:
        continue

    # Fresh VT + z-score on training subjects
    from sklearn.feature_selection import VarianceThreshold as _VT
    vt_l = _VT(threshold=0.001)
    vt_l.fit(allF_raw[tr_m])
    trF_l = vt_l.transform(allF_raw[tr_m])
    teF_l = vt_l.transform(allF_raw[te_m])
    mu_l  = trF_l.mean(axis=0)
    sd_l  = np.where(trF_l.std(axis=0) < 1e-8, 1.0, trF_l.std(axis=0))
    trF_l = np.clip((trF_l - mu_l) / sd_l, -10, 10)
    teF_l = np.clip((teF_l - mu_l) / sd_l, -10, 10)

    # Unsupervised test-time re-centring
    teF_l = subject_recentre(trF_l, teF_l)

    trY_l = allY[tr_m]; teY_l = allY[te_m]; teTK_l = allTK[te_m]

    # MI feature selection
    sub_l  = (np.random.RandomState(42).choice(len(trF_l), 2000, replace=False)
              if len(trF_l) > 2000 else np.arange(len(trF_l)))
    mi_l   = mutual_info_classif(trF_l[sub_l], trY_l[sub_l], random_state=42, n_neighbors=5)
    fi_l   = np.argsort(-mi_l)[:FINAL_K]
    trF_lk = trF_l[:, fi_l]
    teF_lk = teF_l[:, fi_l]

    # SMOTE
    try:
        k_nn = min(5, int(np.bincount(trY_l).min()) - 1)
        if k_nn >= 1:
            sm_l = SMOTE(random_state=42, k_neighbors=k_nn)
            trF_lk, trY_lk = sm_l.fit_resample(trF_lk, trY_l)
        else:
            trY_lk = trY_l
    except Exception:
        trY_lk = trY_l

    # PCA
    pca_l  = PCA(n_components=0.95, random_state=42)
    trF_lp = pca_l.fit_transform(trF_lk)
    teF_lp = pca_l.transform(teF_lk)

    # Stacking ensemble with fallback
    try:
        clf_l = build_stack(FINAL_SH)
        clf_l.fit(trF_lp, trY_lk)
        probs_l = clf_l.predict_proba(teF_lp)
    except Exception as ex:
        print(f"  [{loso_sid}] stack failed ({ex}), using LDA fallback")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        clf_l   = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=FINAL_SH)
        clf_l.fit(trF_lp, trY_lk)
        probs_l = clf_l.predict_proba(teF_lp)

    # EMA smoothing + vote per trial
    win_correct = []
    for tkey in sorted(set(teTK_l)):
        m   = (teTK_l == tkey)
        pr  = smooth_probs_ema(probs_l[m], alpha=0.4)
        pd_ = np.argmax(pr, axis=1)
        tl  = int(teY_l[m][0])
        win_correct.extend((pd_ == teY_l[m]).tolist())
        for r in range(len(pd_)):
            loso_win_rows.append({
                "subject": loso_sid, "trial_key": tkey, "window_idx": r,
                "true_idx": tl, "true_label": IDX_TO_LABEL[tl],
                "pred_idx": int(pd_[r]), "pred_label": IDX_TO_LABEL[int(pd_[r])],
            })
        pi, conf, mp = trial_vote_from_probs(pr)
        loso_trial_rows.append({
            "subject": loso_sid, "trial_key": tkey, "n_windows": int(m.sum()),
            "true_idx": tl, "true_label": IDX_TO_LABEL[tl],
            "trial_pred_idx": pi, "trial_pred_label": IDX_TO_LABEL[pi],
            "trial_confidence": conf,
        })

    w_acc = float(np.mean(win_correct))
    loso_accs.append((loso_sid, w_acc, int(te_m.sum())))
    print(f"  Subject {loso_sid}: win_acc={w_acc:.3f}  n_wins={int(te_m.sum())}")

pd.DataFrame(loso_trial_rows).to_csv("/kaggle/working/maxeffort_loso_trial.csv",  index=False)
pd.DataFrame(loso_win_rows).to_csv(  "/kaggle/working/maxeffort_loso_window.csv", index=False)

loso_wdf = pd.DataFrame(loso_win_rows)
loso_tdf = pd.DataFrame(loso_trial_rows)

if len(loso_wdf):
    yt = loso_wdf["true_idx"].astype(int).values
    yp = loso_wdf["pred_idx"].astype(int).values
    print("\n" + "=" * 60)
    print("MAX EFFORT  --  LOSO WINDOW-LEVEL (all subjects pooled)")
    print("=" * 60)
    print(f"Window accuracy: {(yt == yp).mean():.4f}")
    print(classification_report(yt, yp,
          target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))
    print(confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))))

if len(loso_tdf):
    yt = loso_tdf["true_idx"].astype(int).values
    yp = loso_tdf["trial_pred_idx"].astype(int).values
    print("\n" + "=" * 60)
    print("MAX EFFORT  --  LOSO TRIAL-LEVEL (all subjects pooled)")
    print("=" * 60)
    print(f"Trial accuracy: {(yt == yp).mean():.4f}")
    print(classification_report(yt, yp,
          target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)], zero_division=0))
    print(confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))))
    print("\nPer-subject window accuracies:")
    for sid, acc, n in loso_accs:
        print(f"  {sid}: {acc:.3f}  ({n} windows)")
    mean_l = float(np.mean([a for _, a, _ in loso_accs]))
    std_l  = float(np.std( [a for _, a, _ in loso_accs]))
    print(f"\nMean LOSO window accuracy: {mean_l:.4f} +/- {std_l:.4f}")
"""

with open("MaxEffort.py", "w", encoding="utf-8") as f:
    f.write(preprocess_block + new_section)

total = open("MaxEffort.py", encoding="utf-8").read().count("\n")
print(f"MaxEffort.py written: {total} lines")

# Sanity: check no obvious syntax errors
import ast
try:
    ast.parse(open("MaxEffort.py", encoding="utf-8").read())
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax ERROR: {e}")
