
# =============================================================
# EEGNet -- Cross-Subject Emotion Recognition
# Architecture: Lawhern et al. 2018 (arXiv:1611.08024)
# Input:  raw EEG windows  (batch, 1, 4, 2560)  10s @ 256Hz
# Output: 4-class softmax  (NEUTRAL/ENTHUSIASM/SADNESS/FEAR)
# Evaluation: LOSO (Leave-One-Subject-Out)
# =============================================================

import os, json, glob, time, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# CUDA sanity test -- verify the GPU actually executes kernels
# before committing to it (Kaggle sometimes has compute capability mismatches)
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # synchronous errors for clearer tracebacks

def _check_cuda():
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        t = torch.zeros(4, device="cuda")
        t = t + 1.0           # triggers a real kernel
        _ = t.sum().item()    # forces synchronisation
        return torch.device("cuda")
    except Exception as e:
        print(f"[warn] CUDA sanity test failed ({e}) -- falling back to CPU")
        return torch.device("cpu")

DEVICE = _check_cuda()
print(f"Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"Compute: {torch.cuda.get_device_capability(0)}")

# =============================================================
# PATHS + CONSTANTS
# =============================================================
EMOGNITION_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
OUT_LOSO_CSV    = "/kaggle/working/eegnet_loso_results.csv"

EEG_SR      = 256
WINDOW_SEC  = 10
OVERLAP_FRAC = 0.5          # 50% overlap -- less redundancy
EEG_WIN     = WINDOW_SEC * EEG_SR   # 2560 samples
NUM_CLASSES = 4
N_CHANNELS  = 4

EMOTION_LABELS = {"NEUTRAL": 0, "ENTHUSIASM": 1, "SADNESS": 2, "FEAR": 3}
IDX_TO_LABEL   = {v: k for k, v in EMOTION_LABELS.items()}

EEG_CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

# Training hyperparameters
EPOCHS      = 150
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 20           # early stopping patience
WEIGHT_DECAY = 1e-4

# EEGNet architecture params
F1  = 8    # temporal filters
D   = 2    # depthwise multiplier  (F2 = F1*D = 16)
F2  = F1 * D
KERN_LEN = EEG_SR // 2    # 128 -- half-second temporal kernel

# =============================================================
# HELPERS
# =============================================================
def safe_array(x):
    return np.nan_to_num(np.asarray(x, dtype=np.float32),
                         nan=0.0, posinf=0.0, neginf=0.0)

def bandpass_eeg(sig, sr=EEG_SR, low=0.5, high=50.0, order=4):
    nyq = 0.5 * sr
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1).astype(np.float32)

def _interp_nan(a):
    a = a.copy()
    nans = np.isnan(a)
    if not nans.any():
        return a
    idx = np.arange(len(a))
    a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
    return a

def load_subject_baselines(data_root):
    baselines = {}
    for subj in sorted(os.listdir(data_root)):
        subj_dir = os.path.join(data_root, subj)
        if not os.path.isdir(subj_dir) or not subj.isdigit():
            continue
        candidates = sorted(glob.glob(
            os.path.join(subj_dir, "*_BASELINE_MUSE_cleaned.json")))
        if not candidates:
            continue
        try:
            with open(candidates[0]) as f:
                raw = json.load(f)
            df = pd.DataFrame(raw["data"] if isinstance(raw, dict) and "data" in raw else raw)
            cols = [c for c in EEG_CHANNELS if c in df.columns]
            sig  = np.stack([_interp_nan(df[c].to_numpy(dtype=np.float64)) for c in cols], axis=-1)
            baselines[subj] = (sig.mean(0).astype(np.float32),
                               np.where(sig.std(0) < 1e-6, 1.0, sig.std(0)).astype(np.float32))
        except Exception:
            pass
    print(f"[baselines] loaded for {len(baselines)} subjects")
    return baselines

# =============================================================
# EUCLIDEAN ALIGNMENT  (He & Wu 2019)
# =============================================================
def _regularised_cov(X, reg=1e-4):
    C, T = X.shape
    Xc  = X - X.mean(1, keepdims=True)
    cov = (Xc @ Xc.T) / (T - 1)
    return (1-reg)*cov + reg*(np.trace(cov)/C)*np.eye(C)

def _matrix_sqrt_inv(M):
    v, U = np.linalg.eigh(M)
    return U @ np.diag(1.0/np.sqrt(np.maximum(v, 1e-10))) @ U.T

def euclidean_align(windows):
    """windows: list of (4, T) arrays.  Returns aligned list."""
    covs = [_regularised_cov(w.astype(np.float64)) for w in windows]
    R    = np.stack(covs).mean(0)
    Rinv = _matrix_sqrt_inv(R)
    return [(Rinv @ w.astype(np.float64)).astype(np.float32) for w in windows]

# =============================================================
# LOAD + WINDOW RAW EEG
# =============================================================
def load_trials(data_root):
    baselines = load_subject_baselines(data_root)
    global_bl = None
    if baselines:
        ms = np.stack([v[0] for v in baselines.values()])
        ss = np.stack([v[1] for v in baselines.values()])
        global_bl = (ms.mean(0), ss.mean(0))

    trials = []
    step   = WINDOW_SEC * (1 - OVERLAP_FRAC)

    for subj in sorted(os.listdir(data_root)):
        subj_dir = os.path.join(data_root, subj)
        if not os.path.isdir(subj_dir) or not subj.isdigit():
            continue
        for eeg_json in sorted(glob.glob(os.path.join(subj_dir,
                                "*_STIMULUS_MUSE_cleaned.json"))):
            parts   = os.path.basename(eeg_json).split("_")
            emotion = parts[1].upper() if len(parts) >= 2 else None
            if emotion not in EMOTION_LABELS:
                continue
            try:
                with open(eeg_json) as f:
                    ed = json.load(f)
                eeg = np.stack([safe_array(np.array(ed[ch])) for ch in EEG_CHANNELS])
            except Exception:
                continue

            # baseline z-score
            bl = baselines.get(subj, global_bl)
            if bl is not None:
                eeg = ((eeg.T - bl[0]) / bl[1]).T.astype(np.float32)
            eeg = safe_array(eeg)

            # bandpass 0.5-50 Hz
            try:
                eeg = bandpass_eeg(eeg)
            except Exception:
                pass

            T    = eeg.shape[1]
            dur  = T / EEG_SR
            wins = []
            wi   = 0
            while True:
                t_s = wi * step
                e_s = int(t_s * EEG_SR)
                e_e = e_s + EEG_WIN
                if e_e > T:
                    break
                wins.append(eeg[:, e_s:e_e])
                wi += 1

            if not wins:
                continue

            # Euclidean Alignment per trial (all windows of this trial)
            wins = euclidean_align(wins)

            for w_idx, w in enumerate(wins):
                trials.append({
                    "sid":      subj,
                    "label":    EMOTION_LABELS[emotion],
                    "trial_key": f"{subj}_{emotion}",
                    "win_idx":  w_idx,
                    "n_wins":   len(wins),
                    "eeg":      w,          # (4, 2560) float32
                })

    print(f"Total windows loaded: {len(trials)}")
    return trials

# =============================================================
# DATASET
# =============================================================
class EEGWindowDataset(Dataset):
    def __init__(self, trial_list):
        self.data = trial_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        t = self.data[i]
        # shape: (1, C, T) -- EEGNet expects (batch, 1, C, T)
        x = torch.tensor(t["eeg"], dtype=torch.float32).unsqueeze(0)
        y = torch.tensor(t["label"], dtype=torch.long)
        return x, y

# =============================================================
# EEGNet ARCHITECTURE
# =============================================================
class EEGNet(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, n_channels=N_CHANNELS,
                 n_samples=EEG_WIN, F1=F1, D=D, F2=F2,
                 kern_len=KERN_LEN, dropout=0.5):
        super().__init__()

        # Block 1: Temporal conv + Depthwise spatial conv
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len//2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution (across EEG channels)
            nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1*D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Block 2: Separable conv
        self.block2 = nn.Sequential(
            # Depthwise
            nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False),
            # Pointwise
            nn.Conv2d(F2, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            out   = self.block2(self.block1(dummy))
            flat  = out.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat, n_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# =============================================================
# TRAIN / EVAL HELPERS
# =============================================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss/n, correct/n

@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out  = model(x)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
        n          += len(y)
    return total_loss/n, correct/n

@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        probs = F.softmax(model(x), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)

def weighted_trial_vote(probs):
    """Exponential ramp -- later windows weighted more (emotion builds over trial)."""
    T = len(probs)
    if T == 1:
        mp = probs[0]
    else:
        w  = np.exp(np.linspace(0.0, 1.0, T))
        w /= w.sum()
        mp = np.average(probs, axis=0, weights=w)
    pred = int(np.argmax(mp))
    return pred, float(mp[pred]), mp

# =============================================================
# LOSO EVALUATION
# =============================================================
print("\nLoading data ...")
all_trials = load_trials(EMOGNITION_ROOT)
all_sids   = np.array([t["sid"]       for t in all_trials])
all_tkeys  = np.array([t["trial_key"] for t in all_trials])
all_labels = np.array([t["label"]     for t in all_trials])

unique_sids = sorted(set(all_sids))
print(f"Subjects: {len(unique_sids)}  |  Windows: {len(all_trials)}")

loso_win_rows   = []
loso_trial_rows = []
loso_accs       = []

for hold_sid in unique_sids:
    t0    = time.time()
    tr_m  = (all_sids != hold_sid)
    te_m  = (all_sids == hold_sid)

    tr_list = [all_trials[i] for i in np.where(tr_m)[0]]
    te_list = [all_trials[i] for i in np.where(te_m)[0]]

    if not tr_list or not te_list:
        continue

    # class weights to handle imbalance
    labels_tr = np.array([t["label"] for t in tr_list])
    counts    = np.bincount(labels_tr, minlength=NUM_CLASSES).astype(np.float32)
    cw_np     = 1.0 / (counts + 1e-6)
    cw_np    /= cw_np.sum()                           # stay on CPU
    cw        = torch.tensor(cw_np, dtype=torch.float32).to(DEVICE)

    tr_loader = DataLoader(EEGWindowDataset(tr_list),
                           batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    te_loader = DataLoader(EEGWindowDataset(te_list),
                           batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = EEGNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=cw)

    # Early stopping
    best_val_loss = np.inf
    best_state    = None
    patience_ctr  = 0

    # Use 10% of training data as validation (last-subject mini-split)
    val_sids = sorted(set(all_sids[tr_m]))[-max(1, len(set(all_sids[tr_m]))//10):]
    val_m_local = np.array([t["sid"] in val_sids for t in tr_list])
    tr_sub  = [tr_list[i] for i in np.where(~val_m_local)[0]]
    val_sub = [tr_list[i] for i in np.where( val_m_local)[0]]

    tr_sub_loader  = DataLoader(EEGWindowDataset(tr_sub),
                                batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_sub_loader = DataLoader(EEGWindowDataset(val_sub),
                                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for ep in range(EPOCHS):
        tr_loss, tr_acc = train_epoch(model, tr_sub_loader, optimizer, criterion)
        vl_loss, vl_acc = eval_epoch(model, val_sub_loader, criterion)
        scheduler.step()

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"    early stop at epoch {ep+1}")
                break

    # Restore best weights
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Predict on held-out subject
    probs_all, labels_all = predict_proba(model, te_loader)
    preds_all = np.argmax(probs_all, axis=1)
    te_tkeys  = all_tkeys[te_m]

    win_acc = float((preds_all == labels_all).mean())
    loso_accs.append((hold_sid, win_acc, int(te_m.sum())))

    # Trial-level vote
    for tkey in sorted(set(te_tkeys)):
        m        = (te_tkeys == tkey)
        pr       = probs_all[m]
        true_lbl = int(labels_all[m][0])
        for r, p in enumerate(pr):
            loso_win_rows.append({
                "subject":   hold_sid,
                "trial_key": tkey,
                "window_idx": r,
                "true_idx":  true_lbl,
                "true_label": IDX_TO_LABEL[true_lbl],
                "pred_idx":  int(np.argmax(p)),
                "pred_label": IDX_TO_LABEL[int(np.argmax(p))],
            })
        pi, conf, mp = weighted_trial_vote(pr)
        loso_trial_rows.append({
            "subject":          hold_sid,
            "trial_key":        tkey,
            "n_windows":        int(m.sum()),
            "true_idx":         true_lbl,
            "true_label":       IDX_TO_LABEL[true_lbl],
            "trial_pred_idx":   pi,
            "trial_pred_label": IDX_TO_LABEL[pi],
            "trial_confidence": conf,
        })

    elapsed = time.time() - t0
    print(f"  Subject {hold_sid}: win_acc={win_acc:.3f}  "
          f"n_wins={int(te_m.sum())}  time={elapsed:.1f}s")

# =============================================================
# SAVE + REPORT
# =============================================================
loso_win_df   = pd.DataFrame(loso_win_rows)
loso_trial_df = pd.DataFrame(loso_trial_rows)

loso_win_df.to_csv("/kaggle/working/eegnet_loso_window_predictions.csv",  index=False)
loso_trial_df.to_csv("/kaggle/working/eegnet_loso_trial_predictions.csv", index=False)

if len(loso_win_df):
    yt = loso_win_df["true_idx"].astype(int).values
    yp = loso_win_df["pred_idx"].astype(int).values
    print("\n" + "="*60)
    print("EEGNet LOSO -- WINDOW-LEVEL (all subjects pooled)")
    print("="*60)
    print(f"Window accuracy: {(yt==yp).mean():.4f}")
    print(classification_report(yt, yp,
          target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
          zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))))

if len(loso_trial_df):
    yt = loso_trial_df["true_idx"].astype(int).values
    yp = loso_trial_df["trial_pred_idx"].astype(int).values
    print("\n" + "="*60)
    print("EEGNet LOSO -- TRIAL-LEVEL (all subjects pooled)")
    print("="*60)
    print(f"Trial accuracy: {(yt==yp).mean():.4f}")
    print(classification_report(yt, yp,
          target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
          zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))))
    print("\nPer-subject window accuracies:")
    for sid, acc, n in loso_accs:
        print(f"  {sid}: {acc:.3f}  ({n} windows)")
    mean_loso = np.mean([a for _,a,_ in loso_accs])
    std_loso  = np.std( [a for _,a,_ in loso_accs])
    print(f"\nMean LOSO window accuracy: {mean_loso:.4f} +/- {std_loso:.4f}")
