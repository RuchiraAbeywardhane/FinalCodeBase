# ═══════════════════════════════════════════════════════════════
# FINAL TRAIN + TEST SPLIT PIPELINE WITH WINDOW/TRIAL ACCURACY
#   - Train final LDA on Emognition (non-test folds)
#   - Evaluate on held-out dataset split (TEST_FOLD)
#   - Uses only BVP from Samsung watch
#   - HR/PPI are derived from BVP
#   - MUSE band powers are resampled to true 10 Hz
#   - Reports BOTH window-level and trial-level accuracy
# ═══════════════════════════════════════════════════════════════

import os
import json
import glob
import time
import warnings
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from scipy.signal import welch, coherence as sp_coherence, find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

try:
    from pyriemann.estimation import Covariances as _PyrCov
    PYRIEMANN_OK = True
except ImportError:
    PYRIEMANN_OK = False
    print("[info] pyriemann not found -- manual SPD fallback active")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
# ---- TRAINING DATA (Emognition) ----
# All files (cleaned EEG, band powers, BVP) live under one root.
EMOGNITION_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"


# ---- OUTPUT ----
OUT_TRIAL_CSV  = "/kaggle/working/test_trial_predictions.csv"
OUT_WINDOW_CSV = "/kaggle/working/test_window_predictions.csv"

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
# Held-out test fold (0-3). Final model trains on other folds.
TEST_FOLD = 3

EEG_SR = 256

# True band power update rate
BAND_SR = 10

# Training JSON band powers are stored on 256 Hz grid with duplicates
TRAIN_BAND_STORED_SR = 256

# Training Samsung watch BVP sampling rate
TRAIN_BVP_SR = 20

WINDOW_SEC = 10
OVERLAP_FRAC = 0.75

EEG_WIN  = WINDOW_SEC * EEG_SR
BAND_WIN = WINDOW_SEC * BAND_SR
TRAIN_BVP_WIN = int(WINDOW_SEC * TRAIN_BVP_SR)

# Baseline reduction applied to raw EEG before feature extraction
# Options: 'invbase' | 'zscore' | 'subtract'
BASELINE_METHOD = "zscore"
BASELINE_EPS    = 1e-12

NUM_CLASSES = 4

EMOTION_LABELS = {
    "NEUTRAL": 0,
    "ENTHUSIASM": 1,
    "SADNESS": 2,
    "FEAR": 3,
}
IDX_TO_LABEL = {v: k for k, v in EMOTION_LABELS.items()}

EEG_CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

BAND_CHANNELS = [
    "Alpha_TP9",  "Alpha_AF7",  "Alpha_AF8",  "Alpha_TP10",
    "Beta_TP9",   "Beta_AF7",   "Beta_AF8",   "Beta_TP10",
    "Delta_TP9",  "Delta_AF7",  "Delta_AF8",  "Delta_TP10",
    "Gamma_TP9",  "Gamma_AF7",  "Gamma_AF8",  "Gamma_TP10",
    "Theta_TP9",  "Theta_AF7",  "Theta_AF8",  "Theta_TP10",
]

# Feature sizes
N_FEAT_EEG  = 156
N_FEAT_MUSE = 62
N_FEAT_BVP  = 7
N_FEAT_HR   = 5
N_FEAT_PPI  = 8
N_FEAT_FAA  = 14   # Frontal Alpha Asymmetry + hemispheric asymmetries
N_FEAT_RIEM = 10   # Riemannian tangent-space upper triangle of 4x4 log-cov
N_FEATURES_RAW = (N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP +
                  N_FEAT_HR  + N_FEAT_PPI  + N_FEAT_FAA + N_FEAT_RIEM)

print(f"Raw feature count = {N_FEATURES_RAW}")

# ???????????????????????????????????????????????????????????????
# BASELINE REDUCTION HELPERS
# ???????????????????????????????????????????????????????????????

def _interp_nan(a):
    a = a.copy()
    nans = np.isnan(a)
    if not nans.any():
        return a
    idx = np.arange(len(a))
    a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
    return a


def apply_baseline_reduction(signal, baseline, eps=BASELINE_EPS, method=BASELINE_METHOD):
    signal   = np.nan_to_num(np.asarray(signal,   dtype=np.float64))
    baseline = np.nan_to_num(np.asarray(baseline, dtype=np.float64))
    if baseline.ndim == 2:
        base_mean = np.mean(baseline, axis=0)
        base_std  = np.std(baseline,  axis=0)
    else:
        base_mean = baseline
        base_std  = np.ones_like(base_mean)
    m = method.lower().strip()
    if m == "invbase":
        reduced = signal / (base_mean + eps)
    elif m == "zscore":
        reduced = (signal - base_mean) / (base_std + eps)
    elif m == "subtract":
        reduced = signal - base_mean
    else:
        raise ValueError(f"Unknown baseline method '{method}'.")
    return reduced.astype(np.float32)


def load_subject_baselines(data_root, baseline_keyword="BASELINE"):
    baselines = {}
    for subj in sorted(os.listdir(data_root)):
        subj_dir = os.path.join(data_root, subj)
        if not os.path.isdir(subj_dir) or not subj.isdigit():
            continue

        # Primary: look for the cleaned MUSE baseline file
        # Expected name: <sid>_BASELINE_MUSE_cleaned.json
        candidates = sorted(glob.glob(
            os.path.join(subj_dir, f"*_{baseline_keyword}_MUSE_cleaned.json")
        ))

        # Fallback: any JSON containing the keyword AND "muse" in the name
        if not candidates:
            candidates = [
                fp for fp in glob.glob(os.path.join(subj_dir, "*.json"))
                if baseline_keyword.lower() in os.path.basename(fp).lower()
                and "muse" in os.path.basename(fp).lower()
            ]

        if not candidates:
            print(f"  [baseline] no MUSE baseline file for subject '{subj}' -- skipping.")
            continue

        fp = candidates[0]
        try:
            with open(fp, "r") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict) and "data" in raw:
                df = pd.DataFrame(raw["data"])
            elif isinstance(raw, list):
                df = pd.DataFrame(raw)
            else:
                df = pd.DataFrame(raw)
            eeg_cols = [c for c in EEG_CHANNELS if c in df.columns]
            if not eeg_cols:
                print(f"  [baseline] no EEG columns in '{fp}' -- skipping.")
                continue
            sig = np.stack(
                [_interp_nan(np.nan_to_num(df[c].to_numpy(dtype=np.float64)))
                 for c in eeg_cols],
                axis=-1,
            )
            base_mean = np.mean(sig, axis=0).astype(np.float32)
            base_std  = np.std(sig,  axis=0).astype(np.float32)
            # clamp std to avoid division by zero on flat channels
            base_std  = np.where(base_std < 1e-6, 1.0, base_std)
            baselines[subj] = (base_mean, base_std)
            print(f"  [baseline] subject={subj}  file={os.path.basename(fp)}  channels={len(eeg_cols)}  samples={len(sig)}")
        except Exception as exc:
            print(f"  [baseline] failed to load '{fp}': {exc}")
    print(f"[load_subject_baselines] Loaded baselines for {len(baselines)} subject(s).")
    return baselines


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════
def safe_array(x):
    return np.nan_to_num(np.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)

def infer_sampling_rate_from_time_series(t):
    t = np.asarray(t, dtype=np.float64)
    if len(t) < 2:
        return None
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return 1.0 / np.median(dt)

def resample_1d_by_time(arr, orig_sr, target_sr):
    arr = safe_array(np.asarray(arr, dtype=np.float32))
    if len(arr) == 0:
        return arr.astype(np.float32)

    old_t = np.arange(len(arr), dtype=np.float64) / float(orig_sr)
    new_len = int(np.floor(len(arr) * float(target_sr) / float(orig_sr)))
    new_len = max(new_len, 1)
    new_t = np.arange(new_len, dtype=np.float64) / float(target_sr)

    out = np.interp(new_t, old_t, arr)
    return out.astype(np.float32)

def resample_multich_by_time(arr2d, orig_sr, target_sr):
    return np.vstack([
        resample_1d_by_time(arr2d[i], orig_sr, target_sr)
        for i in range(arr2d.shape[0])
    ])

def parse_emotion_from_training_trial_name(trial_name):
    parts = trial_name.split("_")
    if len(parts) < 2:
        return None
    emo = parts[1].upper()
    return emo if emo in EMOTION_LABELS else None

def parse_trial_key_inference(base_name):
    if base_name.endswith("_eeg.csv"):
        return base_name[:-8]
    if base_name.endswith("_ppg_hr_ibi.csv"):
        return base_name[:-15]
    return os.path.splitext(base_name)[0]

def parse_true_label_from_infer_trial_key(trial_key):
    low = trial_key.lower()
    if low.startswith("neutral"):
        return "NEUTRAL"
    if low.startswith("enthusiasm"):
        return "ENTHUSIASM"
    if low.startswith("sad"):
        return "SADNESS"
    if low.startswith("fear"):
        return "FEAR"
    return None

def trial_vote_from_probs(probs, num_classes=NUM_CLASSES, conf_threshold=0.0):
    # Confidence-weighted soft voting.
    # Each window weighted by its peak softmax prob.
    # Windows below conf_threshold get zero weight.
    probs = np.asarray(probs, dtype=np.float64)
    win_conf = probs.max(axis=1)
    weights  = np.where(win_conf >= conf_threshold, win_conf, 0.0)
    if weights.sum() < 1e-10:
        weights = np.ones(len(probs))
    weights   = weights / weights.sum()
    mean_prob = (probs * weights[:, None]).sum(axis=0)
    pred_idx  = int(np.argmax(mean_prob))
    conf      = float(mean_prob[pred_idx])
    return pred_idx, conf, mean_prob

# ═══════════════════════════════════════════════════════════════
# BVP -> HR / PPI
# ═══════════════════════════════════════════════════════════════
def bandpass_bvp(sig, sr, low=0.7, high=3.5, order=2):
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < max(10, order * 3):
        return sig

    nyq = 0.5 * sr
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.999)

    if low_n >= high_n:
        return sig

    try:
        b, a = butter(order, [low_n, high_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig

def derive_hr_ppi_from_bvp(bvp_window, sr):
    sig = safe_array(np.asarray(bvp_window, dtype=np.float64))
    if len(sig) < 8:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    sig_f = bandpass_bvp(sig, sr=sr, low=0.7, high=3.5, order=2)

    sstd = np.std(sig_f)
    if sstd > 1e-10:
        sig_n = (sig_f - np.mean(sig_f)) / sstd
    else:
        sig_n = sig_f - np.mean(sig_f)

    min_dist = max(1, int(0.35 * sr))

    try:
        peaks, _ = find_peaks(sig_n, distance=min_dist, prominence=0.2)
    except Exception:
        peaks = np.array([], dtype=int)

    if len(peaks) < 2:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    ibi_sec = np.diff(peaks) / float(sr)
    ibi_sec = ibi_sec[(ibi_sec >= 0.35) & (ibi_sec <= 1.5)]

    if len(ibi_sec) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    ppi_vals_ms = (ibi_sec * 1000.0).astype(np.float32)
    hr_vals_bpm = (60.0 / ibi_sec).astype(np.float32)

    return hr_vals_bpm, ppi_vals_ms

# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════
FREQ_BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 100)]
CH_PAIRS = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]

def _compute_psd(sig, sr, nperseg=256):
    return welch(sig, fs=sr, nperseg=min(nperseg, len(sig)))

def _band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    return np.mean(psd[mask]) if mask.any() else 1e-10

def _differential_entropy(sig):
    v = np.var(sig)
    return 0.5 * np.log(2 * np.pi * np.e * v) if v > 1e-12 else 0.0

def _hjorth(sig):
    d1 = np.diff(sig)
    d2 = np.diff(d1)
    act = np.var(sig)
    if act < 1e-12:
        return 0.0, 0.0, 0.0
    mob = np.sqrt(np.var(d1) / act)
    v_d1 = np.var(d1)
    if v_d1 < 1e-12:
        return float(act), float(mob), 0.0
    comp = np.sqrt(np.var(d2) / v_d1) / mob if mob > 1e-12 else 0.0
    return float(act), float(mob), float(comp)

def _zcr(sig):
    return float(np.sum(np.abs(np.diff(np.sign(sig))) > 0)) / max(len(sig) - 1, 1)

def spectral_entropy(sig, sr=EEG_SR, nperseg=256):
    _, psd = welch(sig, fs=sr, nperseg=min(nperseg, len(sig)))
    psd_n = psd / (psd.sum() + 1e-12)
    psd_n = psd_n[psd_n > 0]
    return float(-np.sum(psd_n * np.log2(psd_n))) if len(psd_n) > 0 else 0.0

def permutation_entropy(sig, order=3, delay=1):
    n = len(sig)
    if n < (order - 1) * delay + 1:
        return 0.0
    indices = np.arange(n - (order - 1) * delay)
    cols = np.column_stack([sig[indices + d * delay] for d in range(order)])
    perms = np.argsort(cols, axis=1)
    encoded = np.zeros(perms.shape[0], dtype=np.int64)
    for i in range(order):
        encoded = encoded * order + perms[:, i]
    _, counts = np.unique(encoded, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))

def wavelet_subband_energy(sig, wavelet="db4", level=5):
    try:
        import pywt
        max_lev = pywt.dwt_max_level(len(sig), wavelet)
        coeffs = pywt.wavedec(sig, wavelet, level=min(level, max_lev))
        energies = [float(np.mean(c**2)) for c in coeffs[:5]]
        while len(energies) < 5:
            energies.append(0.0)
        return energies
    except Exception:
        return [0.0] * 5

# 156 EEG features
def extract_eeg_features(eeg_4ch, sr=EEG_SR):
    feats = []
    bp = np.zeros((4, 5), dtype=np.float64)

    for ch in range(4):
        sig = eeg_4ch[ch].astype(np.float64)
        freqs, psd = _compute_psd(sig, sr)

        for bi, (lo, hi) in enumerate(FREQ_BANDS):
            pw = _band_power(freqs, psd, lo, hi)
            bp[ch, bi] = np.log1p(pw)
        feats.extend(bp[ch].tolist())

        for bi, (lo, hi) in enumerate(FREQ_BANDS):
            pw = _band_power(freqs, psd, lo, hi)
            feats.append(0.5 * np.log(2 * np.pi * np.e * pw) if pw > 1e-12 else 0.0)

        feats.extend(_hjorth(sig))
        feats.extend([
            float(np.mean(sig)),
            float(np.std(sig)),
            float(skew(sig)),
            float(kurtosis(sig))
        ])
        feats.append(_zcr(sig))
        feats.append(_differential_entropy(sig))
        feats.append(spectral_entropy(sig, sr))
        feats.append(permutation_entropy(sig))

    for ch in range(4):
        feats.extend(wavelet_subband_energy(eeg_4ch[ch].astype(np.float64)))

    for ch in range(4):
        a, b, t = bp[ch, 2], bp[ch, 3], bp[ch, 1]
        feats.extend([a - b, a - t, t - b])

    for bi in range(5):
        feats.append(bp[2, bi] - bp[1, bi])
    for bi in range(5):
        feats.append(bp[3, bi] - bp[0, bi])

    for (ci, cj) in CH_PAIRS:
        try:
            f_coh, coh = sp_coherence(
                eeg_4ch[ci].astype(np.float64),
                eeg_4ch[cj].astype(np.float64),
                fs=sr,
                nperseg=min(256, len(eeg_4ch[ci]))
            )
            for bi, (lo, hi) in enumerate(FREQ_BANDS):
                mask = (f_coh >= lo) & (f_coh < hi)
                feats.append(float(np.mean(coh[mask])) if mask.any() else 0.0)
        except Exception:
            feats.extend([0.0] * 5)

    return safe_array(np.array(feats, dtype=np.float32))

# 62 band features
def extract_band_features(band_window):
    feats = []

    for ch in range(20):
        col = band_window[ch]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            feats.extend([0.0, 0.0])
        else:
            feats.append(float(np.mean(col)))
            feats.append(float(np.std(col)) + 1e-8)

    def bp(bi, ei):
        col = band_window[bi * 4 + ei]
        col = col[np.isfinite(col)]
        return float(np.mean(col)) if len(col) > 0 else 0.0

    for ei in range(4):
        a, b, t = bp(0, ei), bp(1, ei), bp(4, ei)
        db = b if abs(b) > 1e-6 else 1e-6
        dt = t if abs(t) > 1e-6 else 1e-6
        feats.extend([a / db, a / dt, t / db])

    for bi in range(5):
        feats.append(bp(bi, 2) - bp(bi, 1))
    for bi in range(5):
        feats.append(bp(bi, 3) - bp(bi, 0))

    return safe_array(np.clip(np.array(feats, dtype=np.float32), -1e4, 1e4))

# 7 BVP features
def extract_bvp_features(bvp_window, sr):
    sig = bvp_window.astype(np.float64)
    feats = [
        float(np.mean(sig)),
        float(np.std(sig)),
        float(skew(sig)),
        float(kurtosis(sig))
    ]

    fft_mag = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    mask_hr = (freqs > 0.5) & (freqs < 4.0)

    feats.append(
        float(freqs[mask_hr][np.argmax(fft_mag[mask_hr])])
        if mask_hr.any() and fft_mag[mask_hr].max() > 0 else 0.0
    )
    feats.append(_zcr(sig))
    feats.append(
        float(np.sum(freqs * fft_mag) / fft_mag.sum())
        if len(fft_mag) > 1 and fft_mag.sum() > 0 else 0.0
    )

    return safe_array(np.array(feats, dtype=np.float32))

# 5 HR features
def extract_hr_features(hr_values):
    if len(hr_values) < 2:
        return np.zeros(5, dtype=np.float32)

    hr = hr_values.astype(np.float64)
    return safe_array(np.array([
        float(np.mean(hr)),
        float(np.std(hr)),
        float(np.min(hr)),
        float(np.max(hr)),
        float(np.max(hr) - np.min(hr))
    ], dtype=np.float32))

# 8 PPI features
def extract_ppi_features(ppi_values):
    if len(ppi_values) < 3:
        return np.zeros(8, dtype=np.float32)

    ipi = ppi_values.astype(np.float64)
    ipi_s = ipi / 1000.0 if np.median(ipi) > 10 else ipi.copy()

    feats = [float(np.mean(ipi_s)), float(np.std(ipi_s))]
    sd = np.diff(ipi_s)

    feats.append(float(np.sqrt(np.mean(sd**2))) if len(sd) > 0 else 0.0)
    feats.append(float(np.mean(np.abs(sd) > 0.05)) if len(sd) > 0 else 0.0)
    feats.append(float(np.std(sd)) if len(sd) > 0 else 0.0)

    if len(ipi_s) > 6:
        t_ipi = np.cumsum(ipi_s)
        t_uniform = np.arange(t_ipi[0], t_ipi[-1], 0.25)
        if len(t_uniform) > 8:
            ipi_uniform = np.interp(t_uniform, t_ipi, ipi_s)
            f_ipi, psd_ipi = welch(ipi_uniform, fs=4.0, nperseg=min(len(ipi_uniform), 32))
            lf_m = (f_ipi >= 0.04) & (f_ipi < 0.15)
            hf_m = (f_ipi >= 0.15) & (f_ipi < 0.4)
            lf = float(np.mean(psd_ipi[lf_m])) if lf_m.any() else 0.0
            hf = float(np.mean(psd_ipi[hf_m])) if hf_m.any() else 0.0
            feats.extend([lf, hf, lf / hf if hf > 1e-10 else 0.0])
        else:
            feats.extend([0.0, 0.0, 0.0])
    else:
        feats.extend([0.0, 0.0, 0.0])

    return safe_array(np.array(feats, dtype=np.float32))

# ===============================================================
# FAA -- Frontal Alpha Asymmetry  (14 features)
# Channel order: 0=TP9  1=AF7(left)  2=AF8(right)  3=TP10
# FAA = log(alpha_AF8) - log(alpha_AF7)
# Reference: Davidson 1992; Harmon-Jones & Allen 1997
# ===============================================================
def extract_faa_features(eeg_4ch, sr=EEG_SR):
    feats = []
    ALPHA=(8,13); BETA=(13,30); THETA=(4,8); GAMMA=(30,45); DELTA=(1,4)
    def _logbp(sig, lo, hi):
        freqs, psd = _compute_psd(sig, sr)
        return float(np.log(_band_power(freqs, psd, lo, hi) + 1e-12))
    feats.append(_logbp(eeg_4ch[2],*ALPHA) - _logbp(eeg_4ch[1],*ALPHA))  # 1 core FAA
    feats.append(_logbp(eeg_4ch[2],*BETA)  - _logbp(eeg_4ch[1],*BETA))   # 2 frontal beta asym
    feats.append(_logbp(eeg_4ch[2],*THETA) - _logbp(eeg_4ch[1],*THETA))  # 3 frontal theta asym
    feats.append(_logbp(eeg_4ch[2],*GAMMA) - _logbp(eeg_4ch[1],*GAMMA))  # 4 frontal gamma asym
    for lo,hi in [ALPHA,BETA,THETA,GAMMA,DELTA]:                          # 5-9 posterior asym
        feats.append(_logbp(eeg_4ch[3],lo,hi) - _logbp(eeg_4ch[0],lo,hi))
    a7=_logbp(eeg_4ch[1],*ALPHA); a8=_logbp(eeg_4ch[2],*ALPHA)
    a9=_logbp(eeg_4ch[0],*ALPHA); a10=_logbp(eeg_4ch[3],*ALPHA)
    feats.append((a8-a7)/(abs(a8+a7)+1e-8))                              # 10 frontal lat index
    feats.append((a10-a9)/(abs(a10+a9)+1e-8))                            # 11 posterior lat index
    b7=_logbp(eeg_4ch[1],*BETA); b8=_logbp(eeg_4ch[2],*BETA)
    feats.append(a7/(a7+b7+1e-8))                                        # 12 AF7 engagement
    feats.append(a8/(a8+b8+1e-8))                                        # 13 AF8 engagement
    t7=_logbp(eeg_4ch[1],*THETA); t8=_logbp(eeg_4ch[2],*THETA)
    feats.append(((t7+t8)/2)/((a7+a8)/2+1e-8))                          # 14 theta/alpha ratio
    return safe_array(np.array(feats, dtype=np.float32))


# ===============================================================
# RIEMANNIAN GEOMETRY -- tangent space of SPD manifold (10 features)
# Steps: regularised cov -> matrix log -> upper triangle
# Reference: Barachant et al. 2010, 2012
# ===============================================================
def _sym_matrix_logm(M):
    v, U = np.linalg.eigh(M)
    return U @ np.diag(np.log(np.maximum(v, 1e-10))) @ U.T

def _regularised_cov(X, reg=1e-4):
    C, T = X.shape
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = (Xc @ Xc.T) / (T - 1)
    return (1-reg)*cov + reg*(np.trace(cov)/C)*np.eye(C)

def extract_riemannian_features(eeg_4ch):
    cov  = _regularised_cov(eeg_4ch.astype(np.float64))
    logC = _sym_matrix_logm(cov)
    return safe_array(logC[np.triu_indices(4)].astype(np.float32))


# ===============================================================
# EUCLIDEAN ALIGNMENT  (He & Wu 2019)
# C_aligned = R^{-1/2} @ C @ R^{-1/2},  R = mean SPD cov
# Unsupervised per-subject domain adaptation on SPD manifold
# ===============================================================
def _matrix_sqrt_inv(M):
    v, U = np.linalg.eigh(M)
    return U @ np.diag(1.0/np.sqrt(np.maximum(v, 1e-10))) @ U.T

def euclidean_alignment(eeg_windows_list):
    covs = [_regularised_cov(w.astype(np.float64)) for w in eeg_windows_list]
    R    = np.stack(covs, axis=0).mean(axis=0)
    Rinv = _matrix_sqrt_inv(R)
    return [Rinv @ w.astype(np.float64) for w in eeg_windows_list]

# ===============================================================
# CORAL -- CORrelation ALignment  (Sun & Saenko 2016)
# Aligns target feature covariance to match source covariance.
# Fully unsupervised: requires only unlabelled target windows.
# ===============================================================
def _matrix_sqrt(M):
    v, U = np.linalg.eigh(M)
    return U @ np.diag(np.sqrt(np.maximum(v, 1e-10))) @ U.T

def coral_align(Xs, Xt):
    Xs = Xs.astype(np.float64)
    Xt = Xt.astype(np.float64)
    d  = Xs.shape[1]
    Cs = np.cov(Xs, rowvar=False) + 1e-5 * np.eye(d)
    Ct = np.cov(Xt, rowvar=False) + 1e-5 * np.eye(d)
    A  = _matrix_sqrt_inv(Ct) @ _matrix_sqrt(Cs)
    Xt_aligned  = (Xt - Xt.mean(axis=0)) @ A
    Xt_aligned += Xs.mean(axis=0)
    return Xt_aligned.astype(np.float32)



# ═══════════════════════════════════════════════════════════════
# LOAD TRAINING DATA
# ═══════════════════════════════════════════════════════════════
print("\nLoading training trials from Emognition ...")
t0 = time.time()

# -- Load per-subject baselines --
print("Loading per-subject EEG baselines ...")
subject_baselines = load_subject_baselines(EMOGNITION_ROOT, baseline_keyword="BASELINE")
if subject_baselines:
    all_means = np.stack([v[0] for v in subject_baselines.values()], axis=0)
    all_stds  = np.stack([v[1] for v in subject_baselines.values()], axis=0)
    global_baseline = (np.mean(all_means, axis=0), np.mean(all_stds, axis=0))
    print(f"Global fallback baseline from {len(subject_baselines)} subjects.")
else:
    global_baseline = None
    print("[warning] No baseline files found -- reduction skipped.")

train_trials = []
skipped = defaultdict(int)

for subj in sorted(os.listdir(EMOGNITION_ROOT)):
    subj_dir = os.path.join(EMOGNITION_ROOT, subj)
    if not os.path.isdir(subj_dir) or not subj.isdigit():
        continue

    cleaned_files = sorted(glob.glob(os.path.join(subj_dir, "*_STIMULUS_MUSE_cleaned.json")))
    if not cleaned_files:
        continue

    for eeg_json in cleaned_files:
        base = os.path.basename(eeg_json)           # e.g. 01_NEUTRAL_STIMULUS_MUSE_cleaned.json
        parts = base.split("_")
        emotion = parts[1].upper() if len(parts) >= 2 else None
        if emotion not in EMOTION_LABELS:
            skipped["bad_trial_name"] += 1
            continue

        label_idx = EMOTION_LABELS[emotion]
        sid = subj

        try:
            with open(eeg_json, "r") as f:
                ed = json.load(f)
            eeg = np.stack([np.array(ed[ch], dtype=np.float32) for ch in EEG_CHANNELS])
            eeg = safe_array(eeg)   # (C, T)
        except Exception:
            skipped["bad_eeg_json"] += 1
            continue

        # -- Apply per-subject baseline reduction --
        baseline_entry = subject_baselines.get(sid, global_baseline)
        if baseline_entry is not None:
            base_mean, base_std = baseline_entry
            # zscore: (signal - mean) / std, channel-wise
            eeg = ((eeg.T - base_mean) / base_std).T.astype(np.float32)
            eeg = safe_array(eeg)

        T_eeg = eeg.shape[1]

        muse_json = os.path.join(EMOGNITION_ROOT, sid, f"{sid}_{emotion}_STIMULUS_MUSE.json")
        if not os.path.isfile(muse_json):
            skipped["missing_muse_json"] += 1
            continue

        try:
            with open(muse_json, "r") as f:
                md = json.load(f)

            band_list = []
            for bch in BAND_CHANNELS:
                raw_band = safe_array(np.array(md[bch], dtype=np.float32))
                true_band = resample_1d_by_time(raw_band, TRAIN_BAND_STORED_SR, BAND_SR)
                band_list.append(true_band)

            min_band = min(len(x) for x in band_list)
            band_arr = np.stack([x[:min_band] for x in band_list], axis=0)
        except Exception:
            skipped["bad_muse_json"] += 1
            continue

        sw_json = os.path.join(EMOGNITION_ROOT, sid, f"{sid}_{emotion}_STIMULUS_SAMSUNG_WATCH.json")
        if not os.path.isfile(sw_json):
            skipped["missing_watch_json"] += 1
            continue

        try:
            with open(sw_json, "r") as f:
                sw = json.load(f)

            if "BVPProcessed" not in sw:
                skipped["missing_BVPProcessed"] += 1
                continue

            bvp_vals = safe_array(np.array([r[1] for r in sw["BVPProcessed"]], dtype=np.float32))
        except Exception:
            skipped["bad_watch_json"] += 1
            continue

        dur = T_eeg / EEG_SR
        dur = min(dur, band_arr.shape[1] / BAND_SR)
        dur = min(dur, len(bvp_vals) / TRAIN_BVP_SR)

        eeg      = eeg[:, :int(dur * EEG_SR)]
        band_arr = band_arr[:, :int(dur * BAND_SR)]
        bvp_vals = bvp_vals[:int(dur * TRAIN_BVP_SR)]

        train_trials.append({
            "sid": sid,
            "emotion": emotion,
            "label": label_idx,
            "eeg": eeg,
            "band": band_arr,
            "bvp": bvp_vals,
            "trial_key": f"{sid}_{emotion}"
        })

print(f"Training trials loaded: {len(train_trials)} in {time.time()-t0:.1f}s")
if skipped:
    print("Skipped:", dict(skipped))

# ═══════════════════════════════════════════════════════════════
# WINDOW TRAINING DATA
# ═══════════════════════════════════════════════════════════════
print("\nWindowing training trials ...")
t1 = time.time()

step = WINDOW_SEC * (1 - OVERLAP_FRAC)

all_feat_w = []
all_labels_w = []
all_tkeys_w = []
all_sids_w = []
all_tidx_w = []

for ti, tr in enumerate(train_trials):
    sid = tr["sid"]
    lbl = tr["label"]
    eeg = tr["eeg"]
    band = tr["band"]
    bvp = tr["bvp"]
    tkey = tr["trial_key"]

    dur = eeg.shape[1] / EEG_SR
    dur = min(dur, band.shape[1] / BAND_SR)
    dur = min(dur, len(bvp) / TRAIN_BVP_SR)

    n_wins = int((dur - WINDOW_SEC) / step) + 1
    if n_wins <= 0:
        continue

    for wi in range(n_wins):
        t_start = wi * step

        e_s = int(t_start * EEG_SR);       e_e = e_s + EEG_WIN
        m_s = int(t_start * BAND_SR);      m_e = m_s + BAND_WIN
        b_s = int(t_start * TRAIN_BVP_SR); b_e = b_s + TRAIN_BVP_WIN

        if e_e > eeg.shape[1] or m_e > band.shape[1] or b_e > len(bvp):
            break

        ew = eeg[:, e_s:e_e]
        mw = band[:, m_s:m_e]
        bw = bvp[b_s:b_e]

        hr_win, ppi_win = derive_hr_ppi_from_bvp(bw, sr=TRAIN_BVP_SR)

        f_eeg  = extract_eeg_features(ew)
        f_muse = extract_band_features(mw)
        f_bvp  = extract_bvp_features(bw, sr=TRAIN_BVP_SR)
        f_hr   = extract_hr_features(hr_win)
        f_ppi  = extract_ppi_features(ppi_win)
        f_faa  = extract_faa_features(ew)          # 14 FAA features
        f_riem = extract_riemannian_features(ew)   # 10 Riemannian (pre-EA)

        feat = np.concatenate(
            [f_eeg, f_muse, f_bvp, f_hr, f_ppi, f_faa, f_riem]
        ).astype(np.float32)

        all_feat_w.append(feat)
        all_labels_w.append(lbl)
        all_tkeys_w.append(tkey)
        all_sids_w.append(sid)
        all_tidx_w.append(ti)

allF_raw  = np.array(all_feat_w, dtype=np.float32)
allY      = np.array(all_labels_w)
allTK     = np.array(all_tkeys_w)
allSID    = np.array(all_sids_w)
allTI     = np.array(all_tidx_w)
allIS_AUG = np.zeros(len(allY), dtype=bool)   # False = real window

print(f"Training windows: {len(allY)} in {time.time()-t1:.1f}s")

# ================================================================
# BALANCED FOLD ASSIGNMENT  moved BEFORE augmentation & EA
# Fold assignment must precede both so they can be fold-aware.
print("\nBalanced fold assignment ...")

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

# FEATURE-SPACE DATA AUGMENTATION  (training windows only)
# FIX: MixUp partner is drawn from the SAME fold only, so synthetic
#      training samples never contain information from test-fold windows.
AUG_NOISE_COPIES = 2
AUG_MIXUP_COPIES = 1
AUG_NOISE_STD    = 0.05
AUG_MIXUP_ALPHA  = 0.4

print("Augmenting training windows ...")
rng_aug  = np.random.RandomState(7)
feat_std = np.std(allF_raw[win_pos_orig != TEST_FOLD], axis=0) + 1e-8  # train-fold only

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
            allF_raw[gidxs[k], RIEM_START:RIEM_START+N_FEAT_RIEM] =                 extract_riemannian_features(np.array(aligned, dtype=np.float32))

print("EA done -- R fitted on train-fold only, applied to all windows (no test leakage).")

# Cache per-subject mean SPD covariance for LOSO cross-subject EA.
# LOSO: the test subject's R must be built from OTHER subjects only.
_sid_mean_cov = {}
for _s in sorted(_sid_to_train_ew.keys()):
    _ews = _sid_to_train_ew[_s]
    if len(_ews) >= 2:
        _covs = [_regularised_cov(w.astype(np.float64)) for w in _ews]
        _sid_mean_cov[_s] = np.stack(_covs).mean(axis=0)

# Cache raw EEG window per global row index so LOSO can recompute
# Riemannian features with the cross-subject R (no self-leakage).
_gidx_to_eeg_win = {}
for _s in sorted(_sid_to_all_ew.keys()):
    _gidxs = _sid_to_gidx[_s]
    for _k, _w in enumerate(_sid_to_all_ew[_s]):
        if _k < len(_gidxs):
            _gidx_to_eeg_win[_gidxs[_k]] = _w

print(f"[EA cache] mean-cov for {len(_sid_mean_cov)} subjects; "
      f"EEG windows cached for {len(_gidx_to_eeg_win)} rows.")

# PREPROCESS  -  fit on non-test-fold windows only (no leakage)
# ═══════════════════════════════════════════════════════════════
print("\nApplying preprocessing (fit on non-TEST_FOLD windows only) ...")

tr_win_mask = (win_pos != TEST_FOLD)   # windows used for fitting preprocessors

# ── 1. VarianceThreshold: fit on train windows only, transform all ──
vt = VarianceThreshold(threshold=0.001)
vt.fit(allF_raw[tr_win_mask])
allF_sel = vt.transform(allF_raw)
print(f"VarianceThreshold kept {allF_sel.shape[1]} / {N_FEATURES_RAW}")

# ── 2. Per-subject z-score: stats from train windows only ──────────
allF_n = np.empty_like(allF_sel)
train_sid_stats = {}

for sid in sorted(set(allSID)):
    # fit mask: this subject's windows that are NOT in the test fold
    fit_mask = (allSID == sid) & tr_win_mask
    all_mask = (allSID == sid)

    if fit_mask.sum() == 0:
        # edge case: subject has no training windows (shouldn't happen)
        allF_n[all_mask] = allF_sel[all_mask]
        continue

    mu = allF_sel[fit_mask].mean(axis=0)
    sd = allF_sel[fit_mask].std(axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    train_sid_stats[sid] = (mu, sd)
    allF_n[all_mask] = (allF_sel[all_mask] - mu) / sd

allF_n = np.clip(safe_array(allF_n), -10, 10)
N_FEATURES = allF_n.shape[1]
print("Final feature dims:", N_FEATURES)

# ═══════════════════════════════════════════════════════════════


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
    # Causal exponential moving average over consecutive windows.
    out = probs.copy().astype(np.float64)
    for t in range(1, len(out)):
        out[t] = alpha * probs[t] + (1 - alpha) * out[t - 1]
    return out

def subject_recentre(trF, teF):
    # Shift test features so their mean matches training mean (unsupervised).
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
# PER-FOLD MI  (restricted to non-TEST_FOLD folds only)
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

best_val, FINAL_K, FINAL_SH = -1.0, 100, 0.3
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
sub_g = (np.random.RandomState(42).choice(len(trF_all), 2000, replace=False)
         if len(trF_all) > 2000 else np.arange(len(trF_all)))
mi_g  = mutual_info_classif(trF_all[sub_g], trY_all[sub_g], random_state=42, n_neighbors=5)
fi_g  = np.argsort(-mi_g)[:FINAL_K]

trF = trF_all[:, fi_g]
teF = teF_all[:, fi_g]

# ── SMOTE: oversample minority classes (train only) ─────────────
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

# ── PCA: decorrelate, keep 95% variance ─────────────────────────
pca = PCA(n_components=0.95, random_state=42)
trF_pca = pca.fit_transform(trF_sm)
teF_pca = pca.transform(teF)
print(f"  PCA: {trF_sm.shape[1]} -> {trF_pca.shape[1]} components")

# ── Stacking ensemble ───────────────────────────────────────────
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
    # FIX leakage: MixUp aug windows of other subjects may contain blended
    # features from the test subject, so exclude ALL augmented windows from
    # LOSO training. Only use real (non-augmented) windows of other subjects.
    tr_m = (allSID != loso_sid) & ~allIS_AUG
    if te_m.sum() == 0 or tr_m.sum() == 0:
        continue

    # LOSO EA FIX: rebuild R from other subjects only, patch test-subject Riem features
    _other_covs = [_sid_mean_cov[_s] for _s in _sid_mean_cov if _s != loso_sid]
    if len(_other_covs) >= 1:
        _R_cross    = np.stack(_other_covs).mean(axis=0)
        _Rinv_cross = _matrix_sqrt_inv(_R_cross)
        for _gi in np.where(te_m)[0]:
            _w = _gidx_to_eeg_win.get(_gi)
            if _w is not None:
                allF_raw[_gi, RIEM_START:RIEM_START + N_FEAT_RIEM] = \
                    extract_riemannian_features(
                        np.array(_Rinv_cross @ _w.astype(np.float64), dtype=np.float32))

    # Fresh VT + z-score on training subjects only
    from sklearn.feature_selection import VarianceThreshold as _VT
    vt_l = _VT(threshold=0.001)
    vt_l.fit(allF_raw[tr_m])
    trF_l = vt_l.transform(allF_raw[tr_m])
    teF_l = vt_l.transform(allF_raw[te_m])
    mu_l  = trF_l.mean(axis=0)
    sd_l  = np.where(trF_l.std(axis=0) < 1e-8, 1.0, trF_l.std(axis=0))
    trF_l = np.clip((trF_l - mu_l) / sd_l, -10, 10)
    teF_l = np.clip((teF_l - mu_l) / sd_l, -10, 10)

    # Unsupervised test-time re-centring (no labels needed)
    teF_l = subject_recentre(trF_l, teF_l)

    trY_l = allY[tr_m]; teY_l = allY[te_m]; teTK_l = allTK[te_m]

    # MI feature selection -- computed on LOSO training subjects only (no leakage)
    sub_l  = (np.random.RandomState(42).choice(len(trF_l), 2000, replace=False)
              if len(trF_l) > 2000 else np.arange(len(trF_l)))
    mi_l   = mutual_info_classif(trF_l[sub_l], trY_l[sub_l], random_state=42, n_neighbors=5)

    # FIX leakage: re-tune K and shrinkage using only LOSO training subjects.
    # Use a quick 3-fold CV on training subjects to avoid using FINAL_K/FINAL_SH
    # which were tuned on data that included the current test subject.
    _loso_sids_tr = sorted(set(allSID[tr_m]))
    _loso_K_grid  = [60, 100, 160]
    _loso_sh_grid = [0.1, 0.3, 0.5]
    _best_loso_val, _loso_K, _loso_sh = -1.0, 100, 0.3
    if len(_loso_sids_tr) >= 3:
        for _lk in _loso_K_grid:
            for _lsh in _loso_sh_grid:
                _lfi = np.argsort(-mi_l)[:_lk]
                _lscores = []
                for _vsid in _loso_sids_tr[:3]:   # quick 3-subject held-out
                    _vm = np.array([allSID[i] == _vsid for i in np.where(tr_m)[0]])
                    _tm = ~_vm
                    if _vm.sum() == 0 or _tm.sum() == 0:
                        continue
                    try:
                        _clf = make_lda(_lsh)
                        _clf.fit(trF_l[_tm][:, _lfi], trY_l[_tm])
                        _lscores.append(float((_clf.predict(trF_l[_vm][:, _lfi]) == trY_l[_vm]).mean()))
                    except Exception:
                        pass
                if _lscores and float(np.mean(_lscores)) > _best_loso_val:
                    _best_loso_val = float(np.mean(_lscores))
                    _loso_K, _loso_sh = _lk, _lsh

    fi_l   = np.argsort(-mi_l)[:_loso_K]
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

    # Stacking ensemble with LDA fallback
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
    # per-subject class breakdown
    _sub_rows = [r for r in loso_win_rows if r["subject"] == loso_sid]
    if _sub_rows:
        _yt = [r["true_idx"] for r in _sub_rows]
        _yp = [r["pred_idx"] for r in _sub_rows]
        print(classification_report(_yt, _yp,
              target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
              zero_division=0, digits=2))
        print(confusion_matrix(_yt, _yp, labels=list(range(NUM_CLASSES))))
        print()

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
