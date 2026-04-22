# ================================================================
# BiMamba-MSMDA for Emognition -- LOSO Cross-Subject Evaluation
# ================================================================
# Architecture
# ─────────────────────────────────────────────────────────────
#  Raw EEG (4,T)
#    → clip artefacts (±5σ)
#    → z-score baseline normalisation (per-subject resting baseline)
#    → Euclidean Alignment (R from OTHER subjects -- zero leakage)
#    → bandpass into 5 bands, stack → (20, T)
#    → sliding windows → (20, W)
#                              │
#              ┌───────────────▼──────────────────┐
#              │  BiMambaCFE  (shared)             │
#              │  patch_embed  (20,W)→(B,L,d)      │
#              │  n_layers × BiMambaBlock           │
#              │  mean pool   → (B, d_model)        │
#              └──────────────┬───────────────────┘
#                 optional BVP fusion (+8 HRV feats)
#                             │
#              ┌──────────────▼─────────────────────┐
#              │  DSFE[j]  64→32  (one per source)  │
#              │  CLS[j]   32→4   (one per source)  │
#              └────────────────────────────────────┘
#
# MS-MDA losses (Chen et al. 2021):
#   cls_loss  = CE on labelled source windows
#   mmd_loss  = linear MMD(src_DSFE_j, tgt_DSFE_j)
#   disc_loss = |softmax(tgt_j) − softmax(tgt_k)| for k≠j
#   total     = cls + γ·mmd + β·disc
#   γ,β ramp up via Ganin schedule
#
# Leakage prevention
# ─────────────────────────────────────────────────────────────
#   1. EA R built from OTHER subjects only (never test subject)
#   2. BVP normalisation stats from training subjects only
#   3. No augmentation (avoids cross-subject MixUp blending)
#   4. Target (test) windows fed UNLABELLED -- labels never touch loss
#   5. No global hyperparameter search that sees test subjects
#
# References
# ─────────────────────────────────────────────────────────────
#   Chen et al. MS-MDA, Frontiers in Neuroscience 2021
#   Gu & Dao.   Mamba, arXiv 2312.00752 (2023)
#   He & Wu.    Euclidean Alignment, IEEE TNSRE 2019
# ================================================================

import os, json, glob, math, time, random, warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal  import welch, butter, filtfilt, resample as sp_resample
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

warnings.filterwarnings("ignore")

# ================================================================
# CONFIG  -- edit for your environment
# ================================================================
EMOGNITION_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
OUT_DIR         = "/kaggle/working"

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

# ── Model ──────────────────────────────────────────────────────
D_MODEL     = 64     # BiMambaCFE output dim (= DSFE input dim)
D_STATE     = 16     # Mamba SSM state size
N_LAYERS    = 2      # number of BiMamba layers
PATCH_SIZE  = 256    # 1-second patches at 256 Hz → 10 patches per window
DROPOUT     = 0.35   # applied inside BiMamba blocks
USE_BVP     = True   # fuse 8-dim HRV features from Samsung Watch
BVP_DIM     = 8      # HR_mean, RMSSD, pNN50, IBI_range, SDNN, mean_IBI, LF, HF

# ── Training ───────────────────────────────────────────────────
EPOCHS      = 80
BATCH_SIZE  = 32
LR          = 5e-4
WEIGHT_DECAY= 1e-4
WARMUP_EP   = 5
PATIENCE    = 20     # early stopping on val clip-F1
GAMMA_MAX   = 1.0    # max MMD loss weight
BETA_DIV    = 100.0  # disc_loss weight = gamma / BETA_DIV

# ── Signal ─────────────────────────────────────────────────────
FS          = 256
WINDOW_SEC  = 10
STEP_TR_SEC = 5      # 50 % overlap during training
STEP_EV_SEC = 10     # non-overlapping during evaluation
IN_CHANNELS = 20     # 5 bands × 4 EEG channels

NUM_CLASSES = 4
CLASS_NAMES = ["ENTHUSIASM", "FEAR", "NEUTRAL", "SADNESS"]
EMOTION_MAP = {"ENTHUSIASM": 0, "FEAR": 1, "NEUTRAL": 2, "SADNESS": 3}
IDX_TO_EMO  = {v: k for k, v in EMOTION_MAP.items()}
EEG_CHANNELS= ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]

# Band definitions: (name, lo_hz, hi_hz)
BANDS = [("delta",1,4), ("theta",4,8), ("alpha",8,13), ("beta",13,30), ("gamma",30,45)]

# L-R electrode flip index for augmentation (5 bands × 4 ch = 20 ch)
# MUSE order per band: [TP9(L), AF7(L), AF8(R), TP10(R)]
# Flip: [TP10, AF8, AF7, TP9] = indices [3,2,1,0] per band
_LR_FLIP = np.array([3,2,1,0, 7,6,5,4, 11,10,9,8, 15,14,13,12, 19,18,17,16], dtype=np.intp)


# ================================================================
# REPRODUCIBILITY
# ================================================================
def set_seed(seed: int = SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ================================================================
# PURE-PYTORCH MAMBA / BiMAMBA  (no C extensions)
# ================================================================

class MambaSSM(nn.Module):
    """
    Core Mamba selective SSM block -- pure PyTorch, no mamba_ssm needed.
    Implements the selective scan from Gu & Dao (2023).
    Efficient for short sequences (L ≤ 40); uses sequential scan.

    Input / output shape: (B, L, d_model)
    """
    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_state = d_state
        d_inner      = int(d_model * expand)
        self.d_inner  = d_inner

        # Input projection: splits into main branch + gate
        self.in_proj  = nn.Linear(d_model, d_inner * 2, bias=False)

        # Causal depthwise conv along sequence axis (captures local context)
        self.conv1d   = nn.Conv1d(d_inner, d_inner, kernel_size=d_conv,
                                   padding=d_conv - 1, groups=d_inner, bias=True)

        # SSM input projections: B_coef (N), C_coef (N), dt (1)
        self.x_proj   = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        # dt projection: scalar → d_inner
        self.dt_proj  = nn.Linear(1, d_inner, bias=True)
        nn.init.constant_(self.dt_proj.bias, math.log(math.expm1(1.0)))  # dt init

        # A matrix in log-space (stable parameterisation)
        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).repeat(d_inner, 1)          # (d_inner, N)
        self.A_log    = nn.Parameter(torch.log(A))

        # Skip-connection scale D
        self.D        = nn.Parameter(torch.ones(d_inner))

        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)  →  (B, L, d_model)  [no residual here]"""
        B, L, _ = x.shape

        # Project and split into main path (x_in) and gate (z)
        xz     = self.in_proj(x)                               # (B, L, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                          # each (B, L, d_inner)

        # Causal depthwise conv  →  (B, L, d_inner)
        x_conv = self.conv1d(x_in.transpose(1, 2))             # (B, d_inner, L+pad)
        x_conv = x_conv[:, :, :L].transpose(1, 2)              # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # Compute time-varying SSM parameters
        ssm_in            = self.x_proj(x_conv)                # (B, L, 2N+1)
        B_c, C_c, dt_raw  = ssm_in.split([self.d_state,
                                           self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))                  # (B, L, d_inner)
        A  = -torch.exp(self.A_log.float())                    # (d_inner, N)

        # Selective scan (sequential -- fast for L ≤ 40 patches)
        y = self._scan(x_conv, dt, A, B_c, C_c)               # (B, L, d_inner)

        # Skip connection + SiLU gate
        y = y + x_conv * self.D
        y = y * F.silu(z)

        return self.drop(self.out_proj(y))                     # (B, L, d_model)

    def _scan(self, x: torch.Tensor, dt: torch.Tensor,
              A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
              ) -> torch.Tensor:
        """Selective state-space scan.  O(L·d·N) time, O(d·N) memory."""
        Bsz, L, d = x.shape
        N = self.d_state
        h = torch.zeros(Bsz, d, N, device=x.device, dtype=x.dtype)
        ys = []
        for i in range(L):
            dA = torch.exp(dt[:, i].unsqueeze(-1) * A.unsqueeze(0)) # (B, d, N)
            dB = dt[:, i].unsqueeze(-1) * B[:, i].unsqueeze(1)      # (B, d, N)
            h  = h * dA + dB * x[:, i].unsqueeze(-1)                # (B, d, N)
            yi = (h * C[:, i].unsqueeze(1)).sum(-1)                  # (B, d)
            ys.append(yi)
        return torch.stack(ys, dim=1)                                # (B, L, d)


class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block.
    Runs forward SSM and backward SSM in parallel, merges with a linear layer.
    Pre-norm architecture; residual connection wraps the whole block.

    Input / output: (B, L, d_model)
    """
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm     = nn.LayerNorm(d_model)
        self.fwd_ssm  = MambaSSM(d_model, d_state, dropout=0.0)
        self.bwd_ssm  = MambaSSM(d_model, d_state, dropout=0.0)
        # Merge forward + backward features
        self.merge    = nn.Linear(d_model * 2, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_n  = self.norm(x)                              # pre-norm
        fwd  = self.fwd_ssm(x_n)                        # forward scan
        bwd  = self.bwd_ssm(x_n.flip(1)).flip(1)        # backward scan (flip seq)
        out  = self.merge(torch.cat([fwd, bwd], dim=-1)) # (B, L, d_model)
        return residual + self.drop(out)                 # residual


class BiMambaCFE(nn.Module):
    """
    Bidirectional Mamba Common Feature Extractor.
    Replaces the MLP-based CFE in vanilla MS-MDA with a temporal encoder.

    Pipeline:
      (B, 20, T)
        → Conv1d patch embed  → (B, d_model, n_patches)
        → transpose           → (B, n_patches, d_model)
        → n_layers BiMamba    → (B, n_patches, d_model)
        → LayerNorm + mean    → (B, d_model)

    With WINDOW_SEC=10, FS=256, PATCH_SIZE=256:
      T = 2560,  n_patches = 10 (one patch = 1 second of EEG)
    """
    def __init__(self, in_channels: int = IN_CHANNELS,
                 d_model: int = D_MODEL,
                 n_layers: int = N_LAYERS,
                 d_state:  int = D_STATE,
                 patch_size: int = PATCH_SIZE,
                 dropout: float = DROPOUT):
        super().__init__()
        self.d_model = d_model

        # Patch embedding: each 1-second segment → d_model feature
        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_channels, d_model,
                      kernel_size=patch_size, stride=patch_size, bias=False),
            nn.GELU(),
        )

        # Stacked bidirectional Mamba layers
        self.layers   = nn.ModuleList([
            BiMambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 20, T)  →  (B, d_model)"""
        x = self.patch_embed(x)       # (B, d_model, n_patches)
        x = x.transpose(1, 2)         # (B, n_patches, d_model)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        return x.mean(dim=1)          # (B, d_model)  global average pool


# ================================================================
# MS-MDA COMPONENTS
# ================================================================

class DSFE(nn.Module):
    """Domain-Specific Feature Extractor -- one per source subject."""
    def __init__(self, d_in: int = D_MODEL):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 32),
            nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.net(x)


def mmd_linear(fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """
    Linear (empirical) MMD between two batches of EQUAL size.
    delta = fx - fy;  loss = mean(delta @ delta.T)
    Reference: original MS-MDA implementation (Chen et al. 2021).
    """
    delta = fx - fy
    return torch.mean(torch.mm(delta, delta.t()))


class BiMambaMSMDA(nn.Module):
    """
    BiMamba-MSMDA: Bidirectional Mamba + Multi-Source Marginal Distribution
    Adaptation for cross-subject EEG emotion recognition.

    BiMambaCFE   (shared temporal encoder)  ← replaces MLP CFE
    DSFE[j]      (domain-specific extractor, one per source subject)
    CLS[j]       (classifier head,           one per source subject)
    Optional BVP (8-dim HRV features fused after CFE via linear projection)
    """
    def __init__(self, in_channels: int, d_model: int, n_layers: int,
                 d_state: int, patch_size: int, num_classes: int,
                 n_sources: int, dropout: float = 0.3, bvp_dim: int = 0):
        super().__init__()
        self.n_sources = n_sources
        self.bvp_dim   = bvp_dim

        # Shared BiMamba temporal encoder
        self.cfe = BiMambaCFE(in_channels, d_model, n_layers,
                               d_state, patch_size, dropout)

        # Optional BVP fusion: project (d_model + bvp_dim) → d_model
        if bvp_dim > 0:
            self.bvp_proj = nn.Sequential(
                nn.Linear(d_model + bvp_dim, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
            )

        # Per-source domain heads
        self.dsfe_list = nn.ModuleList([DSFE(d_model) for _ in range(n_sources)])
        self.cls_list  = nn.ModuleList([nn.Linear(32, num_classes)
                                        for _ in range(n_sources)])

    # ── internal helper ──────────────────────────────────────────
    def _encode(self, x_eeg: torch.Tensor,
                x_bvp: torch.Tensor | None = None) -> torch.Tensor:
        emb = self.cfe(x_eeg)                              # (B, d_model)
        if self.bvp_dim > 0 and x_bvp is not None:
            emb = self.bvp_proj(torch.cat([emb, x_bvp], dim=-1))
        return emb

    # ── training forward ─────────────────────────────────────────
    def forward(self, src_eeg: torch.Tensor, src_label: torch.Tensor,
                tgt_eeg: torch.Tensor, mark: int,
                src_bvp: torch.Tensor | None = None,
                tgt_bvp: torch.Tensor | None = None):
        """
        Returns (cls_loss, mmd_loss, disc_loss).
        target labels are NEVER used -- fully unsupervised DA.
        """
        src_cfe = self._encode(src_eeg, src_bvp)           # (B, d_model)
        tgt_cfe = self._encode(tgt_eeg, tgt_bvp)           # (B, d_model)

        # Extract target features through ALL domain-specific heads
        tgt_dsfe = [self.dsfe_list[i](tgt_cfe) for i in range(self.n_sources)]

        # Extract source features through its own domain-specific head
        src_dsfe = self.dsfe_list[mark](src_cfe)

        # ① MMD: align source domain `mark` with target
        mmd_loss = mmd_linear(src_dsfe, tgt_dsfe[mark])

        # ② Discrepancy: keep target representations consistent across heads
        disc_loss = sum(
            torch.mean(torch.abs(
                F.softmax(self.cls_list[mark](tgt_dsfe[mark]), dim=1) -
                F.softmax(self.cls_list[i](tgt_dsfe[i]),       dim=1)
            ))
            for i in range(self.n_sources) if i != mark
        )
        if self.n_sources <= 1:
            disc_loss = torch.zeros(1, device=src_eeg.device).squeeze()

        # ③ Classification on labelled source
        pred     = self.cls_list[mark](src_dsfe)
        cls_loss = F.cross_entropy(pred, src_label.long())

        return cls_loss, mmd_loss, disc_loss

    # ── inference ────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, x_eeg: torch.Tensor,
                x_bvp: torch.Tensor | None = None) -> torch.Tensor:
        """Average softmax over all source heads.  Returns (B, num_classes)."""
        self.eval()
        emb   = self._encode(x_eeg, x_bvp)
        preds = [F.softmax(self.cls_list[i](self.dsfe_list[i](emb)), dim=1)
                 for i in range(self.n_sources)]
        return torch.stack(preds).mean(dim=0)


# ================================================================
# TRAINING UTILITIES
# ================================================================

class LabelSmoothingCE(nn.Module):
    def __init__(self, n_classes: int, smoothing: float = 0.15):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_p = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            soft = torch.full_like(log_p, self.smoothing / (self.n_classes - 1))
            soft.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(soft * log_p).sum(dim=-1).mean()


class WarmupCosine:
    """Linear warmup followed by cosine decay."""
    def __init__(self, opt, warmup: int, total: int, min_lr: float = 1e-7):
        self.opt      = opt
        self.warmup   = warmup
        self.total    = total
        self.min_lr   = min_lr
        self.base_lrs = [pg["lr"] for pg in opt.param_groups]
        self._ep      = 0

    def step(self):
        self._ep += 1
        e = self._ep
        for pg, base in zip(self.opt.param_groups, self.base_lrs):
            if e <= self.warmup:
                pg["lr"] = base * e / max(self.warmup, 1)
            else:
                prog    = (e - self.warmup) / max(self.total - self.warmup, 1)
                pg["lr"]= self.min_lr + (base - self.min_lr) * 0.5 * (
                          1.0 + math.cos(math.pi * prog))

    def get_lr(self): return [pg["lr"] for pg in self.opt.param_groups]


# ================================================================
# SIGNAL PRE-PROCESSING  (from biMamba.py)
# ================================================================

def clip_artefacts(trial: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
    """Clip per-channel spikes to ±n_sigma × channel std.  Input: (C,T)."""
    trial = trial.astype(np.float64, copy=True)
    for c in range(trial.shape[0]):
        σ = trial[c].std()
        if σ > 1e-8:
            trial[c] = np.clip(trial[c], -n_sigma * σ, n_sigma * σ)
    return trial.astype(np.float32)


def apply_zscore_baseline(trial: np.ndarray,
                           baseline: dict | None) -> np.ndarray:
    """
    Z-score normalise with per-channel μ/σ from subject's resting baseline.
    Falls back to within-trial stats if baseline is None.
    Input/output: (4, T) float32.
    """
    trial = trial.astype(np.float32)
    if baseline is not None:
        μ = baseline["μ"][:, None]
        σ = baseline["σ"][:, None] + 1e-8
    else:
        μ = trial.mean(axis=1, keepdims=True)
        σ = trial.std(axis=1,  keepdims=True) + 1e-8
    return ((trial - μ) / σ).astype(np.float32)


def _butter_bp(lo: float, hi: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [np.clip(lo/nyq, 1e-6, 1-1e-6),
                           np.clip(hi/nyq, 1e-6, 1-1e-6)], btype="band")


def apply_band_stack(trial: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    Bandpass into 5 bands and stack along channel axis.
    Input:  (4, T)   Output: (20, T) float32
    Order: [delta_ch0..3, theta_ch0..3, alpha_ch0..3, beta_ch0..3, gamma_ch0..3]
    """
    out = []
    for _, lo, hi in BANDS:
        b, a = _butter_bp(lo, hi, fs)
        out.append(filtfilt(b, a, trial, axis=1).astype(np.float32))
    return np.concatenate(out, axis=0)   # (20, T)


# ================================================================
# EUCLIDEAN ALIGNMENT HELPERS
# ================================================================

def _reg_cov(X: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    C, T = X.shape
    Xc   = X - X.mean(axis=1, keepdims=True)
    cov  = (Xc @ Xc.T) / max(T - 1, 1)
    return (1 - reg) * cov + reg * (np.trace(cov) / C) * np.eye(C)


def _sqrt_inv(M: np.ndarray) -> np.ndarray:
    v, U = np.linalg.eigh(M)
    return U @ np.diag(1.0 / np.sqrt(np.maximum(v, 1e-10))) @ U.T


# ================================================================
# BVP / SAMSUNG WATCH FEATURE LOADING  (from biMamba.py)
# ================================================================

def _parse_paired(raw):
    if not isinstance(raw, list) or len(raw) < 5: return None
    try: return np.array([r[1] for r in raw], dtype=np.float64)
    except: return None


def _lf_hf(ibi: np.ndarray, fs_ibi: float = 4.0):
    try:
        if len(ibi) < 10: return 0.0, 0.0
        n    = max(int(len(ibi) / fs_ibi * fs_ibi), 8)
        iu   = sp_resample(ibi, n)
        f, p = welch(iu, fs=fs_ibi, nperseg=min(64, n))
        lf   = float(np.trapz(p[(f>=0.04)&(f<0.15)] + 1e-12, f[(f>=0.04)&(f<0.15)]))
        hf   = float(np.trapz(p[(f>=0.15)&(f<0.40)] + 1e-12, f[(f>=0.15)&(f<0.40)]))
        return max(lf, 0.0), max(hf, 0.0)
    except: return 0.0, 0.0


def load_bvp_one(fp: str) -> np.ndarray | None:
    """
    Extract 8 HRV features from one Samsung Watch STIMULUS JSON.
    Returns float32 (8,) or None.
    Features: [HR_mean, RMSSD, pNN50, IBI_range, SDNN, mean_IBI, LF, HF]
    """
    try:
        with open(fp) as f: obj = json.load(f)
    except: return None

    ibi = _parse_paired(obj.get("PPInterval"))
    hr  = _parse_paired(obj.get("heartRate"))
    if ibi is not None: ibi = ibi[(ibi>300)&(ibi<2000)&np.isfinite(ibi)]
    if hr  is not None: hr  = hr [(hr>30)  &(hr<220)  &np.isfinite(hr)]
    if ibi is None or len(ibi) < 5: return None

    d      = np.diff(ibi)
    hr_m   = float(np.mean(hr)) if (hr is not None and len(hr)>=3) \
             else float(np.mean(60000.0/ibi))
    lf, hf = _lf_hf(ibi)
    feat   = np.array([hr_m,
                       float(np.sqrt(np.mean(d**2))),
                       float(np.mean(np.abs(d)>50)),
                       float(ibi.max()-ibi.min()),
                       float(ibi.std()),
                       float(ibi.mean()),
                       lf, hf], dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


def build_bvp_lookup(root: str) -> dict:
    """Return dict: (subj_str, EMOTION_STR) → float32(8,)."""
    pats = [os.path.join(root, "*_STIMULUS_SAMSUNG_WATCH.json"),
            os.path.join(root, "*", "*_STIMULUS_SAMSUNG_WATCH.json"),
            os.path.join(root, "**","*_STIMULUS_SAMSUNG_WATCH.json")]
    files = sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})
    lookup = {}
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0].split("_")
        if len(name) < 2: continue
        subj, emo = name[0], name[1].upper()
        if emo not in EMOTION_MAP: continue
        feat = load_bvp_one(fp)
        if feat is not None:
            lookup[(subj, emo)] = feat
    print(f"  BVP: {len(lookup)} clips loaded "
          f"({len(set(s for s,_ in lookup))} subjects)")
    return lookup


# ================================================================
# DATA LOADING
# ================================================================

def load_baselines(root: str) -> dict:
    """
    Load per-subject z-score stats {μ, σ} from BASELINE_MUSE_cleaned.json.
    Returns dict: subj_str → {'μ': float32(4,), 'σ': float32(4,)}
    """
    pats = [os.path.join(root,"*","*_BASELINE_MUSE_cleaned.json"),
            os.path.join(root,"**","*_BASELINE_MUSE_cleaned.json")]
    files = sorted({p for pat in pats for p in glob.glob(pat, recursive=True)})

    # also try the cleaned-combined layout used in MaxEffort
    if not files:
        for subj in sorted(os.listdir(root)):
            d = os.path.join(root, subj)
            if not os.path.isdir(d) or not subj.isdigit(): continue
            files += sorted(glob.glob(os.path.join(d,"*_BASELINE_MUSE_cleaned.json")))

    stats = {}
    for fp in files:
        sid = os.path.basename(fp).split("_")[0]
        try:
            with open(fp) as f: obj = json.load(f)
            raw_ch = []
            for ch in EEG_CHANNELS:
                arr = np.asarray(obj.get(ch, []), dtype=np.float64)
                if len(arr) == 0: raw_ch = []; break
                raw_ch.append(arr)
            if len(raw_ch) != 4: continue
            L   = min(len(a) for a in raw_ch)
            sig = np.nan_to_num(np.stack([a[:L] for a in raw_ch]))  # (4,T)
            mu  = sig.mean(axis=1).astype(np.float32)
            sd  = sig.std(axis=1).astype(np.float32)
            sd  = np.where(sd < 1e-6, 1.0, sd)
            stats[sid] = {"μ": mu, "σ": sd}
        except: pass
    print(f"  Baselines loaded for {len(stats)} subjects")
    return stats


def load_all_trials(root: str) -> list:
    """
    Load all stimulus trials. Returns list of dicts:
      sid, emotion, label, raw_eeg (4,T), tkey
    raw_eeg is artefact-clipped and z-score normalised.
    """
    baselines = load_baselines(root)
    if baselines:
        gbl_mu = np.mean([v["μ"] for v in baselines.values()], axis=0)
        gbl_sd = np.mean([v["σ"] for v in baselines.values()], axis=0)
        gbl    = {"μ": gbl_mu, "σ": gbl_sd}
    else:
        gbl = None

    trials = []
    for subj in sorted(os.listdir(root)):
        d = os.path.join(root, subj)
        if not os.path.isdir(d) or not subj.isdigit(): continue
        bl = baselines.get(subj, gbl)

        for ef in sorted(glob.glob(os.path.join(d,"*_STIMULUS_MUSE_cleaned.json"))):
            parts = os.path.basename(ef).split("_")
            emo   = parts[1].upper() if len(parts) >= 2 else None
            if emo not in EMOTION_MAP: continue
            try:
                with open(ef) as f: obj = json.load(f)
                eeg = np.stack([np.array(obj[ch], dtype=np.float32)
                                for ch in EEG_CHANNELS])           # (4,T)
            except: continue
            eeg = np.nan_to_num(eeg)
            eeg = clip_artefacts(eeg)
            eeg = apply_zscore_baseline(eeg, bl)
            trials.append(dict(sid=subj, emotion=emo,
                               label=EMOTION_MAP[emo],
                               raw_eeg=eeg,
                               tkey=f"{subj}_{emo}"))

    print(f"  Trials loaded: {len(trials)} "
          f"({len(set(t['sid'] for t in trials))} subjects)")
    return trials


def window_trial(eeg: np.ndarray, win_samples: int,
                 step_samples: int) -> list:
    """Slice (4,T) into list of (4, win_samples) windows."""
    T   = eeg.shape[1]
    wins = []
    for s in range(0, max(T - win_samples + 1, 1), step_samples):
        w = eeg[:, s:s + win_samples]
        if w.shape[1] < win_samples:
            w = np.pad(w, ((0,0),(0, win_samples - w.shape[1])))
        wins.append(w.astype(np.float32))
    return wins


# ================================================================
# AUGMENTATION (training only, label-preserving)
# ================================================================

def augment_window(x: np.ndarray) -> np.ndarray:
    """
    Augment a (20, W) band-stacked window.
    ① Gaussian noise  ② amplitude scale  ③ L-R flip  ④ band dropout  ⑤ time mask
    """
    σ = x.std()
    if σ > 1e-8:
        x = x + np.random.randn(*x.shape).astype(np.float32) * σ * 0.05
    x = x * np.random.uniform(0.85, 1.15)
    if np.random.random() < 0.5:
        x = x[_LR_FLIP]
    if np.random.random() < 0.15:
        b = np.random.randint(0, 5)
        x[b*4:(b+1)*4, :] = 0.0
    if np.random.random() < 0.40:
        T  = x.shape[1]
        ml = max(1, int(T * 0.10))
        s  = np.random.randint(0, max(T - ml, 1) + 1)
        x[:, s:s+ml] = 0.0
    return x


# ================================================================
# DATASET
# ================================================================

class EEGDataset(torch.utils.data.Dataset):
    def __init__(self, band_wins, labels, bvp_feats=None,
                 clip_ids=None, augment=False):
        self.wins    = band_wins          # list of (20, W) np arrays
        self.labels  = labels
        self.bvp     = (torch.tensor(np.array(bvp_feats), dtype=torch.float32)
                        if bvp_feats is not None else None)
        self.cids    = clip_ids
        self.augment = augment

    def __len__(self): return len(self.wins)

    def __getitem__(self, idx):
        x   = self.wins[idx].copy()
        if self.augment: x = augment_window(x)
        xt  = torch.from_numpy(x)
        lbl = self.labels[idx]
        cid = self.cids[idx] if self.cids is not None else -1
        if self.bvp is not None:
            return xt, self.bvp[idx], lbl, cid
        return xt, lbl, cid


# ================================================================
# EVALUATION
# ================================================================

@torch.no_grad()
def evaluate(model: BiMambaMSMDA, loader: DataLoader,
             device: str, use_bvp: bool = False):
    """
    Returns (win_acc, win_f1, clip_acc, clip_f1,
             win_preds, win_labels, clip_preds, clip_labels, clip_pred_map).
    """
    model.eval()
    all_probs, all_labels, all_cids = [], [], []

    for batch in loader:
        if use_bvp and len(batch) == 4:
            bx, bb, by, cid = batch
            bb = bb.to(device)
        else:
            bx, by, cid = batch[0], batch[-2], batch[-1]
            bb = None
        bx = bx.to(device)
        p  = model.predict(bx, bb).cpu().numpy()
        all_probs.extend(p)
        all_labels.extend(by.numpy() if isinstance(by, torch.Tensor)
                          else by)
        all_cids.extend(cid.numpy() if isinstance(cid, torch.Tensor)
                        else cid)

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    cids   = np.array(all_cids)
    preds  = probs.argmax(axis=1)

    win_acc = float(np.mean(preds == labels))
    win_f1  = f1_score(labels, preds, average="macro", zero_division=0)

    # Clip-level: uniform average of softmax probs per clip
    clip_preds, clip_true, clip_pred_map = [], [], {}
    for cid in np.unique(cids):
        m = cids == cid
        pred_cls = int(probs[m].mean(axis=0).argmax())
        clip_preds.append(pred_cls)
        clip_true.append(int(labels[m][0]))
        clip_pred_map[int(cid)] = pred_cls

    clip_acc = float(np.mean(np.array(clip_preds) == np.array(clip_true)))
    clip_f1  = f1_score(clip_true, clip_preds, average="macro", zero_division=0)

    return (win_acc, win_f1, clip_acc, clip_f1,
            preds.tolist(), labels.tolist(),
            clip_preds, clip_true, clip_pred_map)


# ================================================================
# LOSO TRAINING LOOP
# ================================================================

def train_one_fold(model: BiMambaMSMDA,
                   src_datasets: list,          # one TensorDataset per source subject
                   tgt_loader_train: DataLoader, # unlabelled target windows
                   optimizer, epochs: int,
                   device: str,
                   use_bvp: bool = False):
    """
    Train MS-MDA for one LOSO fold.
    Target labels are NEVER used -- domain adaptation is fully unsupervised.
    """
    n_src = len(src_datasets)
    eff_bs = min(BATCH_SIZE, min(len(ds) for ds in src_datasets))
    eff_bs = max(eff_bs, 2)

    src_loaders = [DataLoader(ds, sampler=RandomSampler(ds),
                               batch_size=eff_bs, drop_last=True)
                   for ds in src_datasets]
    src_iters   = [iter(l) for l in src_loaders]
    tgt_iter    = iter(tgt_loader_train)

    n_per_ep    = max(1, min(len(l) for l in src_loaders))
    total_iters = epochs * n_per_ep

    model.train()
    for epoch in range(epochs):
        for step_i in range(n_per_ep):
            for j in range(n_src):
                # ── source batch ─────────────────────────────────
                try: src_batch = next(src_iters[j])
                except StopIteration:
                    src_iters[j] = iter(src_loaders[j])
                    src_batch    = next(src_iters[j])

                # ── target batch (labels ignored) ─────────────────
                try: tgt_batch = next(tgt_iter)
                except StopIteration:
                    tgt_iter  = iter(tgt_loader_train)
                    tgt_batch = next(tgt_iter)

                # Unpack EEG (and optional BVP)
                if use_bvp and len(src_batch) == 4:
                    sx, sb, sy, _ = src_batch
                    sb = sb.to(device)
                else:
                    sx, sy, _ = src_batch[0], src_batch[-2], src_batch[-1]
                    sb = None

                if use_bvp and len(tgt_batch) == 4:
                    tx, tb, _, _ = tgt_batch
                    tb = tb.to(device)
                else:
                    tx = tgt_batch[0]
                    tb = None

                # Ensure same batch size for MMD (drop extras)
                bmin = min(sx.shape[0], tx.shape[0])
                if bmin < 2:
                    continue   # BatchNorm1d requires >= 2 samples
                sx, sy, tx = sx[:bmin].to(device), sy[:bmin].to(device), tx[:bmin].to(device)
                if sb is not None: sb = sb[:bmin]
                if tb is not None: tb = tb[:bmin]

                optimizer.zero_grad()
                cls_l, mmd_l, disc_l = model(sx, sy, tx, mark=j,
                                              src_bvp=sb, tgt_bvp=tb)

                # Progressive loss weight ramp-up (Ganin schedule)
                g_step = epoch * n_per_ep + step_i
                gamma  = GAMMA_MAX * (2 / (1 + math.exp(-10 * g_step / total_iters)) - 1)
                beta   = gamma / BETA_DIV
                loss   = cls_l + gamma * mmd_l + beta * disc_l

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()


# ================================================================
# MAIN
# ================================================================

def main():
    set_seed()
    print(f"\n{'='*65}")
    print("  BiMamba-MSMDA  |  Emognition LOSO")
    print(f"{'='*65}")
    print(f"  Device : {DEVICE}")
    print(f"  BVP    : {'ON (8 HRV feats)' if USE_BVP else 'OFF'}")
    print(f"  Model  : d_model={D_MODEL}, n_layers={N_LAYERS}, "
          f"d_state={D_STATE}, patch={PATCH_SIZE}")
    print(f"  Train  : epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}, "
          f"patience={PATIENCE}")
    print(f"{'='*65}\n")

    WIN_SAMPLES  = WINDOW_SEC * FS          # 2560
    STEP_TR      = STEP_TR_SEC * FS         # 1280  (50% overlap)
    STEP_EV      = STEP_EV_SEC * FS         # 2560  (no overlap)

    # ── 1. Load trials ────────────────────────────────────────────
    print("Step 1 — Loading trials ...")
    t0     = time.time()
    trials = load_all_trials(EMOGNITION_ROOT)
    print(f"  Done in {time.time()-t0:.1f}s\n")

    # ── 2. BVP lookup ─────────────────────────────────────────────
    bvp_lookup = {}
    if USE_BVP:
        print("Step 2 — Building BVP lookup ...")
        bvp_lookup = build_bvp_lookup(EMOGNITION_ROOT)
        print()

    # ── 3. Window raw normalised EEG (4, W), keep raw for EA ──────
    print("Step 3 — Windowing raw EEG ...")
    raw_wins, all_labels, all_tkeys, all_sids = [], [], [], []
    all_clip_ids, clip_id = [], 0
    raw_bvp_feats = []   # un-normalised BVP, normalised per fold

    for tr in trials:
        wins = window_trial(tr["raw_eeg"], WIN_SAMPLES, STEP_TR)
        bvp  = (bvp_lookup.get((tr["sid"], tr["emotion"]),
                                np.zeros(BVP_DIM, np.float32))
                if USE_BVP else None)
        for w in wins:
            raw_wins.append(w)
            all_labels.append(tr["label"])
            all_tkeys.append(tr["tkey"])
            all_sids.append(tr["sid"])
            all_clip_ids.append(clip_id)
            raw_bvp_feats.append(bvp if bvp is not None
                                  else np.zeros(BVP_DIM, np.float32))
        clip_id += 1

    all_labels   = np.array(all_labels)
    all_tkeys    = np.array(all_tkeys)
    all_sids     = np.array(all_sids)
    all_clip_ids = np.array(all_clip_ids)
    raw_bvp      = np.array(raw_bvp_feats, dtype=np.float32)
    # Map trial key → clip_id for STEP_EV re-windowing inside the LOSO loop
    tkey_to_clipid = {tk: int(cid) for tk, cid in zip(all_tkeys, all_clip_ids)}
    print(f"  Total windows: {len(all_labels)}  "
          f"Subjects: {len(set(all_sids))}\n")

    # ── 4. Pre-compute per-subject mean covariance (for cross-subject EA) ──
    print("Step 4 — Pre-computing per-subject covariances for EA ...")
    _sid_mean_cov = {}
    sid_to_gidxs  = defaultdict(list)
    for gi, sid in enumerate(all_sids):
        sid_to_gidxs[sid].append(gi)
    for sid, gidxs in sid_to_gidxs.items():
        covs = [_reg_cov(raw_wins[gi].astype(np.float64)) for gi in gidxs]
        _sid_mean_cov[sid] = np.stack(covs).mean(axis=0)
    print(f"  Covariances cached for {len(_sid_mean_cov)} subjects\n")

    # ── 5. LOSO loop ─────────────────────────────────────────────
    all_sids_unique = sorted(set(all_sids))
    loso_win_rows, loso_trial_rows, loso_accs = [], [], []

    for fi, loso_sid in enumerate(all_sids_unique):
        te_mask = (all_sids == loso_sid)
        tr_mask = (all_sids != loso_sid)
        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue

        print(f"\n{'='*65}")
        print(f"  FOLD {fi+1}/{len(all_sids_unique)}  |  "
              f"Test subject: {loso_sid}  "
              f"(tr={tr_mask.sum()}, te={te_mask.sum()} wins)")
        print(f"{'='*65}")
        set_seed(SEED + fi)

        # ── EA: cross-subject R for test subject ──────────────────
        # R = mean of ALL OTHER subjects' mean covariances
        # Never uses the test subject's own data → zero leakage
        other_covs  = [_sid_mean_cov[s] for s in _sid_mean_cov if s != loso_sid]
        R_cross     = np.stack(other_covs).mean(axis=0)
        Rinv_cross  = _sqrt_inv(R_cross)

        # ── EA: per-subject R for training subjects ────────────────
        # Each training subject uses only their own windows → no leakage
        def apply_ea_and_bandstack(gidxs, Rinv):
            """Apply Rinv to raw (4,W) windows, then band-stack → (20,W)."""
            out = []
            for gi in gidxs:
                aligned = Rinv @ raw_wins[gi].astype(np.float64)
                out.append(apply_band_stack(aligned.astype(np.float32)))
            return out  # list of (20, W) arrays

        # Build band-stacked windows for ALL subjects
        # Training subjects: use their own Rinv
        # Test subject:      use cross-subject Rinv
        band_all = [None] * len(all_labels)

        for sid in all_sids_unique:
            gidxs = sid_to_gidxs[sid]
            if sid == loso_sid:
                Rinv = Rinv_cross
            else:
                # per-subject EA with own covariance
                covs_s = [_reg_cov(raw_wins[gi].astype(np.float64))
                           for gi in gidxs]
                R_self = np.stack(covs_s).mean(axis=0)
                Rinv   = _sqrt_inv(R_self)
            bs_wins = apply_ea_and_bandstack(gidxs, Rinv)
            for gi, bw in zip(gidxs, bs_wins):
                band_all[gi] = bw

        # ── BVP normalisation: stats from training subjects ONLY ──
        # Using test-subject BVP in norm stats would be leakage
        tr_bvp_arr = raw_bvp[tr_mask]
        bvp_mu     = tr_bvp_arr.mean(axis=0).astype(np.float32)
        bvp_sd     = np.where(tr_bvp_arr.std(axis=0) < 1e-8,
                               1.0, tr_bvp_arr.std(axis=0)).astype(np.float32)
        norm_bvp   = (raw_bvp - bvp_mu) / bvp_sd   # (N_all, 8)

        # ── Split arrays ──────────────────────────────────────────
        tr_idx = np.where(tr_mask)[0]
        te_idx = np.where(te_mask)[0]

        def make_ds(idxs, augment=False):
            bw = [band_all[i] for i in idxs]
            lb = all_labels[idxs].tolist()
            bv = norm_bvp[idxs].tolist() if USE_BVP else None
            ci = all_clip_ids[idxs].tolist()
            return EEGDataset(bw, lb, bv, ci, augment=augment)

        # ── Per-source-subject datasets (one domain per subject) ──
        tr_sids = all_sids[tr_mask]
        src_datasets = []
        for ssid in sorted(set(tr_sids)):
            sub_idx = tr_idx[tr_sids == ssid]
            if len(sub_idx) == 0: continue
            bw = [band_all[i] for i in sub_idx]
            lb = torch.tensor(all_labels[sub_idx], dtype=torch.long)
            bv = torch.tensor(norm_bvp[sub_idx], dtype=torch.float32)
            ci = torch.tensor(all_clip_ids[sub_idx], dtype=torch.long)
            if USE_BVP:
                src_datasets.append(TensorDataset(
                    torch.tensor(np.stack(bw), dtype=torch.float32), bv, lb, ci))
            else:
                src_datasets.append(TensorDataset(
                    torch.tensor(np.stack(bw), dtype=torch.float32), lb, ci))

        n_sources = len(src_datasets)
        if n_sources == 0:
            print(f"  No sources -- skipping"); continue

        # ── Target dataset: re-windowed with STEP_EV (non-overlapping) ──
        # Test subject is windowed at STEP_EV stride (no overlap) for evaluation,
        # which avoids redundant/correlated predictions at clip-voting time.
        te_trials_list = [tr for tr in trials if tr["sid"] == loso_sid]
        te_ev_bw, te_ev_lb, te_ev_bv_raw, te_ev_ci, te_ev_tkeys = [], [], [], [], []
        for tr in te_trials_list:
            ev_wins = window_trial(tr["raw_eeg"], WIN_SAMPLES, STEP_EV)
            cid     = tkey_to_clipid[tr["tkey"]]
            bvp_raw = (bvp_lookup.get((tr["sid"], tr["emotion"]),
                                      np.zeros(BVP_DIM, np.float32))
                       if USE_BVP else np.zeros(BVP_DIM, np.float32))
            for w in ev_wins:
                aligned = Rinv_cross @ w.astype(np.float64)
                te_ev_bw.append(apply_band_stack(aligned.astype(np.float32)))
                te_ev_lb.append(tr["label"])
                te_ev_bv_raw.append(bvp_raw)
                te_ev_ci.append(cid)
                te_ev_tkeys.append(tr["tkey"])

        # Guard: skip fold if test subject produced no evaluation windows
        if len(te_ev_bw) == 0:
            print(f"  No STEP_EV windows for {loso_sid} -- skipping")
            continue

        # Normalise BVP with training-only stats (no leakage)
        te_ev_bv_norm = (np.array(te_ev_bv_raw, dtype=np.float32) - bvp_mu) / bvp_sd
        te_bw_t = torch.tensor(np.stack(te_ev_bw), dtype=torch.float32)
        # Pass TRUE labels into tgt_ds -- training loop discards them (see unpack
        # with _ below), so this is safe and makes evaluate() metrics meaningful.
        te_lb_t = torch.tensor(te_ev_lb, dtype=torch.long)
        te_ci_t = torch.tensor(te_ev_ci, dtype=torch.long)
        if USE_BVP:
            tgt_ds = TensorDataset(te_bw_t,
                                   torch.tensor(te_ev_bv_norm, dtype=torch.float32),
                                   te_lb_t, te_ci_t)
        else:
            tgt_ds = TensorDataset(te_bw_t, te_lb_t, te_ci_t)

        tgt_inf_loader = DataLoader(tgt_ds, sampler=SequentialSampler(tgt_ds),
                                     batch_size=BATCH_SIZE, drop_last=False)
        n_te_ev = len(te_ev_lb)   # STEP_EV window count for test subject
        tgt_trn_loader = DataLoader(tgt_ds, sampler=RandomSampler(tgt_ds),
                                     batch_size=min(max(BATCH_SIZE, n_te_ev//2+1),
                                                    n_te_ev),
                                     drop_last=True)

        # ── Validation: 15% of training subjects held out ─────────
        rem_sids = sorted(set(tr_sids))
        n_va     = max(1, int(0.15 * len(rem_sids)))
        va_sids  = set(np.random.RandomState(SEED + fi).choice(
                       rem_sids, n_va, replace=False))
        va_idx   = tr_idx[np.array([all_sids[i] in va_sids for i in tr_idx])]
        va_ds    = make_ds(va_idx, augment=False)
        va_loader= DataLoader(va_ds, batch_size=BATCH_SIZE,
                               shuffle=False, drop_last=False)

        # ── Build fresh model for this fold ──────────────────────
        try:
            del model
            torch.cuda.empty_cache()
        except NameError:
            pass
        model = BiMambaMSMDA(
            in_channels=IN_CHANNELS, d_model=D_MODEL, n_layers=N_LAYERS,
            d_state=D_STATE, patch_size=PATCH_SIZE,
            num_classes=NUM_CLASSES, n_sources=n_sources,
            dropout=DROPOUT, bvp_dim=BVP_DIM if USE_BVP else 0
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Model params: {n_params:,}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                       weight_decay=WEIGHT_DECAY, eps=1e-8)
        sched     = WarmupCosine(optimizer, WARMUP_EP, EPOCHS)

        # ── Training with early stopping on val clip-F1 ──────────
        best_f1, best_state, pat_ctr = 0.0, None, 0
        swa_state, swa_count = None, 0
        SWA_START = max(EPOCHS // 2, WARMUP_EP + 1)

        for epoch in range(1, EPOCHS + 1):
            train_one_fold(model, src_datasets, tgt_trn_loader,
                           optimizer, epochs=1,
                           device=DEVICE, use_bvp=USE_BVP)
            sched.step()

            # SWA accumulate (count only post-SWA_START snapshots)
            if epoch >= SWA_START:
                swa_count += 1
                curr = model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.cpu().float().clone() for k, v in curr.items()}
                else:
                    for k in swa_state:
                        swa_state[k] += (curr[k].cpu().float() - swa_state[k]) / swa_count

            # Validate
            va_acc, _, va_clip_acc, va_clip_f1, _, _, _, _, _ = evaluate(
                model, va_loader, DEVICE, USE_BVP)

            if epoch % 10 == 0 or epoch == 1:
                print(f"  Ep{epoch:3d} | va_win={va_acc:.3f} "
                      f"va_clip={va_clip_acc:.3f} va_f1={va_clip_f1:.3f} "
                      f"lr={sched.get_lr()[0]:.1e}")

            if va_clip_f1 > best_f1:
                best_f1   = va_clip_f1
                best_state= {k: v.cpu().clone()
                              for k, v in model.state_dict().items()}
                pat_ctr   = 0
            else:
                pat_ctr  += 1
                if pat_ctr >= PATIENCE:
                    print(f"  Early stop at epoch {epoch}")
                    break

        # Load SWA or best-val weights
        if swa_state is not None and swa_count >= 3:
            ref = model.state_dict()
            model.load_state_dict({k: swa_state[k].to(ref[k].dtype)
                                    for k in swa_state})
            print(f"  Using SWA weights")
        elif best_state:
            model.load_state_dict(best_state)
        model.to(DEVICE)

        # ── Evaluate on test subject ──────────────────────────────
        w_acc, w_f1, c_acc, c_f1, w_preds, w_lbls, c_preds, c_lbls, clip_pred_map = \
            evaluate(model, tgt_inf_loader, DEVICE, USE_BVP)

        teTK = np.array(te_ev_tkeys)
        teY  = np.array(te_ev_lb)

        for idx_w in range(len(te_ev_lb)):
            loso_win_rows.append(dict(
                subject=loso_sid,
                trial_key=te_ev_tkeys[idx_w],
                window_idx=idx_w,
                true_idx=int(teY[idx_w]),
                true_label=IDX_TO_EMO[int(teY[idx_w])],
                pred_idx=int(w_preds[idx_w]),
                pred_label=IDX_TO_EMO[int(w_preds[idx_w])]))

        for tkey in sorted(set(teTK)):
            m  = teTK == tkey
            tl = int(teY[m][0])
            ci_unique = np.unique(np.array(te_ev_ci)[m])
            pi = clip_pred_map.get(int(ci_unique[0]), tl) if len(ci_unique) else tl
            loso_trial_rows.append(dict(
                subject=loso_sid, trial_key=tkey,
                n_windows=int(m.sum()),
                true_idx=tl, true_label=IDX_TO_EMO[tl],
                trial_pred_idx=pi, trial_pred_label=IDX_TO_EMO[pi]))

        loso_accs.append((loso_sid, w_acc, c_acc, c_f1, len(te_ev_lb)))
        print(f"\n  Subject {loso_sid}: "
              f"win_acc={w_acc:.3f}  clip_acc={c_acc:.3f}  clip_f1={c_f1:.3f}")
        print(classification_report(w_lbls, w_preds,
              target_names=[IDX_TO_EMO[i] for i in range(NUM_CLASSES)],
              zero_division=0, digits=2))
        print(confusion_matrix(w_lbls, w_preds,
              labels=list(range(NUM_CLASSES))))

    # ── 6. Save & summarise ───────────────────────────────────────
    wdf = pd.DataFrame(loso_win_rows)
    tdf = pd.DataFrame(loso_trial_rows)
    wdf.to_csv(os.path.join(OUT_DIR, "bimamba_msmda_loso_window.csv"), index=False)
    tdf.to_csv(os.path.join(OUT_DIR, "bimamba_msmda_loso_trial.csv"),  index=False)

    print(f"\n{'='*65}")
    print("  BiMamba-MSMDA  --  LOSO WINDOW-LEVEL (all subjects pooled)")
    print(f"{'='*65}")
    yt = wdf["true_idx"].astype(int).values
    yp = wdf["pred_idx"].astype(int).values
    print(f"  Window accuracy : {(yt==yp).mean():.4f}")
    print(classification_report(yt, yp,
          target_names=[IDX_TO_EMO[i] for i in range(NUM_CLASSES)],
          zero_division=0))
    print(confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))))

    print(f"\n{'='*65}")
    print("  BiMamba-MSMDA  --  LOSO TRIAL-LEVEL (all subjects pooled)")
    print(f"{'='*65}")
    yt = tdf["true_idx"].astype(int).values
    yp = tdf["trial_pred_idx"].astype(int).values
    print(f"  Trial accuracy  : {(yt==yp).mean():.4f}")
    print(classification_report(yt, yp,
          target_names=[IDX_TO_EMO[i] for i in range(NUM_CLASSES)],
          zero_division=0))
    print(confusion_matrix(yt, yp, labels=list(range(NUM_CLASSES))))

    print(f"\n  Per-subject summary:")
    print(f"  {'Subj':>6} {'Win-Acc':>8} {'Clip-Acc':>9} {'Clip-F1':>8} {'Wins':>5}")
    print(f"  {'-'*44}")
    for sid, wa, ca, cf, n in loso_accs:
        print(f"  {sid:>6}   {wa:.3f}    {ca:.3f}    {cf:.3f}  {n:>5}")
    mean_wa = float(np.mean([a[1] for a in loso_accs]))
    mean_ca = float(np.mean([a[2] for a in loso_accs]))
    mean_cf = float(np.mean([a[3] for a in loso_accs]))
    std_ca  = float(np.std( [a[2] for a in loso_accs]))
    print(f"\n  Mean  win_acc ={mean_wa:.4f}")
    print(f"  Mean clip_acc ={mean_ca:.4f} ± {std_ca:.4f}  ← KEY METRIC")
    print(f"  Mean clip_f1  ={mean_cf:.4f}")
    print(f"  Chance        = {100/NUM_CLASSES:.1f}%")


if __name__ == "__main__":
    main()
