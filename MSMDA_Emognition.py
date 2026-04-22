# ================================================================
# MS-MDA for Emognition Dataset -- LOSO Cross-Subject Evaluation
# ================================================================
# Reference:
#   Chen H, Jin M, Li Z, et al.
#   MS-MDA: Multisource marginal distribution adaptation for
#   cross-subject and cross-session EEG emotion recognition.
#   Frontiers in Neuroscience, 2021, 15: 778488.
#
# Leakage prevention:
#   1. EA: test subject R built from OTHER subjects only
#   2. VT + z-score fit on training subjects only
#   3. No augmented windows in training (MixUp could blend test subject)
#   4. Target windows passed to model UNLABELLED (domain adapt only)
#   5. No global hyperparameter tuning that sees test subjects
# ================================================================

import os, json, glob, time, warnings, math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.signal import welch, coherence as sp_coherence, find_peaks, butter, filtfilt
from scipy.stats import skew, kurtosis
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import VarianceThreshold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler

warnings.filterwarnings("ignore")
np.random.seed(42)
torch.manual_seed(42)

# ================================================================
# CONFIG  -- edit these for your environment
# ================================================================
EMOGNITION_ROOT = "/kaggle/input/datasets/ruchiabey/emognitioncleaned-combined"
OUT_DIR         = "/kaggle/working"

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 60
BATCH_SIZE = 32       # small -- Emognition has ~45 wins/subject
LR         = 1e-3
GAMMA_MAX  = 1.0      # max MMD weight (ramps up via schedule)
BETA_DIV   = 100.0    # disc_loss weight = gamma / BETA_DIV

# ================================================================
# CONSTANTS  (identical to MaxEffort.py)
# ================================================================
EEG_SR           = 256
BAND_SR          = 10
TRAIN_BAND_STORED_SR = 256
TRAIN_BVP_SR     = 20
WINDOW_SEC       = 10
OVERLAP_FRAC     = 0.75
EEG_WIN          = WINDOW_SEC * EEG_SR
BAND_WIN         = WINDOW_SEC * BAND_SR
TRAIN_BVP_WIN    = int(WINDOW_SEC * TRAIN_BVP_SR)
BASELINE_EPS     = 1e-12
NUM_CLASSES      = 4

EMOTION_LABELS = {"NEUTRAL": 0, "ENTHUSIASM": 1, "SADNESS": 2, "FEAR": 3}
IDX_TO_LABEL   = {v: k for k, v in EMOTION_LABELS.items()}

EEG_CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
BAND_CHANNELS = [
    "Alpha_TP9","Alpha_AF7","Alpha_AF8","Alpha_TP10",
    "Beta_TP9", "Beta_AF7", "Beta_AF8", "Beta_TP10",
    "Delta_TP9","Delta_AF7","Delta_AF8","Delta_TP10",
    "Gamma_TP9","Gamma_AF7","Gamma_AF8","Gamma_TP10",
    "Theta_TP9","Theta_AF7","Theta_AF8","Theta_TP10",
]

N_FEAT_EEG  = 156
N_FEAT_MUSE = 62
N_FEAT_BVP  = 7
N_FEAT_HR   = 5
N_FEAT_PPI  = 8
N_FEAT_FAA  = 14
N_FEAT_RIEM = 10
N_FEATURES_RAW = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA + N_FEAT_RIEM
RIEM_START  = N_FEAT_EEG + N_FEAT_MUSE + N_FEAT_BVP + N_FEAT_HR + N_FEAT_PPI + N_FEAT_FAA

# ================================================================
# HELPERS
# ================================================================
def safe_array(x):
    return np.nan_to_num(np.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)

def _interp_nan(a):
    a = a.copy(); nans = np.isnan(a)
    if not nans.any(): return a
    idx = np.arange(len(a))
    a[nans] = np.interp(idx[nans], idx[~nans], a[~nans])
    return a

def resample_1d(arr, orig_sr, target_sr):
    arr = safe_array(np.asarray(arr, dtype=np.float32))
    if len(arr) == 0: return arr
    old_t = np.arange(len(arr)) / float(orig_sr)
    new_len = max(1, int(np.floor(len(arr) * target_sr / orig_sr)))
    new_t = np.arange(new_len) / float(target_sr)
    return np.interp(new_t, old_t, arr).astype(np.float32)

def bandpass_bvp(sig, sr, low=0.7, high=3.5, order=2):
    sig = np.asarray(sig, dtype=np.float64)
    if len(sig) < max(10, order * 3): return sig
    nyq = 0.5 * sr
    lo, hi = max(low/nyq, 1e-5), min(high/nyq, 0.999)
    if lo >= hi: return sig
    try:
        b, a = butter(order, [lo, hi], btype="band")
        return filtfilt(b, a, sig)
    except: return sig

def derive_hr_ppi(bvp, sr):
    sig = safe_array(np.asarray(bvp, dtype=np.float64))
    if len(sig) < 8: return np.array([],np.float32), np.array([],np.float32)
    sf = bandpass_bvp(sig, sr)
    std = np.std(sf)
    sn  = (sf - np.mean(sf)) / std if std > 1e-10 else sf - np.mean(sf)
    try: peaks, _ = find_peaks(sn, distance=max(1,int(0.35*sr)), prominence=0.2)
    except: return np.array([],np.float32), np.array([],np.float32)
    if len(peaks) < 2: return np.array([],np.float32), np.array([],np.float32)
    ibi = np.diff(peaks)/float(sr); ibi = ibi[(ibi>=0.35)&(ibi<=1.5)]
    if len(ibi) == 0: return np.array([],np.float32), np.array([],np.float32)
    return (60./ibi).astype(np.float32), (ibi*1000.).astype(np.float32)

# ── Feature extractors (identical to MaxEffort.py) ──────────────
FREQ_BANDS = [(1,4),(4,8),(8,13),(13,30),(30,100)]
CH_PAIRS   = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]

def _compute_psd(sig, sr): return welch(sig, fs=sr, nperseg=min(256,len(sig)))
def _band_power(f,p,lo,hi): m=(f>=lo)&(f<hi); return np.mean(p[m]) if m.any() else 1e-10
def _hjorth(sig):
    d1=np.diff(sig); d2=np.diff(d1); act=np.var(sig)
    if act<1e-12: return 0.,0.,0.
    mob=np.sqrt(np.var(d1)/act); vd1=np.var(d1)
    if vd1<1e-12: return float(act),float(mob),0.
    comp=np.sqrt(np.var(d2)/vd1)/mob if mob>1e-12 else 0.
    return float(act),float(mob),float(comp)
def _zcr(sig): return float(np.sum(np.abs(np.diff(np.sign(sig)))>0))/max(len(sig)-1,1)
def _de(sig): v=np.var(sig); return 0.5*np.log(2*np.pi*np.e*v) if v>1e-12 else 0.
def _se(sig,sr):
    _,p=welch(sig,fs=sr,nperseg=min(256,len(sig))); pn=p/(p.sum()+1e-12); pn=pn[pn>0]
    return float(-np.sum(pn*np.log2(pn))) if len(pn) else 0.
def _pe(sig,order=3,delay=1):
    n=len(sig)
    if n<(order-1)*delay+1: return 0.
    idx=np.arange(n-(order-1)*delay)
    cols=np.column_stack([sig[idx+d*delay] for d in range(order)])
    perms=np.argsort(cols,axis=1); enc=np.zeros(perms.shape[0],dtype=np.int64)
    for i in range(order): enc=enc*order+perms[:,i]
    _,counts=np.unique(enc,return_counts=True); pr=counts/counts.sum()
    return float(-np.sum(pr*np.log2(pr)))
def _wave(sig,wavelet="db4",level=5):
    try:
        import pywt; ml=pywt.dwt_max_level(len(sig),wavelet)
        c=pywt.wavedec(sig,wavelet,level=min(level,ml))
        e=[float(np.mean(x**2)) for x in c[:5]]
        while len(e)<5: e.append(0.)
        return e
    except: return [0.]*5

def extract_eeg_features(eeg_4ch, sr=EEG_SR):
    feats=[]; bp=np.zeros((4,5))
    for ch in range(4):
        sig=eeg_4ch[ch].astype(np.float64); f,p=_compute_psd(sig,sr)
        for bi,(lo,hi) in enumerate(FREQ_BANDS): bp[ch,bi]=np.log1p(_band_power(f,p,lo,hi))
        feats.extend(bp[ch].tolist())
        for lo,hi in FREQ_BANDS:
            pw=_band_power(f,p,lo,hi); feats.append(0.5*np.log(2*np.pi*np.e*pw) if pw>1e-12 else 0.)
        feats.extend(_hjorth(sig))
        feats.extend([float(np.mean(sig)),float(np.std(sig)),float(skew(sig)),float(kurtosis(sig))])
        feats.extend([_zcr(sig),_de(sig),_se(sig,sr),_pe(sig)])
    for ch in range(4): feats.extend(_wave(eeg_4ch[ch].astype(np.float64)))
    for ch in range(4):
        a,b,t=bp[ch,2],bp[ch,3],bp[ch,1]; feats.extend([a-b,a-t,t-b])
    for bi in range(5): feats.append(bp[2,bi]-bp[1,bi])
    for bi in range(5): feats.append(bp[3,bi]-bp[0,bi])
    for ci,cj in CH_PAIRS:
        try:
            fc,coh=sp_coherence(eeg_4ch[ci].astype(np.float64),eeg_4ch[cj].astype(np.float64),
                                fs=sr,nperseg=min(256,len(eeg_4ch[ci])))
            for lo,hi in FREQ_BANDS:
                m=(fc>=lo)&(fc<hi); feats.append(float(np.mean(coh[m])) if m.any() else 0.)
        except: feats.extend([0.]*5)
    return safe_array(np.array(feats,dtype=np.float32))

def extract_band_features(bw):
    feats=[]
    for ch in range(20):
        col=bw[ch]; col=col[np.isfinite(col)]
        feats.extend([0.,0.] if len(col)==0 else [float(np.mean(col)),float(np.std(col))+1e-8])
    def bp(bi,ei):
        col=bw[bi*4+ei]; col=col[np.isfinite(col)]; return float(np.mean(col)) if len(col) else 0.
    for ei in range(4):
        a,b,t=bp(0,ei),bp(1,ei),bp(4,ei)
        feats.extend([a/(b if abs(b)>1e-6 else 1e-6), a/(t if abs(t)>1e-6 else 1e-6), t/(b if abs(b)>1e-6 else 1e-6)])
    for bi in range(5): feats.append(bp(bi,2)-bp(bi,1))
    for bi in range(5): feats.append(bp(bi,3)-bp(bi,0))
    return safe_array(np.clip(np.array(feats,dtype=np.float32),-1e4,1e4))

def extract_bvp_features(bw, sr):
    sig=bw.astype(np.float64)
    feats=[float(np.mean(sig)),float(np.std(sig)),float(skew(sig)),float(kurtosis(sig))]
    fm=np.abs(np.fft.rfft(sig)); fr=np.fft.rfftfreq(len(sig),1./sr)
    mhr=(fr>0.5)&(fr<4.)
    feats.append(float(fr[mhr][np.argmax(fm[mhr])]) if mhr.any() and fm[mhr].max()>0 else 0.)
    feats.append(_zcr(sig))
    feats.append(float(np.sum(fr*fm)/fm.sum()) if len(fm)>1 and fm.sum()>0 else 0.)
    return safe_array(np.array(feats,dtype=np.float32))

def extract_hr_features(hr):
    if len(hr)<2: return np.zeros(5,dtype=np.float32)
    h=hr.astype(np.float64)
    return safe_array(np.array([np.mean(h),np.std(h),np.min(h),np.max(h),np.max(h)-np.min(h)],dtype=np.float32))

def extract_ppi_features(ppi):
    if len(ppi)<3: return np.zeros(8,dtype=np.float32)
    ip=ppi.astype(np.float64); ips=ip/1000. if np.median(ip)>10 else ip.copy()
    feats=[float(np.mean(ips)),float(np.std(ips))]; sd=np.diff(ips)
    feats.append(float(np.sqrt(np.mean(sd**2))) if len(sd) else 0.)
    feats.append(float(np.mean(np.abs(sd)>0.05)) if len(sd) else 0.)
    feats.append(float(np.std(sd)) if len(sd) else 0.)
    if len(ips)>6:
        tc=np.cumsum(ips); tu=np.arange(tc[0],tc[-1],0.25)
        if len(tu)>8:
            iu=np.interp(tu,tc,ips); fi,pi=welch(iu,fs=4.,nperseg=min(len(iu),32))
            lf=float(np.mean(pi[(fi>=0.04)&(fi<0.15)])) if ((fi>=0.04)&(fi<0.15)).any() else 0.
            hf=float(np.mean(pi[(fi>=0.15)&(fi<0.4)]))  if ((fi>=0.15)&(fi<0.4)).any()  else 0.
            feats.extend([lf,hf,lf/hf if hf>1e-10 else 0.]); return safe_array(np.array(feats,dtype=np.float32))
    feats.extend([0.,0.,0.]); return safe_array(np.array(feats,dtype=np.float32))

def extract_faa_features(eeg_4ch, sr=EEG_SR):
    ALPHA=(8,13);BETA=(13,30);THETA=(4,8);GAMMA=(30,45);DELTA=(1,4)
    def lb(sig,lo,hi):
        f,p=_compute_psd(sig,sr); return float(np.log(_band_power(f,p,lo,hi)+1e-12))
    feats=[lb(eeg_4ch[2],*ALPHA)-lb(eeg_4ch[1],*ALPHA),
           lb(eeg_4ch[2],*BETA) -lb(eeg_4ch[1],*BETA),
           lb(eeg_4ch[2],*THETA)-lb(eeg_4ch[1],*THETA),
           lb(eeg_4ch[2],*GAMMA)-lb(eeg_4ch[1],*GAMMA)]
    for lo,hi in [ALPHA,BETA,THETA,GAMMA,DELTA]: feats.append(lb(eeg_4ch[3],lo,hi)-lb(eeg_4ch[0],lo,hi))
    a7=lb(eeg_4ch[1],*ALPHA);a8=lb(eeg_4ch[2],*ALPHA);a9=lb(eeg_4ch[0],*ALPHA);a10=lb(eeg_4ch[3],*ALPHA)
    feats.append((a8-a7)/(abs(a8+a7)+1e-8)); feats.append((a10-a9)/(abs(a10+a9)+1e-8))
    b7=lb(eeg_4ch[1],*BETA);b8=lb(eeg_4ch[2],*BETA)
    feats.append(a7/(a7+b7+1e-8)); feats.append(a8/(a8+b8+1e-8))
    t7=lb(eeg_4ch[1],*THETA);t8=lb(eeg_4ch[2],*THETA)
    feats.append(((t7+t8)/2)/((a7+a8)/2+1e-8))
    return safe_array(np.array(feats,dtype=np.float32))

# ── Riemannian / EA helpers ─────────────────────────────────────
def _sym_logm(M):
    v,U=np.linalg.eigh(M); return U@np.diag(np.log(np.maximum(v,1e-10)))@U.T

def _reg_cov(X, reg=1e-4):
    C,T=X.shape; Xc=X-X.mean(axis=1,keepdims=True)
    cov=(Xc@Xc.T)/(T-1); return (1-reg)*cov+reg*(np.trace(cov)/C)*np.eye(C)

def _sqrt_inv(M):
    v,U=np.linalg.eigh(M); return U@np.diag(1./np.sqrt(np.maximum(v,1e-10)))@U.T

def extract_riem_features(eeg_4ch):
    cov=_reg_cov(eeg_4ch.astype(np.float64)); logC=_sym_logm(cov)
    return safe_array(logC[np.triu_indices(4)].astype(np.float32))

def trial_vote(probs):
    probs=np.asarray(probs,dtype=np.float64); w=probs.max(axis=1)
    if w.sum()<1e-10: w=np.ones(len(probs))
    w/=w.sum(); mp=(probs*w[:,None]).sum(axis=0)
    return int(np.argmax(mp)), float(mp.max()), mp

# ================================================================
# MS-MDA MODEL  (adapted from Chen et al. 2021)
# ================================================================
class CFE(nn.Module):
    """Common Feature Extractor -- shared across all source domains."""
    def __init__(self, dim_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, 128), nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(128, 64),    nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.net(x)

class DSFE(nn.Module):
    """Domain-Specific Feature Extractor -- one per source subject."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32, eps=1e-5, momentum=0.1),
            nn.LeakyReLU(0.01, inplace=True),
        )
    def forward(self, x): return self.net(x)

def mmd_linear(fx, fy):
    """Linear MMD between two batches."""
    delta = fx - fy
    return torch.mean(torch.mm(delta, delta.t()))

class MSMDA(nn.Module):
    def __init__(self, dim_in, num_classes, n_sources):
        super().__init__()
        self.n_sources  = n_sources
        self.cfe        = CFE(dim_in)
        # create one DSFE and one classifier head per source subject
        self.dsfe_list  = nn.ModuleList([DSFE() for _ in range(n_sources)])
        self.cls_list   = nn.ModuleList([nn.Linear(32, num_classes) for _ in range(n_sources)])

    def forward(self, src_data, src_label, tgt_data, mark):
        """Training forward -- returns (cls_loss, mmd_loss, disc_loss)."""
        # shared feature extraction
        src_cfe = self.cfe(src_data)
        tgt_cfe = self.cfe(tgt_data)

        # extract target features through ALL DSFEs
        tgt_dsfe = [self.dsfe_list[i](tgt_cfe) for i in range(self.n_sources)]

        # extract source features through its own DSFE
        src_dsfe = self.dsfe_list[mark](src_cfe)

        # MMD between source domain `mark` and target
        mmd_loss = mmd_linear(src_dsfe, tgt_dsfe[mark])

        # discrepancy loss: keep target representations consistent across DSFEs
        disc_loss = sum(
            torch.mean(torch.abs(
                F.softmax(tgt_dsfe[mark], dim=1) -
                F.softmax(tgt_dsfe[i],    dim=1)
            ))
            for i in range(self.n_sources) if i != mark
        )

        # classification loss on labelled source
        pred = self.cls_list[mark](src_dsfe)
        cls_loss = F.cross_entropy(pred, src_label.long())

        return cls_loss, mmd_loss, disc_loss

    @torch.no_grad()
    def predict(self, x):
        """Inference: average softmax over all DSFE heads."""
        self.eval()
        cfe_out = self.cfe(x)
        preds   = [F.softmax(self.cls_list[i](self.dsfe_list[i](cfe_out)), dim=1)
                   for i in range(self.n_sources)]
        return torch.stack(preds).mean(dim=0)   # (N, num_classes)

# ================================================================
# DATA LOADING  (same pipeline as MaxEffort.py)
# ================================================================
def load_subject_baselines(root):
    baselines = {}
    for subj in sorted(os.listdir(root)):
        d = os.path.join(root, subj)
        if not os.path.isdir(d) or not subj.isdigit(): continue
        cands = sorted(glob.glob(os.path.join(d, "*_BASELINE_MUSE_cleaned.json")))
        if not cands:
            cands = [f for f in glob.glob(os.path.join(d,"*.json"))
                     if "baseline" in f.lower() and "muse" in f.lower()]
        if not cands: continue
        try:
            with open(cands[0]) as fh: raw = json.load(fh)
            df = pd.DataFrame(raw["data"] if isinstance(raw,dict) and "data" in raw
                              else (raw if isinstance(raw,list) else raw))
            cols = [c for c in EEG_CHANNELS if c in df.columns]
            if not cols: continue
            sig = np.stack([_interp_nan(np.nan_to_num(df[c].to_numpy(dtype=np.float64)))
                            for c in cols], axis=-1)
            bm = np.mean(sig,axis=0).astype(np.float32)
            bs = np.std(sig, axis=0).astype(np.float32)
            bs = np.where(bs<1e-6, 1., bs)
            baselines[subj] = (bm, bs)
        except: pass
    return baselines

def load_and_window_all(root):
    """Load all trials and extract features. Returns raw feature matrix + metadata."""
    baselines   = load_subject_baselines(root)
    glob_base   = None
    if baselines:
        glob_base = (np.mean([v[0] for v in baselines.values()],axis=0),
                     np.mean([v[1] for v in baselines.values()],axis=0))

    trials, step = [], WINDOW_SEC*(1-OVERLAP_FRAC)
    for subj in sorted(os.listdir(root)):
        d = os.path.join(root, subj)
        if not os.path.isdir(d) or not subj.isdigit(): continue
        for ef in sorted(glob.glob(os.path.join(d,"*_STIMULUS_MUSE_cleaned.json"))):
            parts  = os.path.basename(ef).split("_")
            emo    = parts[1].upper() if len(parts)>=2 else None
            if emo not in EMOTION_LABELS: continue
            try:
                with open(ef) as f: ed=json.load(f)
                eeg=np.stack([np.array(ed[ch],dtype=np.float32) for ch in EEG_CHANNELS])
                eeg=safe_array(eeg)
            except: continue
            bl = baselines.get(subj, glob_base)
            if bl is not None:
                bm,bs = bl; eeg = safe_array(((eeg.T-bm)/bs).T.astype(np.float32))
            mf = os.path.join(root,subj,f"{subj}_{emo}_STIMULUS_MUSE.json")
            if not os.path.isfile(mf): continue
            try:
                with open(mf) as f: md=json.load(f)
                bl2=[resample_1d(safe_array(np.array(md[b],dtype=np.float32)),
                                  TRAIN_BAND_STORED_SR,BAND_SR) for b in BAND_CHANNELS]
                ml=min(len(x) for x in bl2); band=np.stack([x[:ml] for x in bl2])
            except: continue
            wf=os.path.join(root,subj,f"{subj}_{emo}_STIMULUS_SAMSUNG_WATCH.json")
            if not os.path.isfile(wf): continue
            try:
                with open(wf) as f: sw=json.load(f)
                if "BVPProcessed" not in sw: continue
                bvp=safe_array(np.array([r[1] for r in sw["BVPProcessed"]],dtype=np.float32))
            except: continue
            dur=min(eeg.shape[1]/EEG_SR,band.shape[1]/BAND_SR,len(bvp)/TRAIN_BVP_SR)
            eeg=eeg[:,:int(dur*EEG_SR)]; band=band[:,:int(dur*BAND_SR)]; bvp=bvp[:int(dur*TRAIN_BVP_SR)]
            trials.append(dict(sid=subj,emotion=emo,label=EMOTION_LABELS[emo],
                               eeg=eeg,band=band,bvp=bvp,
                               tkey=f"{subj}_{emo}"))

    feats,labels,tkeys,sids,eeg_wins=[],[],[],[],[]
    for tr in trials:
        e,b,bv=tr["eeg"],tr["band"],tr["bvp"]
        dur=min(e.shape[1]/EEG_SR,b.shape[1]/BAND_SR,len(bv)/TRAIN_BVP_SR)
        nw=int((dur-WINDOW_SEC)/step)+1
        if nw<=0: continue
        for wi in range(nw):
            es=int(wi*step*EEG_SR);     ee=es+EEG_WIN
            ms=int(wi*step*BAND_SR);    me=ms+BAND_WIN
            bs=int(wi*step*TRAIN_BVP_SR); be=bs+TRAIN_BVP_WIN
            if ee>e.shape[1] or me>b.shape[1] or be>len(bv): break
            ew=e[:,es:ee]; mw=b[:,ms:me]; bw=bv[bs:be]
            hr,ppi=derive_hr_ppi(bw,TRAIN_BVP_SR)
            feat=np.concatenate([
                extract_eeg_features(ew),
                extract_band_features(mw),
                extract_bvp_features(bw,TRAIN_BVP_SR),
                extract_hr_features(hr),
                extract_ppi_features(ppi),
                extract_faa_features(ew),
                extract_riem_features(ew),   # will be re-aligned per LOSO fold
            ]).astype(np.float32)
            feats.append(feat); labels.append(tr["label"]); tkeys.append(tr["tkey"])
            sids.append(tr["sid"]); eeg_wins.append(ew)

    return (np.array(feats,dtype=np.float32),
            np.array(labels), np.array(tkeys),
            np.array(sids),   eeg_wins,
            trials)

# ================================================================
# LOSO TRAINING
# ================================================================
def train_loso_fold(model, src_datasets, tgt_loader, optimizer, epochs, device):
    """Train MS-MDA for one LOSO fold. Target data is UNLABELLED."""
    n_src   = len(src_datasets)
    src_loaders = [DataLoader(ds, sampler=RandomSampler(ds),
                              batch_size=BATCH_SIZE, drop_last=True)
                   for ds in src_datasets]
    src_iters   = [iter(l) for l in src_loaders]
    tgt_iter    = iter(tgt_loader)

    n_iter_per_epoch = max(1, min(len(l) for l in src_loaders))
    total_iters = epochs * n_iter_per_epoch

    model.train()
    for epoch in range(epochs):
        for step_i in range(n_iter_per_epoch):
            for j in range(n_src):
                # get source batch
                try: sx, sy = next(src_iters[j])
                except StopIteration:
                    src_iters[j] = iter(src_loaders[j]); sx, sy = next(src_iters[j])
                # get target batch (unlabelled)
                try: tx, _ = next(tgt_iter)
                except StopIteration:
                    tgt_iter = iter(tgt_loader); tx, _ = next(tgt_iter)

                sx, sy, tx = sx.to(device), sy.to(device), tx.to(device)

                optimizer.zero_grad()
                cls_l, mmd_l, disc_l = model(sx, sy, tx, mark=j)
                # progressive ramp-up of domain loss weight (Ganin et al.)
                global_step = epoch * n_iter_per_epoch + step_i
                gamma = GAMMA_MAX * (2/(1+math.exp(-10*global_step/total_iters))-1)
                beta  = gamma / BETA_DIV
                loss  = cls_l + gamma * mmd_l + beta * disc_l
                loss.backward()
                optimizer.step()

# ================================================================
# MAIN LOSO LOOP
# ================================================================
def main():
    print(f"Device: {DEVICE}")
    print("Loading and windowing data ...")
    t0 = time.time()
    allF_raw, allY, allTK, allSID, eeg_wins, trials = load_and_window_all(EMOGNITION_ROOT)
    print(f"  {len(allY)} windows from {len(set(allSID))} subjects in {time.time()-t0:.1f}s")

    # ── per-subject EA: build R from ALL own windows (no fold split needed
    #    here because LOSO will rebuild R from other subjects anyway)
    print("Pre-computing per-subject mean covariances for cross-subject EA ...")
    _sid_mean_cov = {}
    _sid_to_gidxs = defaultdict(list)
    for gi, sid in enumerate(allSID):
        _sid_to_gidxs[sid].append(gi)
    for sid, gidxs in _sid_to_gidxs.items():
        covs = [_reg_cov(eeg_wins[gi].astype(np.float64)) for gi in gidxs]
        _sid_mean_cov[sid] = np.stack(covs).mean(axis=0)

    all_sids = sorted(set(allSID))
    loso_win_rows, loso_trial_rows, loso_accs = [], [], []

    for loso_sid in all_sids:
        te_mask = (allSID == loso_sid)
        tr_mask = (allSID != loso_sid)
        if te_mask.sum() == 0 or tr_mask.sum() == 0:
            continue

        print(f"\n{'='*60}")
        print(f"LOSO subject: {loso_sid}  "
              f"(train={tr_mask.sum()} wins, test={te_mask.sum()} wins)")

        # ── EA FIX: patch test-subject Riem features using cross-subject R ──
        # R = mean of all OTHER subjects' mean covariances (zero self-leakage)
        other_covs = [_sid_mean_cov[s] for s in _sid_mean_cov if s != loso_sid]
        R_cross    = np.stack(other_covs).mean(axis=0)
        Rinv_cross = _sqrt_inv(R_cross)
        allF_raw_fold = allF_raw.copy()   # work on a copy per fold
        for gi in np.where(te_mask)[0]:
            aligned = Rinv_cross @ eeg_wins[gi].astype(np.float64)
            allF_raw_fold[gi, RIEM_START:RIEM_START+N_FEAT_RIEM] = \
                extract_riem_features(np.array(aligned, dtype=np.float32))

        # ── VT + z-score fit on training subjects ONLY ──────────────────────
        vt = VarianceThreshold(threshold=0.001)
        vt.fit(allF_raw_fold[tr_mask])
        trF = vt.transform(allF_raw_fold[tr_mask])
        teF = vt.transform(allF_raw_fold[te_mask])
        mu  = trF.mean(axis=0)
        sd  = np.where(trF.std(axis=0) < 1e-8, 1., trF.std(axis=0))
        trF = np.clip((trF - mu) / sd, -10, 10)
        teF = np.clip((teF - mu) / sd, -10, 10)
        dim = trF.shape[1]

        trY   = allY[tr_mask]
        teY   = allY[te_mask]
        teTK  = allTK[te_mask]
        tr_sids = allSID[tr_mask]

        # ── Build per-source-subject datasets ───────────────────────────────
        # Each source subject is a separate domain in MS-MDA.
        # NO augmented windows are used (avoids MixUp cross-subject blending).
        src_subjects = sorted(set(tr_sids))
        src_datasets = []
        for ssid in src_subjects:
            sm = (tr_sids == ssid)
            if sm.sum() == 0: continue
            sx = torch.tensor(trF[sm], dtype=torch.float32)
            sy = torch.tensor(trY[sm], dtype=torch.long)
            src_datasets.append(TensorDataset(sx, sy))

        n_sources = len(src_datasets)
        if n_sources == 0:
            print(f"  [{loso_sid}] No source subjects -- skipping"); continue

        # ── Target dataset: test subject windows, labels hidden ──────────────
        # Labels are passed as zeros (ignored during training).
        tgt_ds     = TensorDataset(
            torch.tensor(teF, dtype=torch.float32),
            torch.zeros(len(teF), dtype=torch.long))   # dummy labels
        tgt_loader = DataLoader(tgt_ds,
                                sampler=SequentialSampler(tgt_ds),
                                batch_size=BATCH_SIZE, drop_last=False)
        # Larger batch for domain adaptation (needs repeated sampling)
        tgt_loader_train = DataLoader(tgt_ds,
                                      sampler=RandomSampler(tgt_ds),
                                      batch_size=max(BATCH_SIZE, len(teF)//2+1),
                                      drop_last=True)

        # ── Build model ──────────────────────────────────────────────────────
        model     = MSMDA(dim_in=dim, num_classes=NUM_CLASSES,
                          n_sources=n_sources).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        # ── Train ────────────────────────────────────────────────────────────
        train_loso_fold(model, src_datasets, tgt_loader_train,
                        optimizer, EPOCHS, DEVICE)

        # ── Evaluate ─────────────────────────────────────────────────────────
        model.eval()
        all_probs = []
        with torch.no_grad():
            for xb, _ in tgt_loader:
                p = model.predict(xb.to(DEVICE)).cpu().numpy()
                all_probs.append(p)
        all_probs = np.vstack(all_probs)   # (n_test_wins, 4)

        # ── EMA smoothing over consecutive windows ───────────────────────────
        def ema(probs, alpha=0.4):
            out = probs.copy()
            for t in range(1, len(out)):
                out[t] = alpha*probs[t] + (1-alpha)*out[t-1]
            return out

        win_correct = []
        for tkey in sorted(set(teTK)):
            m   = (teTK == tkey)
            pr  = ema(all_probs[m])
            pd_ = np.argmax(pr, axis=1)
            tl  = int(teY[m][0])
            win_correct.extend((pd_ == teY[m]).tolist())
            for r in range(len(pd_)):
                loso_win_rows.append(dict(
                    subject=loso_sid, trial_key=tkey, window_idx=r,
                    true_idx=tl, true_label=IDX_TO_LABEL[tl],
                    pred_idx=int(pd_[r]), pred_label=IDX_TO_LABEL[int(pd_[r])]))
            pi, conf, mp = trial_vote(pr)
            loso_trial_rows.append(dict(
                subject=loso_sid, trial_key=tkey, n_windows=int(m.sum()),
                true_idx=tl, true_label=IDX_TO_LABEL[tl],
                trial_pred_idx=pi, trial_pred_label=IDX_TO_LABEL[pi],
                trial_confidence=conf))

        w_acc = float(np.mean(win_correct))
        loso_accs.append((loso_sid, w_acc, int(te_mask.sum())))
        print(f"  win_acc={w_acc:.3f}  n_wins={int(te_mask.sum())}")
        yt=[r["true_idx"] for r in loso_win_rows if r["subject"]==loso_sid]
        yp=[r["pred_idx"] for r in loso_win_rows if r["subject"]==loso_sid]
        print(classification_report(yt,yp,
              target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],
              zero_division=0, digits=2))
        print(confusion_matrix(yt,yp,labels=list(range(NUM_CLASSES))))

    # ── Save results ─────────────────────────────────────────────────────────
    wdf = pd.DataFrame(loso_win_rows)
    tdf = pd.DataFrame(loso_trial_rows)
    wdf.to_csv(os.path.join(OUT_DIR,"msmda_loso_window.csv"), index=False)
    tdf.to_csv(os.path.join(OUT_DIR,"msmda_loso_trial.csv"),  index=False)

    print("\n" + "="*60)
    print("MS-MDA  --  LOSO WINDOW-LEVEL (all subjects pooled)")
    print("="*60)
    yt=wdf["true_idx"].astype(int).values; yp=wdf["pred_idx"].astype(int).values
    print(f"Window accuracy: {(yt==yp).mean():.4f}")
    print(classification_report(yt,yp,
          target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],zero_division=0))
    print(confusion_matrix(yt,yp,labels=list(range(NUM_CLASSES))))

    print("\n" + "="*60)
    print("MS-MDA  --  LOSO TRIAL-LEVEL (all subjects pooled)")
    print("="*60)
    yt=tdf["true_idx"].astype(int).values; yp=tdf["trial_pred_idx"].astype(int).values
    print(f"Trial accuracy: {(yt==yp).mean():.4f}")
    print(classification_report(yt,yp,
          target_names=[IDX_TO_LABEL[i] for i in range(NUM_CLASSES)],zero_division=0))
    print(confusion_matrix(yt,yp,labels=list(range(NUM_CLASSES))))

    print("\nPer-subject window accuracies:")
    for sid,acc,n in loso_accs:
        print(f"  {sid}: {acc:.3f}  ({n} windows)")
    mean_l=float(np.mean([a for _,a,_ in loso_accs]))
    std_l =float(np.std( [a for _,a,_ in loso_accs]))
    print(f"\nMean LOSO window accuracy: {mean_l:.4f} +/- {std_l:.4f}")

if __name__ == "__main__":
    main()
